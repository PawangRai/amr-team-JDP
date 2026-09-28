import math
import random
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid


class FrontierExplorationNode(Node):
    """
    Frontier-based exploration, combined with SLAM (slam_toolbox builds the
    map; this node only decides WHERE to send the robot next).

    Pipeline:
      1. Subscribe to the live, growing /map published by slam_toolbox
         (NOT the static map_server map used in Parts 1-2).
      2. Detect frontier cells: free cells (value 0) that are adjacent to
         at least one unknown cell (value -1) -- i.e. the boundary between
         explored and unexplored space, as the brief asks for.
      3. Cluster frontier cells into connected regions (flood fill) and
         discard tiny/noisy clusters.
      4. Score each cluster's centroid by distance from the robot (nearest-
         frontier strategy) with a bonus for larger clusters, and pick the
         best one.
      5. Publish that pose to /goal_pose -- the SAME topic astar_global_planner
         already subscribes to, so the existing A* + potential field pipeline
         from Part 1 handles all the actual navigation, unmodified.
      6. Monitor progress via TF; when the robot gets close to the frontier
         goal, or a timeout elapses (stuck/unreachable frontier), pick the
         next one.
      7. When no frontier clusters remain, exploration is complete.

    This node deliberately does NOT reimplement navigation -- it only
    selects poses, per the brief's separation of "exploration component
    selects poses" / "SLAM component creates the map".
    """

    def __init__(self):
        super().__init__('frontier_exploration')
        self.get_logger().info("Frontier Exploration Started")

        # ---------------- Parameters ----------------
        self.min_frontier_size   = 6      # cells; discard smaller clusters as noise
        self.size_bonus_weight   = 0.02   # how much cluster size discounts effective distance
        self.goal_arrival_radius = 0.4    # metres -- close enough to call a frontier "reached"
        self.goal_timeout_sec    = 45.0   # give up on an unreachable/stuck frontier after this
        self.replan_period_sec   = 2.0    # how often to check progress / look for new frontiers
        self.unreachable_blacklist_radius = 0.5  # metres -- avoid re-picking a failed frontier

        # MUST match (or be >=) astar_global_planner's inflation_radius_m, so a
        # frontier goal we publish is never one that A* will immediately reject
        # as "occupied/inflated" -- this was the actual bug: in a narrow
        # corridor, a raw frontier centroid very often lands inside the wall's
        # inflated safety margin, A* silently fails to find a path, and the
        # robot never moves even though a frontier goal was published.
        self.inflation_radius_m = 0.35
        self.occupied_thresh    = 50

        # Pose-selection strategy: instead of deterministically always
        # picking the single best-scoring frontier (which can fixate on one
        # problematic point), sample randomly among the top-K candidates
        # each cycle, weighted toward the better-scoring ones. This is a
        # legitimate alternative exploration strategy (the brief explicitly
        # invites experimenting with pose-selection strategies), and it
        # naturally reduces the odds of repeatedly re-targeting the exact
        # same troublesome frontier on consecutive cycles.
        self.top_k_candidates = 5

        self.map_msg   = None
        self.map_array = None

        self.current_goal   = None   # (x, y) currently being pursued, or None
        self.goal_start_time = None
        self.blacklist = []          # list of (x, y) frontiers that timed out -- avoid re-picking

        # Detect "persistent" frontiers -- a frontier cell that keeps getting
        # picked and immediately marked "reached" (within goal_arrival_radius)
        # but never actually resolves into free/occupied space (e.g. it's
        # behind a wall at a grazing laser angle SLAM can never fill in from
        # any reachable pose). Without this, such a frontier loops forever
        # since it never triggers the timeout-based blacklist path.
        self._last_selected_goal = None
        self._repeat_select_count = 0
        self.repeat_select_limit = 3   # blacklist after this many identical picks in a row
        self.repeat_match_tolerance = 0.15  # metres
        self.exploration_done = False

        # slam_toolbox publishes /map with the same latched QoS as map_server
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, map_qos)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(self.replan_period_sec, self.explore_step)

    # ------------------------------------------------------------------
    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg
        w, h = msg.info.width, msg.info.height
        self.map_array = np.array(msg.data, dtype=np.int8).reshape(h, w)
        self.inflated_occupied = self.compute_inflated_mask(self.map_array, msg.info.resolution)

    def compute_inflated_mask(self, grid, resolution):
        """Boolean mask, True where a cell is occupied OR within
        inflation_radius_m of an occupied cell -- must mirror what
        astar_global_planner does, so we never pick a goal it will reject."""
        radius_cells = max(1, int(round(self.inflation_radius_m / resolution)))
        occupied = (grid >= self.occupied_thresh)
        if not occupied.any():
            return occupied

        inflated = occupied.copy()
        ys, xs = np.where(occupied)
        h, w = grid.shape
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - radius_cells), min(h, y + radius_cells + 1)
            x0, x1 = max(0, x - radius_cells), min(w, x + radius_cells + 1)
            inflated[y0:y1, x0:x1] = True
        return inflated

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF (map->base_link): {e}")
            return None
        return t.transform.translation.x, t.transform.translation.y

    def grid_to_world(self, gx, gy):
        res = self.map_msg.info.resolution
        ox = self.map_msg.info.origin.position.x
        oy = self.map_msg.info.origin.position.y
        return ox + (gx + 0.5) * res, oy + (gy + 0.5) * res

    # ------------------------------------------------------------------
    # Frontier detection
    # ------------------------------------------------------------------

    def find_frontier_cells(self):
        """Free cells (0) with at least one unknown (-1) 4-neighbour."""
        grid = self.map_array
        h, w = grid.shape

        free = (grid == 0)
        unknown = (grid == -1)

        # Shift 'unknown' in four directions and OR together to find free
        # cells that border an unknown cell, without a slow python double loop.
        adjacent_unknown = np.zeros_like(free)
        adjacent_unknown[1:, :]  |= unknown[:-1, :]   # unknown above
        adjacent_unknown[:-1, :] |= unknown[1:, :]    # unknown below
        adjacent_unknown[:, 1:]  |= unknown[:, :-1]   # unknown to the left
        adjacent_unknown[:, :-1] |= unknown[:, 1:]    # unknown to the right

        frontier_mask = free & adjacent_unknown
        ys, xs = np.where(frontier_mask)
        return set(zip(xs.tolist(), ys.tolist()))

    def cluster_frontiers(self, frontier_cells):
        """Group frontier cells into connected clusters via BFS (8-connectivity)."""
        remaining = set(frontier_cells)
        clusters = []

        while remaining:
            start = next(iter(remaining))
            cluster = []
            queue = deque([start])
            remaining.discard(start)

            while queue:
                cx, cy = queue.popleft()
                cluster.append((cx, cy))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        n = (cx + dx, cy + dy)
                        if n in remaining:
                            remaining.discard(n)
                            queue.append(n)

            clusters.append(cluster)

        return [c for c in clusters if len(c) >= self.min_frontier_size]

    def pick_valid_cell(self, cluster):
        """
        Return a (gx, gy) cell from this cluster that is NOT inside the
        inflated-obstacle mask A* uses, preferring the cell closest to the
        cluster's centroid. Returns None if every cell in the cluster is
        inflated-occupied (fully swallowed by a nearby wall's safety margin).

        This is the actual fix for the "robot never moves" bug: publishing
        the raw centroid can land inside A*'s inflation zone in a narrow
        corridor, A* silently fails to find a path, and nothing downstream
        ever gets a /global_path to follow.
        """
        cx = sum(c[0] for c in cluster) / len(cluster)
        cy = sum(c[1] for c in cluster) / len(cluster)

        valid_cells = [
            (x, y) for (x, y) in cluster
            if not self.inflated_occupied[y, x]
        ]
        if not valid_cells:
            return None

        # Closest valid cell to the centroid (keeps the goal representative
        # of the frontier region rather than an arbitrary edge cell)
        return min(valid_cells, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)

    def select_best_frontier(self, clusters, robot_x, robot_y):
        """
        Weighted-random selection among the top-K best-scoring candidates,
        rather than always deterministically picking the single best one.

        Rationale: a pure nearest-frontier strategy can fixate on the same
        frontier cycle after cycle if it's a close second-best every time
        conditions change slightly (this is what caused the repeated-
        selection stuck loop). Sampling among the top few candidates, with
        better-scoring ones more likely but not guaranteed, breaks that
        fixation naturally while still generally preferring good choices --
        this is also a legitimate alternative pose-selection strategy in
        its own right (the brief explicitly allows experimenting with these).
        """
        candidates = []  # list of (score, (wx, wy))

        for cluster in clusters:
            valid_cell = self.pick_valid_cell(cluster)
            if valid_cell is None:
                continue  # entire cluster is inside the inflation zone -- skip it

            wx, wy = self.grid_to_world(*valid_cell)

            if self.is_blacklisted(wx, wy):
                continue

            dist = math.hypot(wx - robot_x, wy - robot_y)
            # Larger clusters (more unexplored area revealed) get a discount
            # on effective distance -- nearest-frontier-with-size-bonus scoring.
            score = dist - self.size_bonus_weight * len(cluster)
            candidates.append((score, (wx, wy)))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0])
        top_k = candidates[:self.top_k_candidates]

        # Weight inversely by score (lower score = better = higher weight).
        # Shift scores positive first since score can be negative (the size
        # bonus can push it below zero) and random.choices needs weights >= 0.
        min_score = top_k[0][0]
        weights = [1.0 / (1.0 + (s - min_score)) for s, _ in top_k]

        chosen_score, chosen_pose = random.choices(top_k, weights=weights, k=1)[0]
        return chosen_pose

    def is_blacklisted(self, x, y):
        for bx, by in self.blacklist:
            if math.hypot(x - bx, y - by) < self.unreachable_blacklist_radius:
                return True
        return False

    # ------------------------------------------------------------------
    # Main exploration loop
    # ------------------------------------------------------------------

    def explore_step(self):
        if self.exploration_done or self.map_array is None:
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return
        robot_x, robot_y = robot_pose

        # --- If we're already pursuing a goal, check progress first ---
        if self.current_goal is not None:
            gx, gy = self.current_goal
            dist = math.hypot(gx - robot_x, gy - robot_y)
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self.goal_start_time

            if dist < self.goal_arrival_radius:
                self.get_logger().info(f"Frontier reached ({gx:.2f}, {gy:.2f})")
                self.current_goal = None
            elif elapsed > self.goal_timeout_sec:
                self.get_logger().warn(
                    f"Frontier timed out ({gx:.2f}, {gy:.2f}) -- blacklisting")
                self.blacklist.append((gx, gy))
                self.current_goal = None
            else:
                return  # still en route, nothing to do this tick

        # --- Need a new frontier ---
        frontier_cells = self.find_frontier_cells()
        clusters = self.cluster_frontiers(frontier_cells)

        if not clusters:
            self.get_logger().info(
                "No frontiers remaining -- exploration complete!")
            self.exploration_done = True
            return

        target = self.select_best_frontier(clusters, robot_x, robot_y)
        if target is None:
            self.get_logger().warn(
                "All remaining frontiers are blacklisted -- exploration complete")
            self.exploration_done = True
            return

        # --- Persistent-frontier detection ---
        # If the same location keeps getting picked cycle after cycle (it
        # gets marked "reached" almost instantly because the robot is
        # already right next to it, but the frontier cell never actually
        # resolves into free/occupied space -- e.g. it's behind a wall at
        # a grazing laser angle SLAM can't fill in), blacklist it instead
        # of looping forever. This is separate from the timeout path, which
        # only catches frontiers the robot can never physically reach.
        if (self._last_selected_goal is not None and
                math.hypot(target[0] - self._last_selected_goal[0],
                           target[1] - self._last_selected_goal[1]) < self.repeat_match_tolerance):
            self._repeat_select_count += 1
        else:
            self._repeat_select_count = 0
        self._last_selected_goal = target

        if self._repeat_select_count >= self.repeat_select_limit:
            self.get_logger().warn(
                f"Frontier ({target[0]:.2f}, {target[1]:.2f}) selected "
                f"{self._repeat_select_count + 1} times in a row without "
                f"resolving -- likely unreachable-to-resolve (e.g. behind a "
                f"wall). Blacklisting and picking a different one.")
            self.blacklist.append(target)
            self._repeat_select_count = 0
            self._last_selected_goal = None
            self.current_goal = None
            return  # re-select on the next tick, excluding this one

        self.publish_goal(*target)
        self.current_goal = target
        self.goal_start_time = self.get_clock().now().nanoseconds / 1e9
        self.get_logger().info(
            f"New frontier goal: ({target[0]:.2f}, {target[1]:.2f}), "
            f"{len(clusters)} clusters available")

    def publish_goal(self, x, y):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.w = 1.0   # orientation doesn't matter for exploration waypoints
        self.goal_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
