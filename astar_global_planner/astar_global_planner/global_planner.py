import rclpy
import math
import heapq
import numpy as np
import tf2_ros
from collections import deque

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped


class AStarGlobalPlanner(Node):
    """
    Grid-based A* global planner.

    - Subscribes to the static map (/map, published by map_server / nav2_map_server)
    - Subscribes to a goal (/goal_pose, e.g. published from RViz '2D Nav Goal',
      or from your own goal-publishing script)
    - Looks up the robot's current pose in the map frame via tf (map -> base_link)
    - Runs A* over the inflated occupancy grid
    - Publishes the resulting path as a pruned list of waypoints on /global_path
      (nav_msgs/Path), which the potential field planner subscribes to.
    """

    def __init__(self):
        super().__init__('astar_global_planner')
        self.get_logger().info("A* Global Planner Started")

        # --- Parameters you will likely want to tune ---
        self.inflation_radius_m = 0.35   # ~ robot radius + safety margin, in metres
        self.occupied_thresh    = 50     # occupancy value >= this counts as occupied
        self.waypoint_spacing_m = 0.5     # prune raw A* path to ~1 waypoint every N metres

        self.map_data   = None   # OccupancyGrid msg
        self.map_array  = None   # numpy 2D array, inflated
        self.map_info   = None

        # nav2_map_server publishes /map once as a latched (TRANSIENT_LOCAL)
        # topic and doesn't republish -- a VOLATILE subscriber (e.g. the
        # default sensor-data QoS) only sees messages published after it
        # connects, so it can permanently miss that one-shot publish if it
        # loses the startup race. Match map_server's QoS instead.
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, map_qos)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.path_pub = self.create_publisher(Path, '/global_path', 10)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    # ------------------------------------------------------------------
    # Map handling
    # ------------------------------------------------------------------

    def map_callback(self, msg: OccupancyGrid):
        self.map_data = msg
        self.map_info = msg.info
        width, height = msg.info.width, msg.info.height
        grid = np.array(msg.data, dtype=np.int8).reshape(height, width)
        self.map_array = self.inflate_obstacles(grid)
        self.get_logger().info(
            f"Map received: {width}x{height} @ {msg.info.resolution:.3f} m/cell")

    def inflate_obstacles(self, grid: np.ndarray) -> np.ndarray:
        """Dilate occupied cells by inflation_radius_m so A* keeps the robot
        away from walls by roughly its own radius plus a safety margin."""
        resolution = self.map_info.resolution
        radius_cells = max(1, int(round(self.inflation_radius_m / resolution)))

        occupied = grid >= self.occupied_thresh
        inflated = occupied.copy()

        ys, xs = np.where(occupied)
        h, w = grid.shape
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - radius_cells), min(h, y + radius_cells + 1)
            x0, x1 = max(0, x - radius_cells), min(w, x + radius_cells + 1)
            inflated[y0:y1, x0:x1] = True

        out = grid.copy()
        out[inflated] = 100
        return out

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def world_to_grid(self, x, y):
        res = self.map_info.resolution
        ox, oy = self.map_info.origin.position.x, self.map_info.origin.position.y
        gx = int((x - ox) / res)
        gy = int((y - oy) / res)
        return gx, gy

    def grid_to_world(self, gx, gy):
        res = self.map_info.resolution
        ox, oy = self.map_info.origin.position.x, self.map_info.origin.position.y
        x = ox + (gx + 0.5) * res
        y = oy + (gy + 0.5) * res
        return x, y

    def get_robot_pose_in_map(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF (map->base_link): {e}")
            return None
        return t.transform.translation.x, t.transform.translation.y

    # ------------------------------------------------------------------
    # A* search
    # ------------------------------------------------------------------

    def astar(self, start, goal):
        """start, goal: (gx, gy) grid cells. Returns list of (gx, gy) or None."""
        h, w = self.map_array.shape

        def in_bounds(c):
            return 0 <= c[0] < w and 0 <= c[1] < h

        def is_free(c):
            # OccupancyGrid cells are -1 for unknown (unmapped) space, not
            # just 0-100 free-to-occupied. Checking only "< occupied_thresh"
            # treats -1 as free too, since -1 < 50 -- so A* would happily
            # plan straight-line shortcuts through unexplored territory
            # outside the actual corridor (seen as the path looping outside
            # the mapped area during real-robot exploration). Require the
            # cell to be KNOWN free, not merely not-yet-known-occupied.
            v = self.map_array[c[1], c[0]]
            return 0 <= v < self.occupied_thresh

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]

        if not (in_bounds(start) and in_bounds(goal)):
            self.get_logger().error("Start or goal outside map bounds")
            return None
        if not is_free(goal):
            self.get_logger().error("Goal cell is occupied/inflated")
            return None

        open_set = [(0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for dx, dy in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if not in_bounds(nxt) or not is_free(nxt):
                    continue
                step_cost = math.hypot(dx, dy)
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(nxt, float('inf')):
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    f_score = tentative_g + heuristic(nxt, goal)
                    heapq.heappush(open_set, (f_score, nxt))

        self.get_logger().error("A*: no path found")
        return None

    @staticmethod
    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def prune_path(self, world_path):
        """Keep the first, last, and points spaced roughly waypoint_spacing_m apart,
        so the potential field planner doesn't get a waypoint every single cell."""
        if len(world_path) <= 2:
            return world_path

        pruned = [world_path[0]]
        last = world_path[0]
        for pt in world_path[1:-1]:
            if math.hypot(pt[0] - last[0], pt[1] - last[1]) >= self.waypoint_spacing_m:
                pruned.append(pt)
                last = pt
        pruned.append(world_path[-1])
        return pruned

    def is_free_cell(self, c):
        v = self.map_array[c[1], c[0]]
        return 0 <= v < self.occupied_thresh

    def find_nearest_free_cell(self, start, max_radius=25):
        """BFS outward from `start` for the nearest known-free cell.

        Needed because blocking unknown (-1) cells in A* (so it can't cut
        through unexplored territory) means a start cell that itself lands
        on unknown ground -- e.g. the robot's own just-clamped edge
        position, before SLAM has scanned under it yet -- has no free
        neighbors to expand into, so search() fails immediately with "no
        path found" even though the robot is standing on real, walkable
        floor. Snapping to the nearest cell the map already knows is free
        gives A* somewhere it can actually search from.
        """
        if self.is_free_cell(start):
            return start
        h, w = self.map_array.shape
        visited = {start}
        queue = deque([start])
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
        while queue:
            cur = queue.popleft()
            if abs(cur[0] - start[0]) > max_radius or abs(cur[1] - start[1]) > max_radius:
                continue
            for dx, dy in neighbors:
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in visited or not (0 <= nxt[0] < w and 0 <= nxt[1] < h):
                    continue
                visited.add(nxt)
                if self.is_free_cell(nxt):
                    return nxt
                queue.append(nxt)
        return None

    # ------------------------------------------------------------------
    # Goal callback: plan and publish
    # ------------------------------------------------------------------

    def goal_callback(self, msg: PoseStamped):
        if self.map_array is None:
            self.get_logger().warn("No map received yet, ignoring goal")
            return

        robot_xy = self.get_robot_pose_in_map()
        if robot_xy is None:
            return

        start_grid = self.world_to_grid(*robot_xy)
        goal_grid = self.world_to_grid(
            msg.pose.position.x, msg.pose.position.y)

        self.get_logger().info(f"Planning from {start_grid} to {goal_grid}")

        # Extra diagnostics: if start/goal end up outside bounds, this prints
        # exactly why (map origin, robot's real world pose, map size) instead
        # of just "outside map bounds" -- saves a debugging round-trip.
        h, w = self.map_array.shape
        if not (0 <= start_grid[0] < w and 0 <= start_grid[1] < h):
            # This genuinely happens during SLAM exploration (Part 3), not just
            # a hypothetical: a forward-only-FOV laser mounted ahead of
            # base_link means the very first map, built before the robot has
            # moved at all, can end just short of covering the robot's own
            # position. Erroring out here is fatal -- the robot never gets a
            # path, never moves, the map never grows, and this never resolves
            # on its own. Clamp the START to the nearest in-bounds cell instead
            # (the goal is left untouched -- it always comes from a real cell
            # already inside the map, e.g. a frontier, so it is never the
            # problem). Planning from the clamped edge cell gets the robot
            # moving in roughly the right direction, which lets slam_toolbox
            # grow the map to actually include the robot within a cycle or two.
            clamped_start = (min(max(start_grid[0], 0), w - 1),
                              min(max(start_grid[1], 0), h - 1))
            self.get_logger().warn(
                f"Robot's own position is outside the current map bounds "
                f"(robot_world=({robot_xy[0]:.3f}, {robot_xy[1]:.3f}), "
                f"map covers x:[{self.map_info.origin.position.x:.2f}, "
                f"{self.map_info.origin.position.x + w * self.map_info.resolution:.2f}], "
                f"y:[{self.map_info.origin.position.y:.2f}, "
                f"{self.map_info.origin.position.y + h * self.map_info.resolution:.2f}]) "
                f"-- likely the SLAM map hasn't grown to cover it yet. Clamping "
                f"the A* start cell to {clamped_start} instead of aborting.")
            start_grid = clamped_start

        if not self.is_free_cell(start_grid):
            snapped = self.find_nearest_free_cell(start_grid)
            if snapped is None:
                self.get_logger().error(
                    f"No known-free cell within range of start {start_grid} "
                    f"-- can't plan yet, waiting for more of the map.")
                return
            self.get_logger().warn(
                f"Start cell {start_grid} is unknown/unmapped, snapped to "
                f"nearest known-free cell {snapped}.")
            start_grid = snapped

        grid_path = self.astar(start_grid, goal_grid)
        if grid_path is None:
            return

        world_path = [self.grid_to_world(gx, gy) for gx, gy in grid_path]
        world_path = self.prune_path(world_path)

        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for (x, y) in world_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation = msg.pose.orientation  # final waypoint gets goal orientation
            path_msg.poses.append(pose)

        # Overwrite the last waypoint's orientation with the actual requested goal
        # orientation (intermediate waypoints' orientation field is unused downstream).
        if path_msg.poses:
            path_msg.poses[-1].pose.orientation = msg.pose.orientation

        self.path_pub.publish(path_msg)
        self.get_logger().info(
            f"Published path with {len(path_msg.poses)} waypoints")


def main(args=None):
    rclpy.init(args=args)
    node = AStarGlobalPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
