import rclpy
import math
import tf2_ros
from collections import deque

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from tf_transformations import euler_from_quaternion


class PotentialFieldPlanner(Node):
    """
    Local planner for following the global path.

    Uses the laser scan to avoid nearby obstacles and publishes
    velocity commands while moving through the received waypoints.
    
    """

    def __init__(self):
        super().__init__('potential_field_planner')
        self.get_logger().info("Potential Field Planner Started")

        # --- Force field gains ---
        self.ka    = 0.6
        self.kr    = 0.4    # lowered from 0.8 -- repulsion was dominating in narrow corridors
        self.rho_0 = 0.6    # lowered from 1.5 -- MUST be < (corridor_width / 2). Tune to your map.

        self.goal_tolerance = 0.15   # distance tolerance to advance to next waypoint
        self.goal_angle_tol = 0.1
        # Start slowing down as the robot gets close to the waypoint.
        # Keep this smaller than the spacing between waypoints so the robot
        # has enough distance to reach normal cruising speed.
        self.taper_radius   = 0.2

        # Rotate in place when the heading error is large.
        self.rotate_first_threshold = 0.6

        # Stuck / local-minimum detection
        self.stuck_force_threshold = 0.15   # net force magnitude below this counts as "cancelling"
        self.stuck_ticks_to_trigger = 15    # ~1.5s at 10Hz control loop
        self._stuck_counter = 0

        # Repulsive-force smoothing (rolling average over last N scans)
        self._rep_history = deque(maxlen=5)

       # Conservative speed limits for the robot.
       # These can be adjusted through ROS parameters.
        self.declare_parameter('max_linear', 0.15)
        self.declare_parameter('max_angular', 0.6)
        self.max_linear  = self.get_parameter('max_linear').value
        self.max_angular = self.get_parameter('max_angular').value

        self.planning_frame = 'map'   # change to 'odom' only if no localisation is running yet

        self.waypoints     = deque()   # queue of waypoints
        self.current_wp    = None
        self.latest_scan   = None
        self.goal_reached  = True      # true until a path is received
        self._idle_stop_sent = False

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.path_sub = self.create_subscription(
            Path, '/global_path', self.path_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timer       = self.create_timer(0.1, self.control_loop)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def scan_callback(self, msg):
        self.latest_scan = msg

    def path_callback(self, msg: Path):
        if not msg.poses:
            self.get_logger().warn("Received empty path, ignoring")
            return

        self.waypoints.clear()
        n = len(msg.poses)
        for i, pose_stamped in enumerate(msg.poses):
            is_final = (i == n - 1)
            theta = None
            if is_final:
                q = pose_stamped.pose.orientation
                (_, _, theta) = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.waypoints.append((
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                is_final,
                theta,
            ))

        self.current_wp = self.waypoints.popleft()
        self.goal_reached = False
        self._stuck_counter = 0
        self._rep_history.clear()
        self.get_logger().info(f"New path received: {n} waypoints")

    # ------------------------------------------------------------------
    # Pose lookup
    # ------------------------------------------------------------------

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.planning_frame, 'base_link', rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF: {e}")
            return None
        x = t.transform.translation.x
        y = t.transform.translation.y
        q = t.transform.rotation
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return x, y, yaw

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    def compute_total_force(self, robot_x, robot_y, yaw, goal_x, goal_y, dist):
        # --- Attractive force, tapered near the waypoint so it decelerates
        #     smoothly instead of driving at full force right to the boundary.
        if dist < 1e-6:
            fx_att, fy_att = 0.0, 0.0
        else:
            taper = min(dist / self.taper_radius, 1.0)
            fx_att = self.ka * taper * (goal_x - robot_x) / dist
            fy_att = self.ka * taper * (goal_y - robot_y) / dist

        # --- Repulsive force from the current raw scan ---
        fx_rep_raw = 0.0
        fy_rep_raw = 0.0

        if self.latest_scan is not None:
            scan = self.latest_scan
            for i, r in enumerate(scan.ranges):
                if math.isnan(r) or math.isinf(r):
                    continue
                if r < scan.range_min or r > self.rho_0:
                    continue

                angle_bl = scan.angle_min + i * scan.angle_increment
                ox_bl = r * math.cos(angle_bl)
                oy_bl = r * math.sin(angle_bl)

                ox = robot_x + ox_bl * math.cos(yaw) - oy_bl * math.sin(yaw)
                oy = robot_y + ox_bl * math.sin(yaw) + oy_bl * math.cos(yaw)

                rx = robot_x - ox
                ry = robot_y - oy
                d  = math.hypot(rx, ry)
                if d < 1e-6:
                    continue

                scale = self.kr * (1.0 / d - 1.0 / self.rho_0) / (d ** 2)
                fx_rep_raw += scale * (rx / d)
                fy_rep_raw += scale * (ry / d)

        # Smooth the repulsive force to reduce laser noise.
        self._rep_history.append((fx_rep_raw, fy_rep_raw))
        fx_rep = sum(f[0] for f in self._rep_history) / len(self._rep_history)
        fy_rep = sum(f[1] for f in self._rep_history) / len(self._rep_history)

        return fx_att + fx_rep, fy_att + fy_rep

    def clamp(self, v, limit):
        return max(-limit, min(limit, v))

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def control_loop(self):
        if self.goal_reached or self.current_wp is None:
            # Stop once when idle instead of repeatedly publishing zero velocity.
            if not self._idle_stop_sent:
                self.cmd_pub.publish(Twist())
                self._idle_stop_sent = True
            return
        self._idle_stop_sent = False

        pose = self.get_robot_pose()
        if pose is None:
            return
        robot_x, robot_y, yaw = pose

        wp_x, wp_y, is_final, wp_theta = self.current_wp
        dist_to_wp = math.hypot(wp_x - robot_x, wp_y - robot_y)

        self.get_logger().info(
            f"x={robot_x:.2f}, y={robot_y:.2f}, yaw={math.degrees(yaw):.1f}°, "
            f"wp=({wp_x:.2f},{wp_y:.2f}), dist={dist_to_wp:.2f}, "
            f"remaining={len(self.waypoints)}")

        cmd = Twist()

        if dist_to_wp > self.goal_tolerance:
            fx, fy = self.compute_total_force(
                robot_x, robot_y, yaw, wp_x, wp_y, dist_to_wp)
            desired_heading = math.atan2(fy, fx)
            heading_error = self._wrap(desired_heading - yaw)
            force_mag = math.hypot(fx, fy)

            self.get_logger().debug(
                f"fx={fx:.3f} fy={fy:.3f} force_mag={force_mag:.3f} "
                f"heading_error_deg={math.degrees(heading_error):.1f} "
                f"stuck_counter={self._stuck_counter}")

            # --- Stuck / local-minimum detection ---
            if force_mag < self.stuck_force_threshold:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0

            if self._stuck_counter >= self.stuck_ticks_to_trigger:
                # Detect when the robot is stuck and apply a small recovery motion.
                self.get_logger().warn(
                    "Local minimum detected (force ~0) -- applying escape nudge")
                cmd.angular.z = self.clamp(0.6, self.max_angular)
                cmd.linear.x = 0.05
                self.cmd_pub.publish(cmd)
                return
            
            # Rotate in place when the heading error is large.
            if abs(heading_error) > self.rotate_first_threshold:
                cmd.linear.x = 0.0
                cmd.angular.z = self.clamp(2.5 * heading_error, self.max_angular)
            else:
                alignment = max(0.0, math.cos(heading_error))
                cmd.linear.x = self.clamp(
                    self.max_linear * alignment * min(force_mag, 1.0),
                    self.max_linear)
                cmd.angular.z = self.clamp(2.0 * heading_error, self.max_angular)

        elif is_final and wp_theta is not None:
            # Final waypoint reached positionally -- align to goal orientation
            heading_error = self._wrap(wp_theta - yaw)
            if abs(heading_error) > self.goal_angle_tol:
                cmd.angular.z = self.clamp(0.8 * heading_error, self.max_angular)
            else:
                self.get_logger().info("Goal fully reached!")
                self.goal_reached = True
                self.current_wp = None

        else:
            # Intermediate waypoint reached -- advance to the next one
            self._stuck_counter = 0
            self._rep_history.clear()
            if self.waypoints:
                self.current_wp = self.waypoints.popleft()
                self.get_logger().info("Waypoint reached, advancing to next")
            else:
                # Shouldn't normally happen (last waypoint is always is_final=True)
                self.goal_reached = True
                self.current_wp = None

        self.cmd_pub.publish(cmd)

    @staticmethod
    def _wrap(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a


def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
