import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)

import tf2_ros
from tf2_ros import TransformBroadcaster
from tf_transformations import euler_from_quaternion, quaternion_from_euler

from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import (
    PoseArray,
    Pose,
    PoseWithCovarianceStamped,
    TransformStamped,
)

from scipy.ndimage import distance_transform_edt


class MCLNode(Node):
    """
    Monte Carlo Localisation (MCL) particle filter for a real ROS 2 robot.

    Main mentions
      * 900 particles by default (ROS parameter, so it is easy to tune).
      * Vectorised likelihood-field sensor update so the larger particle set
        remains practical.
      * Uses the strongest particle mode (cluster around the best particle)
        instead of averaging all particles when the distribution is multimodal.
      * Publishes that same best-mode estimate on /mcl_pose and /mcl_best_pose.
      * Broadcasts map -> odom from the best-mode estimate so RViz RobotModel
        follows the localisation result through map -> odom -> base_link.
      * map -> odom is stamped with the ROBOT odom/TF timestamp instead of the
        laptop's wall-clock time.  This avoids the common laptop/robot clock
        offset problem that produces TF "extrapolation into the past/future".
      * Pose is estimated BEFORE resampling resets particle weights.
      * /initialpose resets the odometry baseline.
      * New scans can update localisation even while the robot is stationary.
      * Resampling is triggered by effective sample size (ESS), reducing jitter.
      * Laser mounting transform is read from TF when available; the old
        0.45 m forward offset is retained only as a fallback.
    """
    
    #MCL begins by creating many hypothetical robot poses called particles.
	#Each particle represents:
	#[
	#p_i = (x_i, y_i, \theta_i, w_i)
	#]
	#where:
	#- (x_i) = x-position of particle i,
	#- (y_i) = y-position,
	#- (theta_i) = orientation,
	#- (w_i) = weight or probability of that particle.

    def __init__(self):
        super().__init__('mcl_localization')

        # ---------------- Parameters ----------------
        self.declare_parameter('num_particles', 900) #no of particles to be samples
        self.declare_parameter('update_rate_hz', 5.0)
        self.declare_parameter('beam_subsample', 12)
        self.declare_parameter('max_range', 6.0)
        self.declare_parameter('sigma_hit', 0.20)
        self.declare_parameter('z_hit', 0.85)
        self.declare_parameter('z_random', 0.15)
        self.declare_parameter('resample_ess_fraction', 0.55)
        self.declare_parameter('best_cluster_radius', 0.70)
        self.declare_parameter('best_cluster_yaw_deg', 45.0)
        self.declare_parameter('min_best_cluster_particles', 30)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('odom_frame', 'odom')

        self.num_particles = int(self.get_parameter('num_particles').value)
        update_rate_hz = float(self.get_parameter('update_rate_hz').value)
        self.beam_subsample = int(self.get_parameter('beam_subsample').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.sigma_hit = float(self.get_parameter('sigma_hit').value)
        self.z_hit = float(self.get_parameter('z_hit').value)
        self.z_random = float(self.get_parameter('z_random').value)
        self.resample_ess_fraction = float(
            self.get_parameter('resample_ess_fraction').value
        )
        self.best_cluster_radius = float(
            self.get_parameter('best_cluster_radius').value
        )
        self.best_cluster_yaw = math.radians(
            float(self.get_parameter('best_cluster_yaw_deg').value)
        )
        self.min_best_cluster_particles = int(
            self.get_parameter('min_best_cluster_particles').value
        )
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)

        # Odometry motion noise parameters (Thrun alpha1-alpha4).
        self.alpha1 = 0.05
        self.alpha2 = 0.02
        self.alpha3 = 0.05
        self.alpha4 = 0.02

        # ---------------- State ----------------
        self.map_info = None
        self.map_array = None
        self.likelihood_field = None

        self.particles = None          # Nx3: x, y, theta
        self.weights = None            # N

        self.last_odom_pose = None     # previous (x, y, theta)
        self._latest_odom = None       # current (x, y, theta)
        self._latest_odom_stamp = None
        self.latest_scan = None
        self._latest_scan_stamp_ns = None
        self._last_processed_scan_stamp_ns = None
        self.initialized = False

        # Last good pose selected from the strongest particle mode.
        self.last_estimate = None      # (x, y, theta, covariance36)

        # Laser extrinsic relative to base_link.  Read from TF when possible.
        self._laser_frame = None
        self._laser_tf_ready = False
        self._laser_warned = False
        self.laser_offset_x = 0.45     # fallback from original project code
        self.laser_offset_y = 0.0
        self.laser_offset_yaw = 0.0

        # ---------------- ROS interfaces ----------------
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, map_qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 20
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )
        self.initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.initialpose_callback,
            10,
        )

        self.particle_pub = self.create_publisher(
            PoseArray, '/mcl_particles', 10
        )
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/mcl_pose', 10
        )
        # Same selected best-mode pose, useful for direct RViz comparison.
        self.best_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/mcl_best_pose', 10
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        period = 1.0 / max(update_rate_hz, 0.5)
        self.timer = self.create_timer(period, self.update_loop)

        self.get_logger().info(
            f'MCL Localization Started with {self.num_particles} particles '
            f'({update_rate_hz:.1f} Hz)'
        )

    # ------------------------------------------------------------------
    # Map handling
    # ------------------------------------------------------------------

    def map_callback(self, msg: OccupancyGrid):
        self.map_info = msg.info
        w, h = msg.info.width, msg.info.height
        grid = np.array(msg.data, dtype=np.int8).reshape(h, w)
        self.map_array = grid

        occupied = grid >= 50
        dist_px = distance_transform_edt(~occupied)
        self.likelihood_field = dist_px * msg.info.resolution

        self.get_logger().info(
            f'Map + likelihood field ready ({w}x{h}, '
            f'{msg.info.resolution:.3f} m/cell)'
        )

        if not self.initialized:
            self.init_particles_uniform()

    # ------------------------------------------------------------------
    # Particle initialisation
    # ------------------------------------------------------------------

    def init_particles_uniform(self):
        """Spread particles uniformly over known free cells."""
        if self.map_array is None:
            return

        free_ys, free_xs = np.where(self.map_array == 0)
        if len(free_xs) == 0:
            self.get_logger().warn(
                'No free cells found in map; cannot initialise particles.'
            )
            return

        idx = np.random.choice(
            len(free_xs), self.num_particles, replace=True
        )
        gx = free_xs[idx]
        gy = free_ys[idx]
        wx, wy = self.grid_to_world(gx, gy)
        theta = np.random.uniform(-math.pi, math.pi, self.num_particles)

        self.particles = np.column_stack([wx, wy, theta])
        self.weights = np.ones(self.num_particles, dtype=float) / self.num_particles
        self.initialized = True
        self.last_estimate = None

        self.get_logger().info(
            f'Initialised {self.num_particles} particles globally.'
        )

    def initialpose_callback(self, msg: PoseWithCovarianceStamped):
        """Re-seed around RViz 2D Pose Estimate."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Use RViz covariance if it is sensible; otherwise retain practical
        # defaults for a hand-clicked initial estimate.
        cov = msg.pose.covariance
        sx = math.sqrt(max(cov[0], 0.0)) if cov[0] > 1e-6 else 0.30
        sy = math.sqrt(max(cov[7], 0.0)) if cov[7] > 1e-6 else 0.30
        st = math.sqrt(max(cov[35], 0.0)) if cov[35] > 1e-6 else 0.20
        sx = min(max(sx, 0.10), 1.00)
        sy = min(max(sy, 0.10), 1.00)
        st = min(max(st, 0.08), 0.80)

        xs = np.random.normal(x, sx, self.num_particles)
        ys = np.random.normal(y, sy, self.num_particles)
        thetas = np.random.normal(theta, st, self.num_particles)
        thetas = self._wrap(thetas)

        self.particles = np.column_stack([xs, ys, thetas])
        self.weights = np.ones(self.num_particles, dtype=float) / self.num_particles
        self.initialized = True

        # IMPORTANT: do not apply an odometry delta from before the manual
        # re-localisation to the newly seeded particles.
        self.last_odom_pose = None
        self.last_estimate = None
        self._last_processed_scan_stamp_ns = None

        self.get_logger().info(
            f'Re-seeded {self.num_particles} particles around '
            f'({x:.2f}, {y:.2f}, {math.degrees(theta):.1f} deg).'
        )

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def grid_to_world(self, gx, gy):
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        return ox + (gx + 0.5) * res, oy + (gy + 0.5) * res

    # ------------------------------------------------------------------
    # ROS input callbacks
    # ------------------------------------------------------------------

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self._latest_odom = (x, y, theta)
        self._latest_odom_stamp = msg.header.stamp

        if msg.header.frame_id:
            self.odom_frame = msg.header.frame_id.lstrip('/')

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        self._latest_scan_stamp_ns = self._stamp_to_ns(msg.header.stamp)

        scan_frame = msg.header.frame_id.lstrip('/')
        if scan_frame and scan_frame != self._laser_frame:
            self._laser_frame = scan_frame
            self._laser_tf_ready = False
            self._laser_warned = False

    # ------------------------------------------------------------------
    # Motion model
    # ------------------------------------------------------------------
    #Here we are using invesrse kinematics because the robots wheel motion is getting determined eventually according to the desired robot motion.

    def motion_update(self, odom_prev, odom_curr):
        x0, y0, th0 = odom_prev
        x1, y1, th1 = odom_curr

        d_trans = math.hypot(x1 - x0, y1 - y0)
        if d_trans < 1e-3:
            d_rot1 = 0.0
        else:
            d_rot1 = self._wrap(
                math.atan2(y1 - y0, x1 - x0) - th0
            )
        d_rot2 = self._wrap(th1 - th0 - d_rot1)

        n = self.particles.shape[0]

        rot1_sigma = self.alpha1 * abs(d_rot1) + self.alpha2 * d_trans
        trans_sigma = (
            self.alpha3 * d_trans
            + self.alpha4 * (abs(d_rot1) + abs(d_rot2))
        )
        rot2_sigma = self.alpha1 * abs(d_rot2) + self.alpha2 * d_trans

        rot1_noisy = d_rot1 - np.random.normal(0.0, rot1_sigma, n)
        trans_noisy = d_trans - np.random.normal(0.0, trans_sigma, n)
        rot2_noisy = d_rot2 - np.random.normal(0.0, rot2_sigma, n)

        theta = self.particles[:, 2]
        self.particles[:, 0] += trans_noisy * np.cos(theta + rot1_noisy)
        self.particles[:, 1] += trans_noisy * np.sin(theta + rot1_noisy)
        self.particles[:, 2] = self._wrap(
            theta + rot1_noisy + rot2_noisy
        )

    # ------------------------------------------------------------------
    # Laser extrinsic
    # ------------------------------------------------------------------

    def _update_laser_extrinsic_from_tf(self):
        if self._laser_tf_ready or not self._laser_frame:
            return

        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame,
                self._laser_frame,
                rclpy.time.Time(),
            )
            self.laser_offset_x = t.transform.translation.x
            self.laser_offset_y = t.transform.translation.y
            q = t.transform.rotation
            _, _, self.laser_offset_yaw = euler_from_quaternion(
                [q.x, q.y, q.z, q.w]
            )
            self._laser_tf_ready = True
            self.get_logger().info(
                'Laser extrinsic loaded from TF: '
                f'x={self.laser_offset_x:.3f}, '
                f'y={self.laser_offset_y:.3f}, '
                f'yaw={math.degrees(self.laser_offset_yaw):.1f} deg '
                f'({self.base_frame} -> {self._laser_frame})'
            )
        except Exception as exc:
            if not self._laser_warned:
                self.get_logger().warn(
                    'Could not read laser mounting TF yet; using fallback '
                    f'x={self.laser_offset_x:.2f}, y={self.laser_offset_y:.2f}, '
                    f'yaw={math.degrees(self.laser_offset_yaw):.1f} deg. '
                    f'Error: {exc}'
                )
                self._laser_warned = True

    # ------------------------------------------------------------------
    # Sensor model: vectorised likelihood-field model
    # ------------------------------------------------------------------

    def sensor_update(self, scan: LaserScan):
        if self.likelihood_field is None or self.particles is None:
            return False

        self._update_laser_extrinsic_from_tf()

        ranges = np.asarray(scan.ranges, dtype=float)
        indices = np.arange(0, len(ranges), self.beam_subsample)
        r = ranges[indices]
        angles = scan.angle_min + indices * scan.angle_increment

        valid = (
            np.isfinite(r)
            & (r > scan.range_min)
            & (r < min(scan.range_max, self.max_range))
        )
        r = r[valid]
        angles = angles[valid]
        if r.size == 0:
            return False

        px = self.particles[:, 0]
        py = self.particles[:, 1]
        ptheta = self.particles[:, 2]

        c = np.cos(ptheta)
        s = np.sin(ptheta)
        laser_x = px + self.laser_offset_x * c - self.laser_offset_y * s
        laser_y = py + self.laser_offset_x * s + self.laser_offset_y * c
        laser_theta = ptheta + self.laser_offset_yaw

        world_angles = laser_theta[:, None] + angles[None, :]
        ex = laser_x[:, None] + r[None, :] * np.cos(world_angles)
        ey = laser_y[:, None] + r[None, :] * np.sin(world_angles)

        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h, w = self.likelihood_field.shape

        gx = ((ex - ox) / res).astype(np.int32)
        gy = ((ey - oy) / res).astype(np.int32)

        in_bounds = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        dist = np.full(gx.shape, self.max_range, dtype=float)
        dist[in_bounds] = self.likelihood_field[gy[in_bounds], gx[in_bounds]]

        prob_hit = np.exp(-(dist ** 2) / (2.0 * self.sigma_hit ** 2))
        prob = self.z_hit * prob_hit + self.z_random / self.max_range
        prob = np.clip(prob, 1e-12, None)

        log_weights = np.sum(np.log(prob), axis=1)
        log_weights -= np.max(log_weights)
        w_lin = np.exp(log_weights)
        total = np.sum(w_lin)

        if not np.isfinite(total) or total < 1e-300:
            self.weights = np.ones(self.num_particles, dtype=float) / self.num_particles
            return False

        self.weights = w_lin / total
        return True

    # ------------------------------------------------------------------
    # Best-mode pose estimation
    # ------------------------------------------------------------------

    def estimate_best_mode(self):
        """
        Estimate pose from the strongest particle mode.

        We deliberately do NOT use a single argmax particle because that can
        jump between neighbouring samples.  We start at the maximum-weight
        particle, gather nearby particles with similar heading, then compute
        the weighted mean of that local mode.
        """
        if self.particles is None or self.weights is None:
            return None

        weights = np.asarray(self.weights, dtype=float)
        if not np.all(np.isfinite(weights)) or np.sum(weights) <= 0.0:
            weights = np.ones(len(self.particles), dtype=float) / len(self.particles)

        best_idx = int(np.argmax(weights))
        bx, by, btheta = self.particles[best_idx]

        dx = self.particles[:, 0] - bx
        dy = self.particles[:, 1] - by
        dpos = np.hypot(dx, dy)
        dtheta = np.abs(self._wrap(self.particles[:, 2] - btheta))

        mask = (
            (dpos <= self.best_cluster_radius)
            & (dtheta <= self.best_cluster_yaw)
        )

        # If the local mode is too sparse, use the top-weighted particles
        # rather than averaging the entire (possibly multimodal) cloud.
        if int(np.sum(mask)) < self.min_best_cluster_particles:
            top_k = max(
                self.min_best_cluster_particles,
                int(round(0.10 * len(self.particles))),
            )
            top_k = min(top_k, len(self.particles))
            top_idx = np.argpartition(weights, -top_k)[-top_k:]
            mask = np.zeros(len(self.particles), dtype=bool)
            mask[top_idx] = True

        p = self.particles[mask]
        w = weights[mask]
        w_sum = np.sum(w)
        if not np.isfinite(w_sum) or w_sum <= 1e-15:
            w = np.ones(len(p), dtype=float) / len(p)
        else:
            w = w / w_sum

        mean_x = float(np.sum(w * p[:, 0]))
        mean_y = float(np.sum(w * p[:, 1]))
        sin_sum = float(np.sum(w * np.sin(p[:, 2])))
        cos_sum = float(np.sum(w * np.cos(p[:, 2])))
        mean_theta = math.atan2(sin_sum, cos_sum)

        ex = p[:, 0] - mean_x
        ey = p[:, 1] - mean_y
        eth = self._wrap(p[:, 2] - mean_theta)

        var_x = float(np.sum(w * ex * ex))
        var_y = float(np.sum(w * ey * ey))
        cov_xy = float(np.sum(w * ex * ey))
        var_theta = float(np.sum(w * eth * eth))

        covariance = [0.0] * 36
        covariance[0] = max(var_x, 1e-6)
        covariance[1] = cov_xy
        covariance[6] = cov_xy
        covariance[7] = max(var_y, 1e-6)
        covariance[35] = max(var_theta, 1e-6)

        cluster_mass = float(np.sum(weights[mask]))
        cluster_count = int(np.sum(mask))

        return (
            mean_x,
            mean_y,
            mean_theta,
            covariance,
            cluster_count,
            cluster_mass,
        )

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def effective_sample_size(self):
        denom = float(np.sum(np.square(self.weights)))
        if denom <= 1e-15:
            return 0.0
        return 1.0 / denom

    def resample(self):
        n = self.particles.shape[0]
        positions = (np.arange(n) + np.random.uniform()) / n
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0

        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(n, dtype=float) / n

    # ------------------------------------------------------------------
    # Main filter loop
    # ------------------------------------------------------------------

    def update_loop(self):
        if not self.initialized or self._latest_odom is None:
            return

        odom_curr = self._latest_odom

        if self.last_odom_pose is None:
            self.last_odom_pose = odom_curr

        dx = odom_curr[0] - self.last_odom_pose[0]
        dy = odom_curr[1] - self.last_odom_pose[1]
        dth = abs(self._wrap(odom_curr[2] - self.last_odom_pose[2]))
        moved = (math.hypot(dx, dy) >= 0.005) or (dth >= 0.005)

        if moved:
            self.motion_update(self.last_odom_pose, odom_curr)
            self.last_odom_pose = odom_curr

        new_scan = (
            self.latest_scan is not None
            and self._latest_scan_stamp_ns is not None
            and self._latest_scan_stamp_ns != self._last_processed_scan_stamp_ns
        )

        sensor_updated = False
        if new_scan:
            sensor_updated = self.sensor_update(self.latest_scan)
            self._last_processed_scan_stamp_ns = self._latest_scan_stamp_ns

        # Crucial ordering: estimate/publish while sensor weights still exist.
        # If no new scan arrived, estimate the current propagated cloud.  When
        # weights are uniform after resampling this is still the resampled
        # posterior, and the best-mode selection prevents a global-cloud mean.
        estimate = self.estimate_best_mode()
        if estimate is not None:
            mx, my, mtheta, covariance, cluster_count, cluster_mass = estimate
            self.last_estimate = (mx, my, mtheta, covariance)
            self.publish_estimate(
                mx,
                my,
                mtheta,
                covariance,
                cluster_count,
                cluster_mass,
            )

        if sensor_updated:
            neff = self.effective_sample_size()
            if neff < self.resample_ess_fraction * self.num_particles:
                self.resample()

    # ------------------------------------------------------------------
    # Output: pose, particles, map -> odom
    # ------------------------------------------------------------------

    def publish_estimate(
        self,
        mx,
        my,
        mtheta,
        covariance,
        cluster_count,
        cluster_mass,
    ):
        if self.particles is None:
            return

        # Get the robot's latest odom -> base_link transform.  Most importantly,
        # we reuse ITS timestamp for BOTH /mcl_pose and map -> odom.  Therefore
        # the TF chain uses the robot's clock even if the laptop wall clock is
        # tens of seconds different.
        t_odom_base = self._lookup_latest_odom_to_base()
        if t_odom_base is not None:
            output_stamp = t_odom_base.header.stamp
        elif self._latest_odom_stamp is not None:
            output_stamp = self._latest_odom_stamp
        else:
            output_stamp = self.get_clock().now().to_msg()

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = 'map'
        pose_msg.header.stamp = output_stamp
        pose_msg.pose.pose.position.x = float(mx)
        pose_msg.pose.pose.position.y = float(my)
        q = quaternion_from_euler(0.0, 0.0, float(mtheta))
        pose_msg.pose.pose.orientation.x = q[0]
        pose_msg.pose.pose.orientation.y = q[1]
        pose_msg.pose.pose.orientation.z = q[2]
        pose_msg.pose.pose.orientation.w = q[3]
        pose_msg.pose.covariance = covariance

        self.pose_pub.publish(pose_msg)
        self.best_pose_pub.publish(pose_msg)

        cloud_msg = PoseArray()
        cloud_msg.header = pose_msg.header
        cloud_msg.poses = []
        for x, y, theta in self.particles:
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            qq = quaternion_from_euler(0.0, 0.0, float(theta))
            p.orientation.x = qq[0]
            p.orientation.y = qq[1]
            p.orientation.z = qq[2]
            p.orientation.w = qq[3]
            cloud_msg.poses.append(p)
        self.particle_pub.publish(cloud_msg)

        # This is what makes RViz RobotModel follow the MCL estimate.
        self.broadcast_map_to_odom(mx, my, mtheta, t_odom_base, output_stamp)

        sigma_x = math.sqrt(max(covariance[0], 0.0))
        sigma_y = math.sqrt(max(covariance[7], 0.0))
        sigma_yaw = math.degrees(math.sqrt(max(covariance[35], 0.0)))
        self.get_logger().info(
            f'MCL best mode: x={mx:.2f}, y={my:.2f}, '
            f'yaw={math.degrees(mtheta):.1f} deg | '
            f'cluster={cluster_count}/{self.num_particles}, '
            f'mass={cluster_mass:.3f}, '
            f'sigma=({sigma_x:.2f} m, {sigma_y:.2f} m, '
            f'{sigma_yaw:.1f} deg)',
            throttle_duration_sec=1.0,
        )

    def _lookup_latest_odom_to_base(self):
        try:
            return self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.base_frame,
                rclpy.time.Time(),
            )
        except Exception as exc:
            # We can still fall back to the /odom message below, but do not
            # flood the terminal every 0.2 s.
            self.get_logger().warn(
                f'TF ({self.odom_frame}->{self.base_frame}) unavailable; '
                f'using /odom fallback if possible: {exc}',
                throttle_duration_sec=2.0,
            )
            return None

    def broadcast_map_to_odom(
        self,
        mx,
        my,
        mtheta,
        t_odom_base,
        output_stamp,
    ):
        """
        Publish map -> odom such that:

            T(map->base) = T(map->odom) * T(odom->base)

        Therefore:

            T(map->odom) = T(map->base_estimate) * inverse(T(odom->base))

        The transform is stamped using the ROBOT odometry/TF timestamp, not
        self.get_clock().now(), which is the key laptop-vs-robot time fix.
        """
        if t_odom_base is not None:
            ox = t_odom_base.transform.translation.x
            oy = t_odom_base.transform.translation.y
            oq = t_odom_base.transform.rotation
            _, _, otheta = euler_from_quaternion(
                [oq.x, oq.y, oq.z, oq.w]
            )
            child_odom_frame = t_odom_base.header.frame_id.lstrip('/') or self.odom_frame
        elif self._latest_odom is not None:
            ox, oy, otheta = self._latest_odom
            child_odom_frame = self.odom_frame
        else:
            return

        # Direct SE(2) composition.  This is algebraically equivalent to
        # T_map_base * inverse(T_odom_base), but clearer and less error-prone.
        delta_theta = self._wrap(mtheta - otheta)
        c = math.cos(delta_theta)
        s = math.sin(delta_theta)

        map_odom_x = mx - (c * ox - s * oy)
        map_odom_y = my - (s * ox + c * oy)
        map_odom_theta = delta_theta

        tf_msg = TransformStamped()
        tf_msg.header.stamp = output_stamp
        tf_msg.header.frame_id = 'map'
        tf_msg.child_frame_id = child_odom_frame
        tf_msg.transform.translation.x = float(map_odom_x)
        tf_msg.transform.translation.y = float(map_odom_y)
        tf_msg.transform.translation.z = 0.0

        q = quaternion_from_euler(0.0, 0.0, float(map_odom_theta))
        tf_msg.transform.rotation.x = q[0]
        tf_msg.transform.rotation.y = q[1]
        tf_msg.transform.rotation.z = q[2]
        tf_msg.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(tf_msg)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _stamp_to_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi


def main(args=None):
    rclpy.init(args=args)
    node = MCLNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
