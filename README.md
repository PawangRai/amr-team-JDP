## Repository Structure

Each ROS 2 package lives at the repo root:

- `astar_global_planner/` — grid-based A* global planner (Task 1 & 3)
- `potential_field_planner/` — local planner using attractive/repulsive forces to follow waypoints (Task 1 & 3)
- `mcl_localization/` — custom Monte Carlo localisation / particle filter (Task 2)
- `frontier_exploration/` — frontier-based exploration node (Task 3)
- `robile_navigation_task1/` — Task 1 launch/config files (AMCL localisation against a static map)
- `robile_navigation_task3/` — Task 3 launch/config files (live SLAM + exploration)

`robile_navigation_task1` and `robile_navigation_task3` are **not** standalone ROS packages — they're launch/config files meant to be merged into the [HBRS-AMR/robile_navigation](https://github.com/HBRS-AMR/robile_navigation) vendor package (config files into its `config/`, launch files into its `launch/`), since that package provides the base bringup infrastructure we build on top of.

## How to Run

1. Clone this repo's packages and the vendor `robile_navigation` package into your workspace's `src/`, then merge the `robile_navigation_task1`/`robile_navigation_task3` files into vendor `robile_navigation`'s `config/`/`launch/` folders.
2. `colcon build --symlink-install`
3. `source install/setup.bash`, and set `export ROS_DOMAIN_ID=<your robot's domain>` in every terminal (must match the robot — check with the robot's own hostname/config if unsure).
4. On the robot: `ros2 launch robile_bringup robot.launch.py`
5. On your machine:
   - **Task 1:** `ros2 launch robile_navigation task1_real_robot.launch.py`
   - **Task 2:** `ros2 launch mcl_localization task2_real_robot.launch.py`
   - **Task 3:** `ros2 launch robile_navigation task3_exploration_real_robot.launch.py`

## Our Approach

**Task 1 — Path and Motion Planning:** We used `nav2_amcl` for localisation against a pre-built map (`c_069_latest`), an A* planner over the inflated occupancy grid to produce a sparse sequence of waypoints to the goal, and a potential field planner as the local controller — attractive force toward the current waypoint, repulsive force from obstacles in the live laser scan, with the two force fields summed to produce the commanded velocity.

**Task 2 — Localisation:** We implemented a Monte Carlo localisation (particle filter) from scratch, following the standard predict → weight (via laser scan likelihood) → resample cycle, and integrated it as a drop-in alternative to AMCL.

**Task 3 — Environment Exploration:** We paired `slam_toolbox` (online async mode, building the map live) with a frontier exploration node that continuously selects goal poses at the boundary between known and unknown space and feeds them to the same A* + potential field planning stack from Task 1, so the robot autonomously explores without a pre-built map.

## Challenges Faced

- **TF frame mismatch on the real robot:** the vendor bringup swaps `base_link`↔`base_footprint` relative to simulation, so AMCL, `slam_toolbox`, and our planners all had to be reconfigured to use `base_link` as the robot's real-robot base frame instead of `base_footprint`.
- **ROS_DOMAIN_ID mismatches:** the biggest real-robot blocker was simply a domain ID mismatch between our laptop and the robot — no sensor data arrived, which looked identical to a dead laser/driver until we diagnosed it directly.
- **A* planning through unmapped space:** our first version of A* treated unexplored (`-1`) grid cells as freely traversable, so during exploration it would plan paths that cut straight through unmapped territory outside the actual corridor. Fixed by requiring cells to be *known*-free, and separately snapping the robot's start cell to the nearest known-free cell when it landed on not-yet-scanned ground (common right after the map first starts building).
- **Potential field getting stuck at obstacles:** near round obstacles (pillars), the repulsive force's `1/d²` term made the desired heading jitter tick-to-tick, and a bug in our stuck/local-minimum recovery logic (the escape counter never reset after firing) caused the robot to spin in place indefinitely instead of recovering. Fixed with a distance floor on the repulsive force, hysteresis on the rotate-vs-drive mode switch, and resetting the stuck counter after each recovery nudge.
- **Map noise from in-place rotation:** much of the map "ghosting"/doubled walls we saw during exploration traced back to the robot spinning in place near obstacles (see above) — scans captured mid-spin smear before `slam_toolbox`'s scan matcher can correct them, so fixing the spin issue substantially cleaned up map quality too.
