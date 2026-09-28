import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robile_nav_dir = get_package_share_directory('robile_navigation')
    robile_gazebo_dir = get_package_share_directory('robile_gazebo')

    max_linear = LaunchConfiguration('max_linear')
    max_angular = LaunchConfiguration('max_angular')
    use_rviz = LaunchConfiguration('use_rviz')

    declare_use_rviz_cmd = DeclareLaunchArgument(
        'use_rviz',
        default_value='True',
        description='Whether to start RViz2 with the saved nav view')

    declare_max_linear_cmd = DeclareLaunchArgument(
        'max_linear',
        default_value='0.5',
        description='Potential field planner max linear speed (m/s) for the sim run.')

    declare_max_angular_cmd = DeclareLaunchArgument(
        'max_angular',
        default_value='1.2',
        description='Potential field planner max angular speed (rad/s) for the sim run.')

    # --- Gazebo: spawns the Robile, robot_state_publisher, base_footprint->
    #     base_link static TF, and its own RViz instance (left running; the
    #     view we actually use is the one launched below with the saved
    #     nav config, pointed at /map, /global_path, etc.) ---
    gazebo_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robile_gazebo_dir, 'launch', 'gazebo_4_wheel.launch.py')
        ),
    )

    # --- SLAM: slam_toolbox builds /map live from /scan + odom->base_footprint
    #     (confirmed connected in Gazebo; see Task 1/3 debugging notes for why
    #     this differs from the real robot, which needs base_link instead) ---
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robile_nav_dir, 'launch', 'online_async.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    # --- Task 3 exploration + Task 1 planning stack, reused unmodified ---
    frontier_exploration_node = Node(
        package='frontier_exploration',
        executable='frontier_exploration',
        name='frontier_exploration',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    global_planner_node = Node(
        package='astar_global_planner',
        executable='global_planner',
        name='astar_global_planner',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    potential_field_node = Node(
        package='potential_field_planner',
        executable='potential_field',
        name='potential_field_planner',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'max_linear': max_linear,
            'max_angular': max_angular,
        }],
    )

    rviz_config = os.path.join(robile_nav_dir, 'config', 'robile_ros2_nav.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_exploration',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(use_rviz),
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_max_linear_cmd)
    ld.add_action(declare_max_angular_cmd)
    ld.add_action(gazebo_bringup)
    ld.add_action(slam_toolbox)
    ld.add_action(frontier_exploration_node)
    ld.add_action(global_planner_node)
    ld.add_action(potential_field_node)
    ld.add_action(rviz_node)
    return ld
