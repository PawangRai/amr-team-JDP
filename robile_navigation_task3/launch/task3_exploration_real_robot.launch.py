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

    max_linear = LaunchConfiguration('max_linear')
    max_angular = LaunchConfiguration('max_angular')
    use_rviz = LaunchConfiguration('use_rviz')
    slam_params_file = LaunchConfiguration('slam_params_file')

    declare_use_rviz_cmd = DeclareLaunchArgument(
        'use_rviz',
        default_value='True',
        description='Whether to start RViz2 with the saved nav view')

    declare_max_linear_cmd = DeclareLaunchArgument(
        'max_linear',
        default_value='0.15',
        description='Potential field planner max linear speed (m/s). '
                     'Keep this LOW for the first exploration run on real hardware.')

    declare_max_angular_cmd = DeclareLaunchArgument(
        'max_angular',
        default_value='0.6',
        description='Potential field planner max angular speed (rad/s). '
                     'Keep this LOW for the first exploration run on real hardware.')

    declare_slam_params_cmd = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(robile_nav_dir, 'config', 'mapper_params_real_robot.yaml'),
        description='slam_toolbox params -- real-robot version (base_frame: base_link, '
                     'NOT the Gazebo mapper_params_online_async.yaml, which uses '
                     'base_footprint -- see Task 1/3 debugging notes).')

    # --- SLAM: builds /map live. NO map_server/AMCL here -- slam_toolbox
    #     both builds the map and publishes map->odom itself. ---
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robile_nav_dir, 'launch', 'online_async.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'false',
            'slam_params_file': slam_params_file,
        }.items(),
    )

    # --- Task 3 exploration + Task 1 planning stack, reused unmodified ---
    frontier_exploration_node = Node(
        package='frontier_exploration',
        executable='frontier_exploration',
        name='frontier_exploration',
        output='screen',
    )

    global_planner_node = Node(
        package='astar_global_planner',
        executable='global_planner',
        name='astar_global_planner',
        output='screen',
    )

    potential_field_node = Node(
        package='potential_field_planner',
        executable='potential_field',
        name='potential_field_planner',
        output='screen',
        parameters=[{
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
        parameters=[{'use_sim_time': False}],
        condition=IfCondition(use_rviz),
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_max_linear_cmd)
    ld.add_action(declare_max_angular_cmd)
    ld.add_action(declare_slam_params_cmd)
    ld.add_action(slam_toolbox)
    ld.add_action(frontier_exploration_node)
    ld.add_action(global_planner_node)
    ld.add_action(potential_field_node)
    ld.add_action(rviz_node)
    return ld
