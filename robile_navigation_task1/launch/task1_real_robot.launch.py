import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robile_nav_dir = get_package_share_directory('robile_navigation_task1')

    map_yaml_file = LaunchConfiguration('map')
    max_linear = LaunchConfiguration('max_linear')
    max_angular = LaunchConfiguration('max_angular')
    use_rviz = LaunchConfiguration('use_rviz')

    declare_map_cmd = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(robile_nav_dir, 'maps', 'c_069_latest.yaml'),
        description='Full path to the map yaml file to localize/plan against')

    declare_use_rviz_cmd = DeclareLaunchArgument(
        'use_rviz',
        default_value='True',
        description='Whether to start RViz2 with the saved nav view')

    declare_max_linear_cmd = DeclareLaunchArgument(
        'max_linear',
        default_value='0.15',
        description='Potential field planner max linear speed (m/s). '
                     'Keep this LOW for the first run on real hardware.')

    declare_max_angular_cmd = DeclareLaunchArgument(
        'max_angular',
        default_value='0.6',
        description='Potential field planner max angular speed (rad/s). '
                     'Keep this LOW for the first run on real hardware.')

    # --- Map server: serves the pre-built map of the real lab room ---
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'yaml_filename': map_yaml_file,
        }],
    )

    lifecycle_manager_map_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map_server',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['map_server'],
        }],
    )

    # --- AMCL: real-robot tuned params (amcl_config.yaml), NOT the sim
    #     nav2_params.yaml amcl block which hardcodes use_sim_time/initial_pose
    #     for Gazebo ---
    amcl_config = os.path.join(robile_nav_dir, 'config', 'amcl_config.yaml')

    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[amcl_config, {'use_sim_time': False}],
    )

    lifecycle_manager_localization_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['amcl'],
        }],
    )

    # --- Task 1 planning stack: A* global planner -> potential field local
    #     planner, both ported for the real robot (map frame, tf-based pose) ---
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
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': False}],
        condition=IfCondition(use_rviz),
    )

    ld = LaunchDescription()
    ld.add_action(declare_map_cmd)
    ld.add_action(declare_max_linear_cmd)
    ld.add_action(declare_max_angular_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(map_server_node)
    ld.add_action(lifecycle_manager_map_node)
    ld.add_action(amcl_node)
    ld.add_action(lifecycle_manager_localization_node)
    ld.add_action(global_planner_node)
    ld.add_action(potential_field_node)
    ld.add_action(rviz_node)
    return ld
