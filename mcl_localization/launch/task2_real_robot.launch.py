import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml = LaunchConfiguration('map')

    # Reuse the saved map already stored in the Task 1 navigation package.
    task1_share = get_package_share_directory('robile_navigation_task1')

    default_map = os.path.join(
        task1_share,
        'maps',
        'c_069_latest.yaml'
    )

    mcl_share = get_package_share_directory('mcl_localization')

    mcl_config = os.path.join(
        mcl_share,
        'config',
        'task2_mcl.yaml'
    )

    return LaunchDescription([

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time; false for the real Robile'
        ),

        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Path to saved occupancy-grid YAML map'
        ),

        # -------------------------------------------------------------
        # Static map used by our custom Monte Carlo localization
        # -------------------------------------------------------------
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[
                {
                    'yaml_filename': map_yaml,
                    'use_sim_time': use_sim_time
                }
            ]
        ),

        # map_server is a lifecycle node, so configure and activate it
        # automatically.
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_task2',
            output='screen',
            parameters=[
                {
                    'autostart': True,
                    'node_names': ['map_server'],
                    'use_sim_time': use_sim_time
                }
            ]
        ),

        # -------------------------------------------------------------
        # Our Task 2 custom Monte Carlo Localization implementation
        # -------------------------------------------------------------
        Node(
            package='mcl_localization',
            executable='mcl_localization',
            name='mcl_localization',
            output='screen',
            parameters=[
                mcl_config,
                {
                    'use_sim_time': use_sim_time
                }
            ]
        ),
    ])
