from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    map_file    = '/home/david/ros2_ws/src/prob_lab/maps/turtlebot3_world.yaml'
    rviz_file   = '/home/david/ros2_ws/src/prob_lab/rviz/prob_lab.rviz'
    plot_script = '/home/david/ros2_ws/src/prob_lab/plot_results.py'

    turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    nav2_bringup      = get_package_share_directory('nav2_bringup')

    return LaunchDescription([

        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),

        # ── Gazebo ──────────────────────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(turtlebot3_gazebo, 'launch', 'turtlebot3_world.launch.py')
            )
        ),

        # ── Nav2 Localization (Map Server + AMCL + TF Tree) ─
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup, 'launch', 'localization_launch.py')
            ),
            launch_arguments={
                'map':          map_file,
                'use_sim_time': 'false',
                'autostart':    'true',
            }.items()
        ),

        # ── KF Node ─────────────────────────────────────────
        Node(
            package='prob_lab', executable='kf_node', name='kf_node',
            parameters=[{'q_x':0.01,'q_y':0.01,'q_theta':0.01,
                         'r_x':0.1, 'r_y':0.1, 'r_theta':0.1}],
            output='screen'
        ),

        # ── EKF Node ────────────────────────────────────────
        Node(
            package='prob_lab', executable='ekf_node', name='ekf_node',
            parameters=[{'q_x':0.01,'q_y':0.01,'q_theta':0.01,
                         'r_x':0.1, 'r_y':0.1, 'r_theta':0.1}],
            output='screen'
        ),

        # ── PF Node ─────────────────────────────────────────
        Node(
            package='prob_lab', executable='pf_node', name='pf_node',
            parameters=[{
                'num_particles': 500,
                'process_noise_v': 0.3,
                'process_noise_w': 0.2,
                'measurement_noise_r': 0.3,
                'measurement_noise_yaw': 0.3,
            }],
            output='screen'
        ),

        # ── Path Publisher ───────────────────────────────────
        Node(
            package='prob_lab', executable='path_publisher_node',
            name='path_publisher_node', output='screen'
        ),

        # ── Evaluation Node ──────────────────────────────────
        Node(
            package='prob_lab', executable='evaluation_node',
            name='evaluation_node',
            parameters=[{
                'landmark_x':0.6,'landmark_y':0.6,
                'dropout_start_sec':15.0,'dropout_duration_sec':5.0,
                'simulate_dropout':True,'plot_script':plot_script,
            }],
            output='screen'
        ),

        # ── Initiale Pose für AMCL nach 8s ──────────────────
        TimerAction(period=10.0, actions=[
            ExecuteProcess(
                cmd=['ros2', 'topic', 'pub', '--once', '/initialpose',
                     'geometry_msgs/msg/PoseWithCovarianceStamped',
                     '{"header":{"frame_id":"map"},'
                     '"pose":{"pose":{"position":{"x":0.0,"y":0.0,"z":0.0},'
                     '"orientation":{"x":0.0,"y":0.0,"z":0.0,"w":1.0}},'
                     '"covariance":[0.25,0,0,0,0,0,0,0.25,0,0,0,0,0,0,0,0,0,0,'
                     '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.07]}}'],
                output='screen'
            ),
        ]),

        # ── RViz (nach 5s) ───────────────────────────────────
        TimerAction(period=5.0, actions=[
            Node(
                package='rviz2', executable='rviz2', name='rviz2',
                arguments=['-d', rviz_file], output='screen'
            ),
        ]),

        # ── Trajectory (nach 15s — nach AMCL Initialisierung) 
        TimerAction(period=15.0, actions=[
            Node(
                package='prob_lab', executable='trajectory_node',
                name='trajectory_node', output='screen'
            ),
        ]),

    ])
