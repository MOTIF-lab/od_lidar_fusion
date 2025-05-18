from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Path to the parameter file
    param_file = '/home/linux/ws_livox/launch/c920.launch.yaml'

    return LaunchDescription([
        # Node 1: usb_cam node
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usbcam',
            namespace='cam',
            parameters=[param_file]
        ),

        # Node 2: image_proc rectify node
        Node(
            package='image_proc',
            executable='rectify_node',
            name='image_proc_rect',
            namespace='cam',
            remappings=[
                ('image', '/cam/image_raw'),
                ('camera_info', '/cam/camera_info')
            ]
        ),

        # Node 3: tf2_ros static transform publisher
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar_camera_tf_publisher',
            arguments=[
                '-0.07133886811939455', '0.05215759108248205', '0.07042566622467987',
                '-0.49848021662437386', '0.541814791282095', '-0.5052786147334285', '0.45016411126957356',
                'livox_frame', 'camera'
            ]
        ),
    ])