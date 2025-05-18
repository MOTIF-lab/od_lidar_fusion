#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import message_filters
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import ros2_numpy
from scipy.spatial.transform import Rotation

class CalibrationChecker(Node):
    def __init__(self):
        super().__init__('calibration_checker')

        # Camera intrinsic parameters (replace with your rectified camera values)
        self.camera_matrix = np.array([[650.72962, 0, 393.59377], [0, 644.92734, 240.86657], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(5, dtype=np.float32)  # Zero distortion for rectified images

        # Initialize CvBridge
        self.bridge = CvBridge()

        # TF2 listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers with time synchronizer
        image_sub = message_filters.Subscriber(self, Image, '/cam/image_rect')
        pc_sub = message_filters.Subscriber(self, PointCloud2, '/livox/lidar')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, pc_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.get_logger().info('Calibration checker node started')

    def callback(self, image_msg, pc_msg):
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # Get transform from lidar to camera
        try:
            transform = self.tf_buffer.lookup_transform(
                'camera','livox_frame', pc_msg.header.stamp, rclpy.duration.Duration(seconds=1.0))
        except TransformException as e:
            self.get_logger().warn(f'Could not get transform: {e}')
            return

        # Convert point cloud to numpy array
        try:
            pc_data = ros2_numpy.point_cloud2.point_cloud2_to_array(pc_msg)
            points = pc_data['xyz']  # Shape: (N, 3) for x, y, z
            if len(points) == 0:
                self.get_logger().warn('Empty point cloud')
                return
        except Exception as e:
            self.get_logger().error(f'Error reading point cloud: {e}')
            return

        # Add homogeneous coordinates
        points = np.hstack((points, np.ones((points.shape[0], 1))))

        # Transform points to camera frame
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = Rotation.from_quat([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]).as_matrix()
        transform_matrix[:3, 3] = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ]
        points_camera = (transform_matrix @ points.T).T[:, :3]

        # Project points to image plane
        points_3d = points_camera[points_camera[:, 2] > 0]  # Filter points in front of camera
        if len(points_3d) == 0:
            self.get_logger().warn('No points in front of camera')
            return

        points_2d, _ = cv2.projectPoints(
            points_3d, np.zeros(3), np.zeros(3), self.camera_matrix, self.dist_coeffs)
        points_2d = points_2d.reshape(-1, 2).astype(int)

        # Filter points within image bounds
        h, w = cv_image.shape[:2]
        valid = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        points_2d = points_2d[valid]

        for u, v in points_2d:
            cv2.circle(cv_image, (u, v), 2, (0, 255, 0), -1)

        # Display image
        cv2.imshow('Calibration Check', cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CalibrationChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()