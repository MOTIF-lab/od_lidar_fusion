#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from yolo_msgs.msg import DetectionArray
from cv_bridge import CvBridge
import message_filters
from tf2_ros import Buffer, TransformListener, TransformException
import ros2_numpy
from scipy.spatial.transform import Rotation
from collections import defaultdict
import threading
from fusion_msgs.msg import FusionDetections, FusionDetection

class ObjectDistanceEstimator(Node):
    def __init__(self):
        super().__init__('object_distance_estimator')

        # Camera intrinsics (replace with your actual values)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros(5, dtype=np.float32)  # Assuming rectified images
        self.image_width = None
        self.image_height = None

        self.tracks = defaultdict(lambda: {'distance': None, 'timestamp': None, 'speed': None})

        # Initialize utilities
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Camera info subscriber
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/cam/camera_info', self.camera_info_callback, 10)

        # Synchronized subscribers
        image_sub = message_filters.Subscriber(self, Image, '/cam/image_rect')
        pc_sub = message_filters.Subscriber(self, PointCloud2, '/livox/lidar')
        det_sub = message_filters.Subscriber(self, DetectionArray, '/yolo/tracking')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, pc_sub, det_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.track_pub = self.create_publisher(FusionDetections, '/fusion/detections', 10)

        self.get_logger().info('Node initialized')

    def camera_info_callback(self, msg):
        # Populate camera matrix from CameraInfo
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            self.get_logger().info(f'Camera matrix: {self.camera_matrix}')
            # Verify image size
            self.image_width = msg.width
            self.image_height = msg.height
            self.get_logger().info(f'Image size: {self.image_width}x{self.image_height}')

    def callback(self, image_msg, pc_msg, det_msg):
         # Wait for camera info
        if self.camera_matrix is None:
            self.get_logger().warn('Waiting for camera info')
            return
        
        # Convert image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        height, width = cv_image.shape[:2]

        # Verify image size matches CameraInfo
        if width != self.image_width or height != self.image_height:
            self.get_logger().warn(f'Image size mismatch: {width}x{height} vs {self.image_width}x{self.image_height}')
            return

        # Get LiDAR-to-camera transform
        try:
            transform = self.tf_buffer.lookup_transform(
                'camera', 'livox_frame', pc_msg.header.stamp, rclpy.duration.Duration(seconds=0.5))
        except TransformException as e:
            self.get_logger().warn(f'Transform failed: {e}')
            return

        # Extract and transform point cloud
        pc_array = ros2_numpy.point_cloud2.point_cloud2_to_array(pc_msg)
        points = pc_array['xyz']
        # downsample points
        points = points[::10]  # Downsample by taking every 10th point
        if points.size == 0:
            self.get_logger().warn('Empty point cloud')
            return

        # Build transformation matrix
        rot = Rotation.from_quat([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]).as_matrix()
        trans = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ])
        points_camera = (rot @ points.T + trans[:, np.newaxis]).T

        # Filter points in front of camera and project to 2D
        valid = points_camera[:, 2] > 0
        points_3d = points_camera[valid]
        if points_3d.size == 0:
            return

        points_2d, _ = cv2.projectPoints(
            points_3d, np.zeros(3), np.zeros(3), self.camera_matrix, self.dist_coeffs)
        points_2d = points_2d.reshape(-1, 2).astype(int)
        valid_bounds = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
                       (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
        points_2d = points_2d[valid_bounds]
        points_3d = points_3d[valid_bounds]

        # Create a FusionDetections message
        fusion_detections = FusionDetections()
        fusion_detections.header = pc_msg.header
        fusion_detections.detections = []
        
        # Process each detection
        for det in det_msg.detections:
            # reject low confidence detections
            if det.score < 0.5:
                continue

            bbox = det.bbox
            x_min = int(bbox.center.position.x - bbox.size.x / 2)
            x_max = int(bbox.center.position.x + bbox.size.x / 2)
            y_min = int(bbox.center.position.y - bbox.size.y / 2)
            y_max = int(bbox.center.position.y + bbox.size.y / 2)

            # Find points within bounding box
            in_bbox = (points_2d[:, 0] >= x_min) & (points_2d[:, 0] <= x_max) & \
                      (points_2d[:, 1] >= y_min) & (points_2d[:, 1] <= y_max)
            bbox_points = points_3d[in_bbox]

            # Calculate and display distance
            if bbox_points.size > 0:
                tracked_detection = FusionDetection()
                tracked_detection.detection = det
                distance = np.median(bbox_points[:, 2])
                label = f'{det.class_name}({det.id}): {distance:.2f}m'
                tracked_detection.center.x = np.mean(bbox_points[:, 0])
                tracked_detection.center.y = np.mean(bbox_points[:, 1])
                tracked_detection.center.z = distance
                self.compute_speed(det.id, distance, pc_msg.header.stamp)
                if self.tracks[det.id]['speed'] is not None:
                    speed = self.tracks[det.id]['speed']
                    tracked_detection.twist.linear.z = speed
                    label += f', Speed: {speed:.2f} m/s'
                # Draw points
                # for u, v in points_2d[in_bbox]:
                #     cv2.circle(cv_image, (u, v), 2, (0, 255, 0), -1)
                fusion_detections.detections.append(tracked_detection)
            else:
                label = f'{det.class_name}({det.id}): No points'

            self.track_pub.publish(fusion_detections)
            cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(cv_image, label, (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show result
        cv2.imshow('Detections with Distance', cv_image)
        cv2.waitKey(1)

    def compute_speed(self, obj_id, current_distance, timestamp):
        if obj_id in self.tracks:
            previous_distance = self.tracks[obj_id]['distance']
            previous_timestamp = self.tracks[obj_id]['timestamp']
            if previous_distance is not None and previous_timestamp is not None:
                speed = (current_distance - previous_distance) / (timestamp.nanosec - previous_timestamp.nanosec) * 1e-9
                self.tracks[obj_id]['speed'] = speed
        self.tracks[obj_id]['distance'] = current_distance
        self.tracks[obj_id]['timestamp'] = timestamp
            

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDistanceEstimator()
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