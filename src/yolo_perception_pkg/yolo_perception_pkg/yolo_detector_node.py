#!/usr/bin/env python3

import os
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Import vision_msgs for publishing bounding boxes
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("The 'ultralytics' library is required to run the yolo_detector_node. Please install it using: pip install ultralytics")

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')

        # Declare parameters
        self.declare_parameter('yolo_weights_path', '/home/dell_ubuntu/ros2_ws/assets/pt_files/yolo.pt')
        self.declare_parameter('confidence_threshold', 0.5)

        # Get parameter values
        self.yolo_weights_path = self.get_parameter('yolo_weights_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        self.bridge = CvBridge()
        self.model = None

        self.model_path = self.yolo_weights_path

        # Try to pre-load the model to save startup latency
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"Successfully initialized YOLO model with source: {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize YOLO model on startup: {str(e)}. Will attempt initialization on first frame.")

        # Create Subscriber for the cleaned/restored frames
        self.subscription = self.create_subscription(
            Image,
            '/camera/front/cleaned',
            self.image_callback,
            10
        )
        self.get_logger().info("Subscribed to topic: '/camera/front/cleaned'")

        # Create Publisher for bounding box detections
        self.publisher = self.create_publisher(
            Detection2DArray,
            '/yolo/detections',
            10
        )
        self.get_logger().info("Configured publisher for topic: '/yolo/detections'")
        self.get_logger().info("YOLO Detector node initialized successfully.")

    def image_callback(self, msg):
        start_time = self.get_clock().now()

        try:
            # 1. Convert ROS2 Image message to BGR OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 2. Lazily initialize the model if it wasn't loaded during startup
            if self.model is None:
                self.model = YOLO(self.model_path)

            # 3. Perform inference
            results = self.model.predict(cv_image, conf=self.confidence_threshold, verbose=False)

            # 4. Construct Detection2DArray message
            detection_array = Detection2DArray()
            
            # CRITICAL: Copy the original incoming msg.header directly into the outbound detections header
            # to preserve simulator time synchronization for the downstream components
            detection_array.header = msg.header

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates [xmin, ymin, xmax, ymax]
                    xyxy = box.xyxy[0].cpu().numpy()
                    xmin, ymin, xmax, ymax = map(float, xyxy)

                    # Get class and confidence
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    # Construct Detection2D message
                    detection = Detection2D()
                    detection.header = msg.header

                    # Construct hypothesis compatibility logic (handles multiple vision_msgs versions)
                    hypothesis = ObjectHypothesisWithPose()
                    if hasattr(hypothesis, 'hypothesis'):
                        # Modern vision_msgs
                        if hasattr(hypothesis.hypothesis, 'class_id'):
                            hypothesis.hypothesis.class_id = str(cls_id)
                        else:
                            hypothesis.hypothesis.id = str(cls_id)
                        hypothesis.hypothesis.score = conf
                    else:
                        # Legacy vision_msgs
                        if hasattr(hypothesis, 'class_id'):
                            hypothesis.class_id = str(cls_id)
                        else:
                            hypothesis.id = str(cls_id)
                        hypothesis.score = conf

                    detection.results.append(hypothesis)

                    # Setup bounding box dimensions and center coordinates
                    detection.bbox.center.position.x = (xmin + xmax) / 2.0
                    detection.bbox.center.position.y = (ymin + ymax) / 2.0
                    detection.bbox.size_x = xmax - xmin
                    detection.bbox.size_y = ymax - ymin

                    detection_array.detections.append(detection)

            # 5. Broadcast the detections to downstream path planners / trackers
            self.publisher.publish(detection_array)

            # Compute execution latency
            end_time = self.get_clock().now()
            latency_ms = (end_time - start_time).nanoseconds / 1e6

            self.get_logger().info(
                f"YOLO Inference Latency: {latency_ms:.2f} ms | Detections: {len(detection_array.detections)}"
            )

        except Exception as e:
            self.get_logger().error(f"YOLO detector processing failed: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down YOLO detector node via KeyboardInterrupt.")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
