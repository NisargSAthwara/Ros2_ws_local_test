#!/usr/bin/env python3
"""
TH OWL — YOLO Object Detector Node
====================================
Subscribes to the cleaned/restored image stream from the perception node,
runs YOLOv8/YOLOv10 inference, and publishes bounding box detections as
vision_msgs/Detection2DArray.

Subscribes:   /camera/front/cleaned   (sensor_msgs/Image)
Publishes:    /yolo/detections         (vision_msgs/Detection2DArray)

Authors:  TH OWL Project 8
Node:     yolo_detector_node
"""

import os

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "The 'ultralytics' library is required to run yolo_detector_node.\n"
        "Install it with:  pip install ultralytics"
    )

# Portable default — expands to current user's home, no hardcoded username
_DEFAULT_YOLO_PATH = os.path.join(
    os.path.expanduser('~'), 'ros2_ws', 'assets', 'pt_files', 'yolo.pt')


class YoloDetectorNode(Node):

    def __init__(self) -> None:
        super().__init__('yolo_detector_node')

        # ── ROS2 Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('yolo_weights_path',    _DEFAULT_YOLO_PATH)
        self.declare_parameter('confidence_threshold', 0.5)

        self.yolo_weights_path   = self.get_parameter(
            'yolo_weights_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter(
            'confidence_threshold').get_parameter_value().double_value

        # ── YOLO Model Initialization ───────────────────────────────────────────────
        self.bridge = CvBridge()
        self.model  = None
        self._load_yolo_model()

        # ── Subscriptions ──────────────────────────────────────────────────────────
        self.subscription = self.create_subscription(
            Image,
            '/camera/front/cleaned',
            self.image_callback,
            10,
        )
        self.get_logger().info("Subscribed to: '/camera/front/cleaned'")

        # ── Publishers ─────────────────────────────────────────────────────────────
        self.publisher = self.create_publisher(
            Detection2DArray, '/yolo/detections', 10)
        self.get_logger().info(
            "Publisher ready: '/yolo/detections' | "
            f"Confidence threshold: {self.confidence_threshold}")
        self.get_logger().info('YoloDetectorNode initialized successfully.')

    # ─────────────────────────────────────────────────────────────────────────────
    #  Model Loading
    # ─────────────────────────────────────────────────────────────────────────────

    def _load_yolo_model(self) -> None:
        """
        Attempt to pre-load the YOLO model at startup.
        Logs a FATAL message and keeps self.model = None if the file is missing
        or the load fails. The model will not be lazily retried — a missing file
        at startup indicates a configuration error that should be fixed.
        """
        path = self.yolo_weights_path
        if not os.path.isfile(path):
            self.get_logger().fatal(
                f'[YOLO] Weight file not found: {path}\n'
                f'  → Provide the file or override the "yolo_weights_path" parameter.\n'
                f'  → Inference calls will be skipped until the model is available.')
            return
        try:
            self.model = YOLO(path)
            self.get_logger().info(f'[YOLO] Model loaded from: {path}')
        except Exception as exc:
            self.get_logger().fatal(f'[YOLO] Model load failed — {exc}')

    # ─────────────────────────────────────────────────────────────────────────────
    #  Image Callback
    # ─────────────────────────────────────────────────────────────────────────────

    def image_callback(self, msg: Image) -> None:
        start_time = self.get_clock().now()

        # Skip inference if the model is unavailable (missing weights)
        if self.model is None:
            self.get_logger().warn(
                '[YOLO] Model not loaded — skipping inference for this frame.',
                throttle_duration_sec=5.0,
            )
            return

        try:
            # ── Step 1: Decode ROS2 Image ──────────────────────────────────────────
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # ── Step 2: YOLO Inference ─────────────────────────────────────────────
            results = self.model.predict(
                cv_image, conf=self.confidence_threshold, verbose=False)

            # ── Step 3: Build Detection2DArray ────────────────────────────────────
            detection_array        = Detection2DArray()
            detection_array.header = msg.header   # preserve simulator timestamp

            for result in results:
                for box in result.boxes:
                    xyxy             = box.xyxy[0].cpu().numpy()
                    xmin, ymin, xmax, ymax = map(float, xyxy)
                    cls_id           = int(box.cls[0].item())
                    conf             = float(box.conf[0].item())

                    detection        = Detection2D()
                    detection.header = msg.header

                    # vision_msgs version-compatibility shim
                    hypothesis = ObjectHypothesisWithPose()
                    if hasattr(hypothesis, 'hypothesis'):
                        # Modern vision_msgs (≥ 4.x)
                        h_inner = hypothesis.hypothesis
                        if hasattr(h_inner, 'class_id'):
                            h_inner.class_id = str(cls_id)
                        else:
                            h_inner.id = str(cls_id)
                        h_inner.score = conf
                    else:
                        # Legacy vision_msgs (< 4.x)
                        if hasattr(hypothesis, 'class_id'):
                            hypothesis.class_id = str(cls_id)
                        else:
                            hypothesis.id = str(cls_id)
                        hypothesis.score = conf

                    detection.results.append(hypothesis)

                    detection.bbox.center.position.x = (xmin + xmax) / 2.0
                    detection.bbox.center.position.y = (ymin + ymax) / 2.0
                    detection.bbox.size_x             = xmax - xmin
                    detection.bbox.size_y             = ymax - ymin

                    detection_array.detections.append(detection)

            # ── Step 4: Publish Detections ─────────────────────────────────────────
            self.publisher.publish(detection_array)

            end_time   = self.get_clock().now()
            latency_ms = (end_time - start_time).nanoseconds / 1e6

            self.get_logger().info(
                f'YOLO Latency: {latency_ms:.2f} ms | '
                f'Detections: {len(detection_array.detections)}'
            )

        except Exception as exc:
            self.get_logger().error(f'yolo_detector image_callback failed: {exc}')


# -------------------------------------------------------------------------------------
#  Entry Point
# -------------------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            'Shutting down YoloDetectorNode via KeyboardInterrupt.')
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
