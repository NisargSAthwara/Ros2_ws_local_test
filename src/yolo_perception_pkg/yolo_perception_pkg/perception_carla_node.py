#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models

# Append the directory containing gating.py so it can be imported directly
sys.path.append('/home/dell_ubuntu/ros2_ws/assets/pt_files')
import gating

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class LightweightRestorer(nn.Module):
    def __init__(self):
        super(LightweightRestorer, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class PerceptionCarlaNode(Node):
    def __init__(self):
        super().__init__('perception_carla_node')

        # Declare ROS2 parameters for model paths
        self.declare_parameter(
            'restorer_weights_path', 
            '/home/dell_ubuntu/ros2_ws/assets/pt_files/restorer.pt'
        )
        self.declare_parameter(
            'classifier_weights_path', 
            '/home/dell_ubuntu/ros2_ws/assets/pt_files/classifier.pt'
        )

        # Get parameter values
        self.restorer_weights_path = self.get_parameter('restorer_weights_path').get_parameter_value().string_value
        self.classifier_weights_path = self.get_parameter('classifier_weights_path').get_parameter_value().string_value

        # Select standard device with CUDA fallback
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Initialized perception device: {self.device}")



        # Initialize and load model weights
        self._init_models()

        # Initialize CV Bridge for converting ROS2 Images to/from OpenCV matrices
        self.bridge = CvBridge()

        # Create subscription directly to CARLA's raw front-camera topic
        self.subscription = self.create_subscription(
            Image,
            '/carla/ego_vehicle/rgb_front/image',
            self.image_callback,
            10
        )
        self.get_logger().info("Subscribed directly to topic: '/carla/ego_vehicle/rgb_front/image'")

        # Create publisher for the final processed (restored/bypassed) image
        self.publisher = self.create_publisher(
            Image,
            '/camera/front/cleaned',
            10
        )
        self.get_logger().info("Configured publisher for topic: '/camera/front/cleaned'")
        self.get_logger().info("Perception CARLA node successfully initialized and ready to process incoming frames.")

    def _init_models(self):
        # 1. Initialize custom restorer
        self.restorer = LightweightRestorer().to(self.device)
        checkpoint = torch.load(self.restorer_weights_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            restorer_state = checkpoint['model_state']
        else:
            restorer_state = checkpoint
        self.restorer.load_state_dict(restorer_state)
        self.get_logger().info(f"Successfully loaded LightweightRestorer weights from: {self.restorer_weights_path}")

        # 2. Initialize ResNet50 classifier
        self.classifier = models.resnet50()
        checkpoint = torch.load(self.classifier_weights_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            classifier_state = checkpoint['model_state']
        else:
            classifier_state = checkpoint

        # Adjust the fc layer structure based on checkpoint
        if 'fc.1.weight' in classifier_state:
            num_classes = classifier_state['fc.1.weight'].size(0)
            self.classifier.fc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(2048, num_classes)
            )
        elif 'fc.weight' in classifier_state:
            num_classes = classifier_state['fc.weight'].size(0)
            if num_classes != 1000:
                self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)

        self.classifier.load_state_dict(classifier_state)
        self.get_logger().info(f"Successfully loaded ResNet50 classifier weights from: {self.classifier_weights_path}")

        self.classifier = self.classifier.to(self.device)

        # Set both models to eval mode
        self.restorer.eval()
        self.classifier.eval()

    def image_callback(self, msg):
        start_time = self.get_clock().now()

        try:
            # 1. Convert ROS2 Image message to an OpenCV BGR frame using cv_bridge
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 2. Convert the incoming frame to grayscale for routing metrics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute metrics: global mean brightness, Laplacian variance, white pixel clipping ratio
            mean_brightness = float(np.mean(gray))
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            total_pixels = gray.size
            white_clipped_pixels = int(np.sum(gray > 240))
            white_clipping = float(white_clipped_pixels) / float(total_pixels)

            # 3. Illumination & Gating Switching Logic
            route_to_restorer, route_reason = gating.check_gating(mean_brightness, laplacian_var, white_clipping)

            # 4. Image Processing & Inference
            # Normalize the frame to a PyTorch tensor shaped (Channels, Height, Width) scaled between [0.0, 1.0]
            # Convert BGR frame from HWC numpy array to CHW PyTorch tensor
            tensor_bgr = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            # Add batch dimension and load to GPU/CPU
            tensor_bgr = tensor_bgr.unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Run the ResNet50 classifier on every frame to extract weather classification profiles
                classifier_logits = self.classifier(tensor_bgr)
                # Convert logits to list of floats for high-fidelity performance logging
                logits_list = classifier_logits.squeeze(0).cpu().numpy().tolist()

                if route_to_restorer:
                    # Pass the tensor through the LightweightRestorer
                    restored_tensor = self.restorer(tensor_bgr)
                    # Clamp outputs to [0.0, 1.0] range
                    restored_tensor = torch.clamp(restored_tensor, 0.0, 1.0)
                    
                    # Convert processed tensor back to an OpenCV BGR image
                    restored_numpy = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    final_frame = (restored_numpy * 255.0).astype(np.uint8)
                    active_pathway = f"Restored ({route_reason})"
                else:
                    # Bypass: pass the raw simulator frame forward
                    final_frame = frame
                    active_pathway = "Bypass"

            # 5. Convert processed frame back to a ROS2 Image message
            out_msg = self.bridge.cv2_to_imgmsg(final_frame, encoding="bgr8")

            # CRITICAL: Copy the original incoming 'msg.header' directly into the outbound message
            # to preserve simulator time synchronization for the downstream YOLO node.
            out_msg.header = msg.header

            # Broadcast the final image onto the unified topic
            self.publisher.publish(out_msg)

            # Compute execution latency
            end_time = self.get_clock().now()
            latency_ms = (end_time - start_time).nanoseconds / 1e6

            # 6. Production Performance Logging
            self.get_logger().info(
                f"Latency: {latency_ms:.2f} ms | Route: {active_pathway} | Logits: {logits_list}"
            )

        except Exception as e:
            self.get_logger().error(f"Perception processing failed: {str(e)}")

    def destroy_node(self):
        self.get_logger().info("Shutting down perception_carla_node. Cleaning up resources.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionCarlaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down perception_carla_node via KeyboardInterrupt.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
