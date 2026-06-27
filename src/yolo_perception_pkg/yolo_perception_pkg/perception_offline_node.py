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

class PerceptionOfflineNode(Node):
    def __init__(self):
        super().__init__('perception_offline_node')
        
        # Declare ROS2 Parameters for file paths
        self.declare_parameter('video_path', '/home/dell_ubuntu/ros2_ws/assets/sample_driving_test.mp4')
        self.declare_parameter('restorer_weights_path', '/home/dell_ubuntu/ros2_ws/assets/pt_files/restorer.pt')
        self.declare_parameter('classifier_weights_path', '/home/dell_ubuntu/ros2_ws/assets/pt_files/classifier.pt')
        
        # Get parameter values
        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        self.restorer_weights_path = self.get_parameter('restorer_weights_path').get_parameter_value().string_value
        self.classifier_weights_path = self.get_parameter('classifier_weights_path').get_parameter_value().string_value
        


        # Device selection (CUDA with CPU fallback)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")
        
        # Initialize and load model weights
        self._init_models()
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Initialize Publisher for unified topic
        self.publisher = self.create_publisher(Image, '/camera/front/cleaned', 10)
        
        # Check if video_path is a directory (image sequence) or video file
        self.is_image_sequence = os.path.isdir(self.video_path)
        self.cap = None
        
        if self.is_image_sequence:
            # Read image files sorted alphabetically
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
            self.image_files = sorted([
                os.path.join(self.video_path, f) for f in os.listdir(self.video_path)
                if f.lower().endswith(valid_extensions)
            ])
            if not self.image_files:
                raise FileNotFoundError(f"No valid images found in directory: {self.video_path}")
            self.image_index = 0
            self.get_logger().info(f"Loaded image sequence with {len(self.image_files)} frames from: {self.video_path}")
        else:
            # Initialize VideoCapture object
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.get_logger().error(f"Failed to open video file: {self.video_path}")
                raise FileNotFoundError(f"Video file not found or could not be opened: {self.video_path}")
            self.get_logger().info(f"Video file opened successfully: {self.video_path}")
        
        # Spawn ROS2 Timer running at 30 FPS (~0.0333s interval)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info("Perception offline node initialized successfully. 30 FPS timer started.")

    def _init_models(self):
        # 1. Initialize custom denoiser
        self.restorer = LightweightRestorer().to(self.device)
        checkpoint = torch.load(self.restorer_weights_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            restorer_state = checkpoint['model_state']
        else:
            restorer_state = checkpoint
        self.restorer.load_state_dict(restorer_state)
        self.get_logger().info(f"Successfully loaded restorer weights from {self.restorer_weights_path}")
        
        # 2. Initialize classifier
        self.classifier_name = "resnet50"
        self.classifier = models.resnet50()
        checkpoint = torch.load(self.classifier_weights_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            classifier_state = checkpoint['model_state']
        else:
            classifier_state = checkpoint
        
        # Determine architecture from checkpoint keys
        keys = list(classifier_state.keys())
        is_efficientnet = any('classifier.1' in k for k in keys) or 'efficientnet' in self.classifier_weights_path.lower()
        
        if is_efficientnet:
            self.classifier_name = "efficientnet_b0"
            self.classifier = models.efficientnet_b0()
            if 'classifier.1.weight' in classifier_state:
                num_classes = classifier_state['classifier.1.weight'].size(0)
                if num_classes != 1000:
                    self.classifier.classifier[1] = nn.Linear(self.classifier.classifier[1].in_features, num_classes)
        else:
            self.classifier_name = "resnet50"
            self.classifier = models.resnet50()
            if 'fc.weight' in classifier_state:
                num_classes = classifier_state['fc.weight'].size(0)
                if num_classes != 1000:
                    self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        
        self.classifier.load_state_dict(classifier_state)
        self.get_logger().info(f"Successfully loaded {self.classifier_name} classifier weights from {self.classifier_weights_path}")
        self.classifier = self.classifier.to(self.device)
        
        # Set models to evaluation mode
        self.restorer.eval()
        self.classifier.eval()

    def timer_callback(self):
        start_time = self.get_clock().now()
        
        # 1. Read next frame
        if self.is_image_sequence:
            if self.image_index >= len(self.image_files):
                # Loop infinitely: reset index to 0
                self.image_index = 0
            img_path = self.image_files[self.image_index]
            frame = cv2.imread(img_path)
            if frame is None:
                self.get_logger().error(f"Failed to read image frame from sequence: {img_path}")
                self.image_index += 1
                return
            self.image_index += 1
        else:
            ret, frame = self.cap.read()
            if not ret:
                # End of video stream reached, loop back to the start
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    self.get_logger().error("End of video reached and failed to loop back.")
                    return
                
        try:
            # 2. Convert to grayscale and calculate lighting metrics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray))
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            total_pixels = gray.size
            white_clipped_pixels = int(np.sum(gray > 240))
            white_clipping = float(white_clipped_pixels) / float(total_pixels)
            
            # 3. Illumination & Gating Switching Logic
            # 3. Illumination & Gating Switching Logic
            route_to_restorer, route_reason = gating.check_gating(mean_brightness, laplacian_var, white_clipping)
            
            # 4. Image Processing (PyTorch tensor preparation)
            # Normalizing BGR image shape (H, W, C) to Tensor (C, H, W) scaled between [0.0, 1.0]
            tensor_bgr = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            tensor_bgr = tensor_bgr.unsqueeze(0).to(self.device) # shape (1, C, H, W)
            
            with torch.no_grad():
                # Run the ResNet50/EfficientNet classifier on every single frame to get the weather class logit
                classifier_logits = self.classifier(tensor_bgr)
                weather_id = int(torch.argmax(classifier_logits, dim=1).item())
                
                # Apply LightweightRestorer if routed, otherwise preserve original frame
                if route_to_restorer:
                    restored_tensor = self.restorer(tensor_bgr)
                    restored_tensor = torch.clamp(restored_tensor, 0.0, 1.0)
                    
                    # Convert PyTorch tensor back to OpenCV BGR image
                    restored_numpy = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    final_frame = (restored_numpy * 255.0).astype(np.uint8)
                else:
                    final_frame = frame
            
            # 5. Convert back to ROS2 Image and publish
            msg = self.bridge.cv2_to_imgmsg(final_frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_front_link"
            self.publisher.publish(msg)
            
            # Calculate latency in milliseconds
            end_time = self.get_clock().now()
            processing_latency = (end_time - start_time).nanoseconds / 1e6
            
            # Log progress tick info
            self.get_logger().info(
                f"Latency: {processing_latency:.2f} ms | Route: {route_reason} | Weather ID: {weather_id}"
            )
            
        except Exception as e:
            self.get_logger().error(f"Error during perception tick processing: {str(e)}")

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap is not None:
            if isinstance(self.cap, cv2.VideoCapture) and self.cap.isOpened():
                self.cap.release()
                self.get_logger().info("Successfully released VideoCapture resource.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionOfflineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node via keyboard interrupt.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
