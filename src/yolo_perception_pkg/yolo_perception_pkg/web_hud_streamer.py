#!/usr/bin/env python3
"""
TH OWL — Web HUD Streamer Node
===================================
Subscribes to the telemetry HUD image topic and hosts a lightweight
MJPEG HTTP stream server, allowing the HUD to be viewed in a web browser
from headless/WSL environments.

Subscribes:   /perception/telemetry_hud   (sensor_msgs/Image)
Hosts:        http://localhost:8080
"""

import sys
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Global frame buffer and sync lock
latest_frame = None
frame_lock = threading.Lock()

class WebHUDStreamer(Node):
    def __init__(self) -> None:
        super().__init__('web_hud_streamer')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/perception/telemetry_hud',
            self.image_callback,
            10,
        )
        self.get_logger().info("WebHUDStreamer node initialized.")
        self.get_logger().info("Subscribed to: '/perception/telemetry_hud'")

    def image_callback(self, msg: Image) -> None:
        global latest_frame
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with frame_lock:
                latest_frame = cv_image
        except Exception as exc:
            self.get_logger().error(f"Failed to convert HUD image: {exc}")


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    frame_bytes = None
                    with frame_lock:
                        if latest_frame is not None:
                            ret, jpeg = cv2.imencode('.jpg', latest_frame)
                            if ret:
                                frame_bytes = jpeg.tobytes()
                    
                    if frame_bytes:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Content-length', str(len(frame_bytes)))
                        self.end_headers()
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b'\r\n')
                    else:
                        # If no frame received yet, send a blank message to keep connection alive
                        time.sleep(0.1)
                        continue
                    
                    time.sleep(0.05)  # Throttle streaming rate
            except Exception:
                pass
        else:
            self.send_response(404)
            self.end_headers()


def start_server() -> None:
    # Bind to all interfaces so it works across WSL and local network hostnames
    server = HTTPServer(('0.0.0.0', 8080), StreamHandler)
    print("-----------------------------------------------------------------")
    print("Web HUD stream active! Open your browser and navigate to:")
    print(" 👉 http://localhost:8080")
    print("-----------------------------------------------------------------")
    server.serve_forever()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WebHUDStreamer()
    
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down WebHUDStreamer via KeyboardInterrupt.')
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
