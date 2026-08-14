#!/usr/bin/env python3
"""
TH OWL — Adverse Weather Perception Node (Offline / Dataset Evaluation Mode)
=============================================================================
Reads frames from a local video file or sorted image directory, applies the
full two-stage adverse weather perception pipeline at a configurable FPS rate,
and publishes results to ROS2 topics plus an optional OpenCV display window.

Pipeline per frame:
  1. Read frame from VideoCapture / image sequence
  2. Compute grayscale image metrics for metric-based gating (Stage B)
  3. EfficientNetV2-M weather classification (Stage A) with correct
     BGR→RGB, 512×512 resize, ImageNet normalization preprocessing
  4. Dual-layer routing decision:
       route = classifier_adverse OR metric_gating_flag
  5. Conditional LightweightRestorer pass (if route=RESTORE)
  6. CWQI severity scoring (4-metric no-reference quality index)
  7. Publish cleaned frame → /camera/front/cleaned
  8. Render + publish telemetry HUD → /perception/telemetry_hud
  9. Async telemetry log → results/telemetry_log.csv
 10. Optional: display OpenCV HUD window (show_display parameter)

Authors:  TH OWL Project 8
Node:     perception_offline_node
"""

import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
import torchvision.models as models

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from yolo_perception_pkg.gating import check_gating
from yolo_perception_pkg.severity_analyzer import compute_cwqi
from yolo_perception_pkg.telemetry_logger import (
    TelemetryLogger,
    compute_weather_entropy,
    render_hud_panel,
)

# -------------------------------------------------------------------------------------
#  Constants
# -------------------------------------------------------------------------------------

WEATHER_CLASS_LABELS_FALLBACK = ['Fog', 'Mist', 'Night', 'Rain', 'Snowy', 'Sunny']

# Portable default assets directory — expands to current user's home, no hardcoded name
_ASSETS_DIR = os.path.join(os.path.expanduser('~'), 'ros2_ws', 'assets', 'pt_files')

# ImageNet normalization parameters for EfficientNetV2-M
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Default video path (test images bundled with workspace)
_DEFAULT_VIDEO_PATH = os.path.join(
    os.path.expanduser('~'), 'ros2_ws', 'assets', 'test_images')

# HUD panel width in pixels (appended to right of frame)
_HUD_PANEL_WIDTH = 300

# OpenCV display target height (pixels)
_DISPLAY_HEIGHT = 600


# -------------------------------------------------------------------------------------
#  LightweightRestorer Network
# -------------------------------------------------------------------------------------

class LightweightRestorer(nn.Module):
    """
    Custom lightweight residual image restoration network.
    Architecture: Input → Conv2d(3→64,k=3,p=1) → ReLU → Conv2d(64→3,k=3,p=1) → + Input
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64,  3, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


# -------------------------------------------------------------------------------------
#  PerceptionOfflineNode
# -------------------------------------------------------------------------------------

class PerceptionOfflineNode(Node):

    def __init__(self) -> None:
        super().__init__('perception_offline_node')

        # ── ROS2 Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('video_path',             _DEFAULT_VIDEO_PATH)
        self.declare_parameter('restorer_weights_path',
                               os.path.join(_ASSETS_DIR, 'restorer.pt'))
        self.declare_parameter('classifier_weights_path',
                               os.path.join(_ASSETS_DIR, 'best_effi.pt'))
        self.declare_parameter('show_display', True)
        self.declare_parameter('save_video', True)

        self.video_path              = self.get_parameter(
            'video_path').get_parameter_value().string_value
        self.restorer_weights_path   = self.get_parameter(
            'restorer_weights_path').get_parameter_value().string_value
        self.classifier_weights_path = self.get_parameter(
            'classifier_weights_path').get_parameter_value().string_value
        self.show_display            = self.get_parameter(
            'show_display').get_parameter_value().bool_value
        self.save_video              = self.get_parameter(
            'save_video').get_parameter_value().bool_value

        # ── Device Selection ────────────────────────────────────────────────────────
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Compute device: {self.device}')

        # Pre-allocate ImageNet normalization tensors on the target device
        self._norm_mean = torch.tensor(
            _IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self._norm_std  = torch.tensor(
            _IMAGENET_STD,  dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

        # ── Model Initialization ────────────────────────────────────────────────────
        if not self._init_models():
            raise RuntimeError(
                'perception_offline_node: One or more models failed to load. '
                'Check the log above for FATAL messages and verify weight file paths.')

        # ── Telemetry & Runtime State ───────────────────────────────────────────────
        self._telemetry  = TelemetryLogger()
        self.frame_count = 0
        self._fps        = 0.0
        self._prev_wall  = time.time()
        self.video_writer = None

        # ── CV Bridge ──────────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Input Source (video file or image sequence directory) ───────────────────
        self.is_image_sequence = os.path.isdir(self.video_path)
        self.cap               = None

        if self.is_image_sequence:
            valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
            self.image_files = sorted([
                os.path.join(self.video_path, f)
                for f in os.listdir(self.video_path)
                if f.lower().endswith(valid_ext)
                and not f.endswith(':Zone.Identifier')
            ])
            if not self.image_files:
                raise FileNotFoundError(
                    f'No valid images found in directory: {self.video_path}')
            self.image_index = 0
            self.get_logger().info(
                f'Loaded image sequence: {len(self.image_files)} frames from '
                f'{self.video_path}')
        else:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.get_logger().fatal(
                    f'Failed to open video file: {self.video_path}')
                raise FileNotFoundError(
                    f'Video file not found or unreadable: {self.video_path}')
            self.get_logger().info(f'Video file opened: {self.video_path}')

        # ── Publishers ─────────────────────────────────────────────────────────────
        self.pub_cleaned = self.create_publisher(Image, '/camera/front/cleaned', 10)
        self.pub_hud     = self.create_publisher(Image, '/perception/telemetry_hud', 10)

        # ── 30 FPS Timer ───────────────────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info(
            'PerceptionOfflineNode initialized — 30 FPS timer started.')

    # ─────────────────────────────────────────────────────────────────────────────
    #  Model Loading
    # ─────────────────────────────────────────────────────────────────────────────

    def _load_restorer(self) -> bool:
        """Load LightweightRestorer weights. Returns True on success."""
        path = self.restorer_weights_path
        if not os.path.isfile(path):
            self.get_logger().fatal(
                f'[RESTORER] Weight file not found: {path}\n'
                f'  → Provide the file or override the "restorer_weights_path" parameter.')
            return False
        try:
            self.restorer = LightweightRestorer().to(self.device)
            ckpt  = torch.load(path, map_location=self.device, weights_only=False)
            state = ckpt['model_state'] if (
                isinstance(ckpt, dict) and 'model_state' in ckpt) else ckpt
            self.restorer.load_state_dict(state)
            self.restorer.eval()
            self.get_logger().info(f'[RESTORER] Loaded from: {path}')
            return True
        except Exception as exc:
            self.get_logger().fatal(f'[RESTORER] Load failed — {exc}')
            return False

    def _load_classifier(self) -> bool:
        """Load EfficientNetV2-M weather classifier. Returns True on success."""
        path = self.classifier_weights_path
        if not os.path.isfile(path):
            self.get_logger().fatal(
                f'[CLASSIFIER] Weight file not found: {path}\n'
                f'  → Provide best_effi.pt or override "classifier_weights_path".')
            return False
        try:
            ckpt  = torch.load(path, map_location=self.device, weights_only=False)
            state = ckpt['model_state'] if (
                isinstance(ckpt, dict) and 'model_state' in ckpt) else ckpt

            # Read embedded class labels, fallback to hardcoded list
            if isinstance(ckpt, dict) and 'classes' in ckpt:
                self.weather_class_labels = [c.capitalize() for c in ckpt['classes']]
                self.get_logger().info(
                    f'[CLASSIFIER] Labels from checkpoint: {self.weather_class_labels}')
            else:
                self.weather_class_labels = WEATHER_CLASS_LABELS_FALLBACK
                self.get_logger().warn(
                    '[CLASSIFIER] No "classes" key — '
                    f'using fallback: {self.weather_class_labels}')

            # Build EfficientNetV2-M and adapt the classification head
            self.classifier = models.efficientnet_v2_m()
            if 'classifier.1.weight' in state:
                num_classes = state['classifier.1.weight'].size(0)
                if num_classes != self.classifier.classifier[1].out_features:
                    self.classifier.classifier[1] = nn.Linear(
                        self.classifier.classifier[1].in_features, num_classes)

            # Filtered load — tolerant of minor shape mismatches
            model_sd = self.classifier.state_dict()
            filtered = {k: v for k, v in state.items()
                        if k in model_sd and v.shape == model_sd[k].shape}
            n_skipped = len(state) - len(filtered)
            if n_skipped > 0:
                self.get_logger().warn(
                    f'[CLASSIFIER] Skipped {n_skipped} keys (shape mismatch).')

            self.classifier.load_state_dict(filtered, strict=False)
            self.classifier = self.classifier.to(self.device)
            self.classifier.eval()
            self.get_logger().info(
                f'[CLASSIFIER] EfficientNetV2-M loaded from: {path}')
            return True
        except Exception as exc:
            self.get_logger().fatal(f'[CLASSIFIER] Load failed — {exc}')
            return False

    def _init_models(self) -> bool:
        ok_r = self._load_restorer()
        ok_c = self._load_classifier()
        return ok_r and ok_c

    # ─────────────────────────────────────────────────────────────────────────────
    #  Preprocessing Helpers
    # ─────────────────────────────────────────────────────────────────────────────

    def _preprocess_for_classifier(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Prepare an OpenCV BGR frame for EfficientNetV2-M inference.
        Steps: uint8→float32 [0,1] → RGB → 512×512 → ImageNet normalize.
        Returns Tensor (1, 3, 512, 512) on self.device.
        """
        t = torch.from_numpy(frame_bgr).permute(2, 0, 1).float().div(255.0)
        t = t.unsqueeze(0).to(self.device)
        t = t[:, [2, 1, 0], :, :]   # BGR → RGB
        t = F_nn.interpolate(t, size=(512, 512), mode='bilinear', align_corners=False)
        return (t - self._norm_mean) / self._norm_std

    def _preprocess_for_restorer(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Prepare raw BGR frame for LightweightRestorer (no color conversion).
        Returns Tensor (1, 3, H, W) on self.device in [0.0, 1.0].
        """
        t = torch.from_numpy(frame_bgr).permute(2, 0, 1).float().div(255.0)
        return t.unsqueeze(0).to(self.device)

    # ─────────────────────────────────────────────────────────────────────────────
    #  Frame Acquisition
    # ─────────────────────────────────────────────────────────────────────────────

    def _read_next_frame(self):
        """Read the next frame from the active source. Returns BGR ndarray or None."""
        if self.is_image_sequence:
            if self.image_index >= len(self.image_files):
                self.image_index = 0   # Loop infinitely
            img_path = self.image_files[self.image_index]
            frame    = cv2.imread(img_path)
            self.image_index += 1
            if frame is None:
                self.get_logger().error(
                    f'Failed to read image: {img_path} — skipping.')
            return frame
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    self.get_logger().error('End of video — loop reset failed.')
                    return None
            return frame

    # ─────────────────────────────────────────────────────────────────────────────
    #  Timer Callback (30 FPS)
    # ─────────────────────────────────────────────────────────────────────────────

    def timer_callback(self) -> None:
        t_total_start = time.perf_counter()

        # ── Step 1: Acquire Frame ─────────────────────────────────────────────────
        frame = self._read_next_frame()
        if frame is None:
            return
        self.frame_count += 1

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── Step 2: Image Metrics for Stage-B Gating ─────────────────────────
            mean_brightness  = float(np.mean(gray))
            laplacian_var    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            white_clip_ratio = float(np.sum(gray > 240)) / float(gray.size)

            # ── Step 3: Weather Classification (Stage A) ──────────────────────────
            t_cls      = time.perf_counter()
            cls_tensor = self._preprocess_for_classifier(frame)
            with torch.no_grad():
                logits       = self.classifier(cls_tensor)
                probs        = torch.softmax(logits, dim=1)
                conf, idx    = torch.max(probs, dim=1)
                pred_idx     = int(idx.item())
                conf_val     = float(conf.item())
                conf_profile = probs.squeeze(0).cpu().numpy().tolist()
            classifier_latency_ms = (time.perf_counter() - t_cls) * 1000.0

            weather_label = (
                self.weather_class_labels[pred_idx]
                if pred_idx < len(self.weather_class_labels)
                else f'Unknown({pred_idx})'
            )

            # ── Step 4: Dual-Layer Gating ─────────────────────────────────────────
            classifier_adverse = weather_label.lower() != 'sunny'
            gating_restore, gating_reason = check_gating(
                mean_brightness, laplacian_var, white_clip_ratio)
            route_to_restorer = classifier_adverse or gating_restore

            # ── Step 5: Conditional Restoration ──────────────────────────────────
            restorer_latency_ms = 0.0
            if route_to_restorer:
                t_rest = time.perf_counter()
                rest_t = self._preprocess_for_restorer(frame)
                with torch.no_grad():
                    restored = torch.clamp(self.restorer(rest_t), 0.0, 1.0)
                final_frame = (
                    restored.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
                ).astype(np.uint8)
                restorer_latency_ms = (time.perf_counter() - t_rest) * 1000.0
                route = 'RESTORED'
            else:
                final_frame   = frame
                gating_reason = 'bypass'
                route         = 'BYPASS'

            # ── Step 6: CWQI Severity Scoring ─────────────────────────────────────
            cwqi_result = compute_cwqi(gray)

            # ── Step 7: Publish Cleaned Frame ─────────────────────────────────────
            msg               = self.bridge.cv2_to_imgmsg(final_frame, encoding='bgr8')
            msg.header.stamp  = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_front_link'
            self.pub_cleaned.publish(msg)

            # ── Step 8: Render & Publish Telemetry HUD ───────────────────────────
            total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0
            now = time.time()
            dt  = now - self._prev_wall
            self._fps       = (1.0 / dt) if dt > 0.0 else 0.0
            self._prev_wall = now

            weather_entropy = compute_weather_entropy(conf_profile)

            hud_data = {
                'frame_num':             self.frame_count,
                'weather_class':         weather_label,
                'confidence':            conf_val * 100.0,
                'weather_entropy':       weather_entropy,
                'gating_reason':         gating_reason,
                'route':                 route,
                'cwqi':                  cwqi_result['cwqi'],
                'severity_level':        cwqi_result['severity_level'],
                'severity_label':        cwqi_result['severity_label'],
                'classifier_latency_ms': classifier_latency_ms,
                'restorer_latency_ms':   restorer_latency_ms,
                'total_latency_ms':      total_latency_ms,
                'fps':                   self._fps,
                'device':                str(self.device),
            }

            hud_frame = render_hud_panel(final_frame.copy(), _HUD_PANEL_WIDTH, hud_data)

            # Save HUD telemetry frames to video and images if enabled
            if self.save_video:
                frames_dir = os.path.expanduser('~/ros2_ws/results/frames')
                os.makedirs(frames_dir, exist_ok=True)
                cv2.imwrite(os.path.join(frames_dir, f'frame_{self.frame_count:05d}.png'), hud_frame)

                if self.video_writer is None:
                    h_hud, w_hud = hud_frame.shape[:2]
                    os.makedirs(os.path.expanduser('~/ros2_ws/results'), exist_ok=True)
                    video_out_path = os.path.expanduser('~/ros2_ws/results/hud_output.avi')
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.video_writer = cv2.VideoWriter(video_out_path, fourcc, 10.0, (w_hud, h_hud))
                    self.get_logger().info(f'Saving HUD telemetry video to: {video_out_path}')
                self.video_writer.write(hud_frame)

            # Publish HUD as ROS2 Image topic
            hud_msg               = self.bridge.cv2_to_imgmsg(hud_frame, encoding='bgr8')
            hud_msg.header.stamp  = msg.header.stamp
            hud_msg.header.frame_id = 'camera_front_link'
            self.pub_hud.publish(hud_msg)

            # ── Step 9: Async Telemetry Logging ───────────────────────────────────
            self._telemetry.log({
                'frame_id':               str(self.frame_count),
                'weather_label':          weather_label,
                'weather_confidence':     f'{conf_val:.4f}',
                'weather_entropy':        f'{weather_entropy:.4f}',
                'gating_reason':          gating_reason,
                'route':                  route,
                'cwqi':                   f"{cwqi_result['cwqi']:.4f}",
                'severity_level':         cwqi_result['severity_level'],
                'classifier_latency_ms':  f'{classifier_latency_ms:.2f}',
                'restorer_latency_ms':    f'{restorer_latency_ms:.2f}',
                'total_latency_ms':       f'{total_latency_ms:.2f}',
            })

            # ── Step 10: Console Log ──────────────────────────────────────────────
            self.get_logger().info(
                f'Frame {self.frame_count:05d} | '
                f'Latency: {total_latency_ms:.1f} ms | '
                f'Route: {route} ({gating_reason}) | '
                f'Weather: {weather_label} ({conf_val * 100:.1f}%) | '
                f'CWQI: {cwqi_result["cwqi"]:.3f} '
                f'[L{cwqi_result["severity_level"]} {cwqi_result["severity_label"]}]'
            )

            # ── Step 11: Optional OpenCV Display Window ───────────────────────────
            if self.show_display:
                h, w = hud_frame.shape[:2]
                scale    = _DISPLAY_HEIGHT / h
                disp_w   = int(w * scale)
                disp_frm = cv2.resize(hud_frame, (disp_w, _DISPLAY_HEIGHT))
                cv2.imshow('TH OWL — Perception HUD', disp_frm)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):  # q or ESC
                    self.get_logger().info('Display closed by user — shutting down.')
                    rclpy.shutdown()

        except Exception as exc:
            self.get_logger().error(f'timer_callback error on frame '
                                    f'{self.frame_count}: {exc}')

    # ─────────────────────────────────────────────────────────────────────────────
    #  Shutdown
    # ─────────────────────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        self.get_logger().info(
            'Shutting down perception_offline_node — flushing telemetry log.')
        self._telemetry.shutdown()
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info('VideoWriter released and video saved.')
        if self.cap is not None and isinstance(self.cap, cv2.VideoCapture):
            if self.cap.isOpened():
                self.cap.release()
                self.get_logger().info('VideoCapture released.')
        cv2.destroyAllWindows()
        super().destroy_node()


# -------------------------------------------------------------------------------------
#  Entry Point
# -------------------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionOfflineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            'Shutting down perception_offline_node via KeyboardInterrupt.')
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
