#!/usr/bin/env python3
"""
TH OWL — Adverse Weather Perception Node (CARLA Live Mode)
==========================================================
Subscribes to the CARLA simulator front-camera topic, applies the full
two-stage adverse weather perception pipeline, and publishes a cleaned
image stream for downstream YOLO object detection.

Pipeline per frame:
  1. Decode ROS2 Image → OpenCV BGR
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

Authors:  TH OWL Project 8
Node:     perception_carla_node
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

# Fallback weather class labels if checkpoint does not embed a 'classes' key
WEATHER_CLASS_LABELS_FALLBACK = ['Fog', 'Mist', 'Night', 'Rain', 'Snowy', 'Sunny']

# Portable default assets directory — expands to current user's home, no hardcoded name
_ASSETS_DIR = os.path.join(os.path.expanduser('~'), 'ros2_ws', 'assets', 'pt_files')

# ImageNet normalization parameters for EfficientNetV2-M
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# HUD panel width in pixels (appended to right of cleaned frame)
_HUD_PANEL_WIDTH = 300


# -------------------------------------------------------------------------------------
#  LightweightRestorer Network
# -------------------------------------------------------------------------------------

class LightweightRestorer(nn.Module):
    """
    Custom lightweight residual image restoration network.
    Architecture: Input → Conv2d(3→64,k=3,p=1) → ReLU → Conv2d(64→3,k=3,p=1) → + Input
    The residual skip connection ensures the output retains the original scene
    structure; the two-conv branch learns only the correction delta.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64,  3, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


# -------------------------------------------------------------------------------------
#  PerceptionCarlaNode
# -------------------------------------------------------------------------------------

class PerceptionCarlaNode(Node):

    def __init__(self) -> None:
        super().__init__('perception_carla_node')

        # ── ROS2 Parameters ────────────────────────────────────────────────────────
        self.declare_parameter(
            'restorer_weights_path',
            os.path.join(_ASSETS_DIR, 'restorer.pt'),
        )
        self.declare_parameter(
            'classifier_weights_path',
            os.path.join(_ASSETS_DIR, 'best_effi.pt'),
        )

        self.restorer_weights_path   = self.get_parameter(
            'restorer_weights_path').get_parameter_value().string_value
        self.classifier_weights_path = self.get_parameter(
            'classifier_weights_path').get_parameter_value().string_value

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
                'perception_carla_node: One or more models failed to load. '
                'Check the log above for FATAL messages and verify weight file paths.')

        # ── Telemetry & Runtime State ───────────────────────────────────────────────
        self._telemetry  = TelemetryLogger()
        self._fps        = 0.0
        self._prev_wall  = time.time()

        # ── CV Bridge ──────────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Subscriptions ──────────────────────────────────────────────────────────
        self.subscription = self.create_subscription(
            Image,
            '/carla/ego_vehicle/rgb_front/image',
            self.image_callback,
            10,
        )
        self.get_logger().info(
            "Subscribed to: '/carla/ego_vehicle/rgb_front/image'")

        # ── Publishers ─────────────────────────────────────────────────────────────
        self.pub_cleaned = self.create_publisher(Image, '/camera/front/cleaned', 10)
        self.pub_hud     = self.create_publisher(Image, '/perception/telemetry_hud', 10)

        self.get_logger().info(
            "Publishers ready: '/camera/front/cleaned', '/perception/telemetry_hud'")
        self.get_logger().info(
            'PerceptionCarlaNode initialized — ready to process frames.')

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
                    '[CLASSIFIER] No "classes" key in checkpoint — '
                    f'using fallback: {self.weather_class_labels}')

            # Build EfficientNetV2-M and adapt the classification head
            self.classifier = models.efficientnet_v2_m()
            if 'classifier.1.weight' in state:
                num_classes = state['classifier.1.weight'].size(0)
                if num_classes != self.classifier.classifier[1].out_features:
                    self.classifier.classifier[1] = nn.Linear(
                        self.classifier.classifier[1].in_features, num_classes)

            # Filtered load — tolerant of minor shape mismatches across versions
            model_sd = self.classifier.state_dict()
            filtered = {k: v for k, v in state.items()
                        if k in model_sd and v.shape == model_sd[k].shape}
            n_skipped = len(state) - len(filtered)
            if n_skipped > 0:
                self.get_logger().warn(
                    f'[CLASSIFIER] Skipped {n_skipped} keys due to shape mismatch.')

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

        Steps:
          1. HWC uint8 → CHW float32, normalized to [0.0, 1.0]
          2. BGR → RGB channel swap (EfficientNet trained on RGB)
          3. Bilinear resize to 512×512
          4. ImageNet mean/std normalization

        Returns:
            Tensor (1, 3, 512, 512) on self.device — ready for model forward pass.
        """
        t = torch.from_numpy(frame_bgr).permute(2, 0, 1).float().div(255.0)
        t = t.unsqueeze(0).to(self.device)
        # BGR → RGB
        t = t[:, [2, 1, 0], :, :]
        # Resize
        t = F_nn.interpolate(t, size=(512, 512), mode='bilinear', align_corners=False)
        # ImageNet normalization
        t = (t - self._norm_mean) / self._norm_std
        return t

    def _preprocess_for_restorer(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Prepare an OpenCV BGR frame for LightweightRestorer inference.

        The restorer is trained on raw BGR [0, 1] tensors at original resolution.
        No color conversion or normalization is applied.

        Returns:
            Tensor (1, 3, H, W) on self.device in [0.0, 1.0].
        """
        t = torch.from_numpy(frame_bgr).permute(2, 0, 1).float().div(255.0)
        return t.unsqueeze(0).to(self.device)

    # ─────────────────────────────────────────────────────────────────────────────
    #  Main Image Callback
    # ─────────────────────────────────────────────────────────────────────────────

    def image_callback(self, msg: Image) -> None:
        t_total_start = time.perf_counter()

        try:
            # ── Step 1: Decode ────────────────────────────────────────────────────
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── Step 2: Image Metrics for Stage-B Gating ─────────────────────────
            mean_brightness  = float(np.mean(gray))
            laplacian_var    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            white_clip_ratio = float(np.sum(gray > 240)) / float(gray.size)

            # ── Step 3: Weather Classification (Stage A) ──────────────────────────
            t_cls = time.perf_counter()
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
            # Stage A: classifier verdict
            classifier_adverse = weather_label.lower() != 'sunny'
            # Stage B: image metric thresholds (check_gating)
            gating_restore, gating_reason = check_gating(
                mean_brightness, laplacian_var, white_clip_ratio)
            # Conservative union — restore if either stage flags degradation
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
            out_msg        = self.bridge.cv2_to_imgmsg(final_frame, encoding='bgr8')
            out_msg.header = msg.header   # preserve simulator timestamp
            self.pub_cleaned.publish(out_msg)

            # ── Step 8: Render & Publish Telemetry HUD ───────────────────────────
            total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0
            now = time.time()
            dt  = now - self._prev_wall
            self._fps       = (1.0 / dt) if dt > 0.0 else 0.0
            self._prev_wall = now

            weather_entropy = compute_weather_entropy(conf_profile)

            hud_data = {
                'frame_num':             msg.header.frame_id,
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

            hud_frame      = render_hud_panel(final_frame.copy(), _HUD_PANEL_WIDTH, hud_data)
            hud_msg        = self.bridge.cv2_to_imgmsg(hud_frame, encoding='bgr8')
            hud_msg.header = msg.header
            self.pub_hud.publish(hud_msg)

            # ── Step 9: Async Telemetry Logging ───────────────────────────────────
            self._telemetry.log({
                'frame_id':               msg.header.frame_id,
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
                f'Latency: {total_latency_ms:.1f} ms | '
                f'Route: {route} ({gating_reason}) | '
                f'Weather: {weather_label} ({conf_val * 100:.1f}%) | '
                f'CWQI: {cwqi_result["cwqi"]:.3f} '
                f'[L{cwqi_result["severity_level"]} {cwqi_result["severity_label"]}]'
            )

        except Exception as exc:
            self.get_logger().error(f'image_callback failed: {exc}')

    # ─────────────────────────────────────────────────────────────────────────────
    #  Shutdown
    # ─────────────────────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        self.get_logger().info(
            'Shutting down perception_carla_node — flushing telemetry log.')
        self._telemetry.shutdown()
        super().destroy_node()


# -------------------------------------------------------------------------------------
#  Entry Point
# -------------------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionCarlaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            'Shutting down perception_carla_node via KeyboardInterrupt.')
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
