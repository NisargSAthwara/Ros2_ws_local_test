#!/usr/bin/env python3
"""
=====================================================================================
  TH OWL — Adverse Weather Perception Pipeline (Dataset Evaluation Mode)
=====================================================================================
  Deployment script: dataset_perception_pipeline.py

  Purpose:
      Replaces the live multi-machine CARLA simulator network loop with a localized,
      file-based dataset evaluation paradigm. Streams frames from an .mp4 video or
      a directory of sequentially numbered images, runs the full perception stack
      (Weather Classification → Gating → Restoration → Object Detection), computes
      analytical severity metrics, and renders a high-fidelity telemetry HUD.

  Usage:
      python3 dataset_perception_pipeline.py /absolute/path/to/video.mp4
      python3 dataset_perception_pipeline.py /absolute/path/to/image_sequence_dir/

  Local weight files expected at:
      ~/ros2_ws/assets/pt_files/fully_unified_all_weather_restorer.pt
      ~/ros2_ws/assets/pt_files/weather_classifier_resnet50.pt
      ~/ros2_ws/assets/pt_files/yolov8n.pt
=====================================================================================
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models

# Append the directory containing severity.py so it can be imported directly
# (Same sys.path pattern used by perception_carla_node.py / perception_offline_node.py for gating.py)
sys.path.append(os.path.expanduser('~/ros2_ws/assets/pt_files'))
import severity

# -------------------------------------------------------------------------------------
#  YOLO Import Guard (soft-fail — pipeline runs without detection if unavailable)
# -------------------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None   # Placeholder so type hints don't break
    print(
        "[WARN] 'ultralytics' library not found — YOLOv8 detection DISABLED.\n"
        "       Install via: pip install ultralytics\n"
        "       The pipeline will still run weather classification, gating,\n"
        "       severity estimation, and image restoration."
    )

# =====================================================================================
#  Section 1: Neural Network Architecture Definitions
#  (Preserved from existing local codebase — perception_carla_node.py / perception_offline_node.py)
# =====================================================================================

class LightweightRestorer(nn.Module):
    """
    Custom lightweight image restoration network with residual skip connection.
    Architecture: Input → Conv2d(3→64) → ReLU → Conv2d(64→3) → + Input (residual)
    Preserved exactly from local ~/ros2_ws/src/yolo_perception_pkg/ source files.
    """
    def __init__(self):
        super(LightweightRestorer, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


# =====================================================================================
#  Section 2: Configuration Constants
# =====================================================================================

# Default model weight file paths
DEFAULT_RESTORER_PATH = os.path.expanduser(
    '~/ros2_ws/assets/pt_files/restorer.pt'
)
DEFAULT_CLASSIFIER_PATH = os.path.expanduser(
    '~/ros2_ws/assets/pt_files/classifier.pt'
)
DEFAULT_YOLO_PATH = os.path.expanduser(
    '~/ros2_ws/assets/pt_files/yolov8n.pt'
)

# Weather class label map (must match training order of weather_classifier_resnet50.pt)
WEATHER_CLASS_LABELS = ["Fog", "Mist", "Night", "Rain", "Snowy", "Sunny"]

# Gating threshold constants (preserved from local gating.py)
LOW_LIGHT_THRESHOLD = 50.0     # Mean brightness below 50 → night / low-light
HIGH_GLARE_THRESHOLD = 0.05    # > 5% pixels clipped above 240 → significant glare
BLUR_THRESHOLD = 100.0         # Laplacian variance below 100 → heavy blur / fog

# Clear-day Laplacian variance baseline for severity mapping
CLEAR_DAY_LAPLACIAN_BASELINE = 500.0

# Target streaming FPS for image sequence mode
TARGET_FPS = 30.0

# YOLO detection confidence floor
YOLO_CONFIDENCE_THRESHOLD = 0.5

# HUD rendering configuration
HUD_PANEL_WIDTH = 380
HUD_PANEL_ALPHA = 0.65
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SCALE = 0.52
HUD_LINE_THICKNESS = 1
HUD_TEXT_COLOR = (255, 255, 255)          # White
HUD_HEADER_COLOR = (0, 220, 255)         # Amber/Gold
HUD_BYPASS_COLOR = (0, 255, 128)         # Green
HUD_RESTORED_COLOR = (80, 180, 255)      # Orange-ish
HUD_SEVERITY_LOW_COLOR = (0, 255, 0)     # Green
HUD_SEVERITY_MED_COLOR = (0, 200, 255)   # Yellow
HUD_SEVERITY_HIGH_COLOR = (0, 80, 255)   # Red

# YOLO bounding box drawing colors (BGR) — one per COCO supercategory for visual clarity
DETECTION_BOX_COLOR = (0, 255, 0)
DETECTION_TEXT_COLOR = (255, 255, 255)
DETECTION_TEXT_BG_COLOR = (0, 180, 0)


# =====================================================================================
#  Section 3: Model Loading Utilities
#  (Checkpoint loading logic preserved from local perception_carla_node.py and
#   perception_offline_node.py — handles both 'model_state' dict wrappers and
#   raw state_dicts, plus fc.1.weight sequential dropout variants)
# =====================================================================================

def load_restorer(weights_path: str, device: torch.device) -> LightweightRestorer:
    """
    Initialize and load the LightweightRestorer network with pre-trained weights.
    Handles both raw state_dict and {'model_state': ...} wrapper checkpoint formats.
    """
    model = LightweightRestorer().to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INIT] LightweightRestorer loaded from: {weights_path}")
    return model


def load_classifier(weights_path: str, device: torch.device) -> nn.Module:
    """
    Initialize and load the Weather Classifier with pre-trained weights.
    Handles multiple checkpoint formats and architectures:
      - ResNet50 ('fc.1.weight' or 'fc.weight')
      - EfficientNet ('classifier.1.weight')
      - 'model_state' wrapper dictionaries
    """
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint

    global WEATHER_CLASS_LABELS
    if isinstance(checkpoint, dict) and 'classes' in checkpoint:
        WEATHER_CLASS_LABELS = [c.capitalize() for c in checkpoint['classes']]
        print(f"[INIT] Overriding classes from checkpoint: {WEATHER_CLASS_LABELS}")

    keys = list(state_dict.keys())
    is_efficientnet = any('classifier.1' in k for k in keys) or 'efficientnet' in weights_path.lower()

    if is_efficientnet:
        classifier = models.efficientnet_v2_m()
        if 'classifier.1.weight' in state_dict:
            num_classes = state_dict['classifier.1.weight'].size(0)
            if num_classes != 1000:
                classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, num_classes)
        classifier_name = "efficientnet_v2_m"
    else:
        classifier = models.resnet50()
        # Adapt the final classification head based on checkpoint structure
        if 'fc.1.weight' in state_dict:
            num_classes = state_dict['fc.1.weight'].size(0)
            classifier.fc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(2048, num_classes)
            )
        elif 'fc.weight' in state_dict:
            num_classes = state_dict['fc.weight'].size(0)
            if num_classes != 1000:
                classifier.fc = nn.Linear(classifier.fc.in_features, num_classes)
        classifier_name = "resnet50"

    # Filter state dict for shape mismatch
    model_state_dict = classifier.state_dict()
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            filtered_state[k] = v
        else:
            print(f"[INIT] Skipping {k} due to shape mismatch or missing key.")
    
    classifier.load_state_dict(filtered_state, strict=False)
    classifier = classifier.to(device)
    classifier.eval()
    print(f"[INIT] {classifier_name} Weather Classifier loaded from: {weights_path}")
    return classifier


def load_yolo(weights_path: str):
    """
    Initialize the YOLOv8 object detection model.
    Returns None if ultralytics is not installed or weights file is missing.
    Preserved from local yolo_detector_node.py.
    """
    if not YOLO_AVAILABLE:
        print("[INIT] YOLOv8 SKIPPED — ultralytics library not available.")
        return None
    if not os.path.isfile(weights_path):
        print(f"[INIT] YOLOv8 SKIPPED — weights file not found: {weights_path}")
        return None
    model = YOLO(weights_path)
    print(f"[INIT] YOLOv8 model loaded from: {weights_path}")
    return model


# =====================================================================================
#  Section 4: Multi-Format Dataset Streamer
# =====================================================================================

class DatasetStreamer:
    """
    Unified frame streamer that accepts either:
      - An .mp4 video file  → streamed via cv2.VideoCapture
      - A directory folder  → parses and sorts .png/.jpg images via glob,
                               throttled to 30 FPS execution rate

    Loops indefinitely by resetting the frame tracker to index 0 on exhaustion.
    """

    def __init__(self, input_path: str):
        self.input_path = os.path.abspath(input_path)
        self.is_video = False
        self.is_image_sequence = False
        self.cap = None
        self.image_files = []
        self.image_index = 0
        self.frame_interval = 1.0 / TARGET_FPS  # ~33.33 ms per frame at 30 FPS

        if os.path.isfile(self.input_path):
            # Validate video file extension
            _, ext = os.path.splitext(self.input_path)
            if ext.lower() != '.mp4':
                raise ValueError(
                    f"Unsupported file format '{ext}'. Only .mp4 video files are supported.\n"
                    f"Path provided: {self.input_path}"
                )
            self.cap = cv2.VideoCapture(self.input_path)
            if not self.cap.isOpened():
                raise FileNotFoundError(
                    f"Failed to open video file: {self.input_path}"
                )
            self.is_video = True
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            native_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"[STREAMER] Video source opened: {self.input_path}")
            print(f"[STREAMER]   Total frames: {total_frames} | Native FPS: {native_fps:.1f}")

        elif os.path.isdir(self.input_path):
            # Glob for .png and .jpg files, then sort numerically
            png_files = glob.glob(os.path.join(self.input_path, '*.png'))
            jpg_files = glob.glob(os.path.join(self.input_path, '*.jpg'))
            jpeg_files = glob.glob(os.path.join(self.input_path, '*.jpeg'))
            all_files = png_files + jpg_files + jpeg_files

            if not all_files:
                raise FileNotFoundError(
                    f"No .png or .jpg images found in directory: {self.input_path}"
                )

            # Sort by filename to ensure sequential frame order
            self.image_files = sorted(all_files, key=lambda f: os.path.basename(f))
            self.image_index = 0
            self.is_image_sequence = True
            print(f"[STREAMER] Image sequence directory opened: {self.input_path}")
            print(f"[STREAMER]   Total images: {len(self.image_files)} | "
                  f"Throttled playback: {TARGET_FPS:.0f} FPS")

        else:
            raise FileNotFoundError(
                f"Input path does not exist or is not a valid file/directory:\n"
                f"  {self.input_path}"
            )

    def read_frame(self) -> np.ndarray:
        """
        Read the next frame from the active source.
        Returns a BGR numpy array (H, W, 3).
        Loops indefinitely on sequence/video exhaustion.
        """
        if self.is_video:
            ret, frame = self.cap.read()
            if not ret:
                # End of video reached — loop back to frame 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    raise RuntimeError(
                        "Video playback failed: Unable to read frames after loop reset."
                    )
            return frame

        elif self.is_image_sequence:
            if self.image_index >= len(self.image_files):
                # End of image sequence — reset to index 0
                self.image_index = 0

            img_path = self.image_files[self.image_index]
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[STREAMER] WARNING: Failed to read image: {img_path}, skipping.")
                self.image_index += 1
                return self.read_frame()  # Recursive retry on corrupt/missing file

            self.image_index += 1
            return frame

        else:
            raise RuntimeError("DatasetStreamer is not properly initialized.")

    def release(self):
        """Release all held resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("[STREAMER] VideoCapture resource released.")


# =====================================================================================
#  Section 5: Weather Classification & Gating Logic
#  (ResNet50 classifier-driven, replaces the old metric-only gating.py approach)
# =====================================================================================

def classify_weather(
    frame_tensor: torch.Tensor,
    classifier: nn.Module,
    device: torch.device
) -> tuple:
    """
    Run the ResNet50 Weather Classifier on a single frame tensor.

    Args:
        frame_tensor: Tensor of shape (1, 3, H, W), normalized [0.0, 1.0]
        classifier:   Loaded ResNet50 model in eval mode
        device:       Active torch device

    Returns:
        predicted_label (str):   Human-readable weather class string
        predicted_index (int):   Argmax class index
        confidence (float):      Softmax confidence of the predicted class (0.0–1.0)
        confidence_profile (list[float]): Full softmax probability vector
    """
    import torch.nn.functional as F_nn
    
    # Preprocess for Torchvision Classifier (BGR -> RGB, Resize 224x224, Normalize)
    rgb_tensor = frame_tensor[:, [2, 1, 0], :, :]
    resized_tensor = F_nn.interpolate(rgb_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    normalized_tensor = (resized_tensor - mean) / std

    with torch.no_grad():
        logits = classifier(normalized_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)
        predicted_index = int(predicted_index.item())
        confidence = float(confidence.item())
        confidence_profile = probabilities.squeeze(0).cpu().numpy().tolist()

    predicted_label = WEATHER_CLASS_LABELS[predicted_index] \
        if predicted_index < len(WEATHER_CLASS_LABELS) else f"Unknown({predicted_index})"

    return predicted_label, predicted_index, confidence, confidence_profile


def apply_gating_logic(weather_label: str) -> tuple:
    """
    Classifier-based gating switch logic:
      - "Sunny"                            → BYPASS (raw frame to detection)
      - "Fog", "Mist", "Night", "Rain", "Snowy" → RESTORE (run through LightweightRestorer)

    Returns:
        route_to_restorer (bool): True if frame should pass through the restorer.
        route_mode (str):         'BYPASS' or 'RESTORED' for HUD display.
    """
    # Match all possible adverse classes from different checkpoint versions
    adverse_conditions = {"Fog", "Mist", "Night", "Rain", "Snowy", "Rainy"}

    # Also make case-insensitive just in case
    if weather_label.capitalize() in adverse_conditions:
        return True, "RESTORED"
    else:
        return False, "BYPASS"


# =====================================================================================
#  Section 6: Script-Based Analytical Severity Estimation
#  Delegated to external module: ~/ros2_ws/assets/pt_files/severity.py
#  (Same pattern as gating.py — single authoritative source for severity logic)
# =====================================================================================

def compute_analytical_severity(
    gray_frame: np.ndarray,
    weather_label: str
) -> tuple:
    """
    Wrapper that delegates to the external severity.compute_severity() module.
    All core metric extraction and severity calculation logic lives in severity.py.

    Returns:
        severity_pct (float):       Severity percentage [0.0 – 100.0]
        laplacian_var (float):      Raw Laplacian variance value
        mean_brightness (float):    Mean grayscale intensity [0–255]
        white_clip_ratio (float):   Fraction of pixels > 240 [0.0–1.0]
    """
    result = severity.compute_severity(gray_frame, weather_label)
    return (
        result['severity_pct'],
        result['laplacian_var'],
        result['mean_brightness'],
        result['white_clip_ratio']
    )


# =====================================================================================
#  Section 7: Image Restoration Pipeline
# =====================================================================================

def restore_frame(
    frame_tensor: torch.Tensor,
    restorer: LightweightRestorer
) -> torch.Tensor:
    """
    Run a normalized frame tensor through the LightweightRestorer and clamp output.

    Args:
        frame_tensor: Tensor (1, 3, H, W) normalized to [0.0, 1.0]
        restorer:     Loaded LightweightRestorer in eval mode

    Returns:
        Restored tensor (1, 3, H, W) clamped to [0.0, 1.0]
    """
    with torch.no_grad():
        restored = restorer(frame_tensor)
        restored = torch.clamp(restored, 0.0, 1.0)
    return restored


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (1, 3, H, W) float tensor in [0.0, 1.0] back to a BGR uint8 numpy array.
    Preserved tensor convention from local perception nodes (CHW → HWC).
    """
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (img * 255.0).astype(np.uint8)


# =====================================================================================
#  Section 8: YOLOv8 Object Detection & Bounding Box Rendering
#  (Drawing conventions preserved from local yolo_detector_node.py)
# =====================================================================================

def run_yolo_detection(
    frame: np.ndarray,
    yolo_model: YOLO,
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD
) -> tuple:
    """
    Run YOLOv8 inference and draw bounding boxes, class strings, and confidence
    markers directly onto the frame.

    Args:
        frame:                BGR numpy array (H, W, 3)
        yolo_model:           Loaded YOLO model instance
        confidence_threshold: Minimum detection confidence

    Returns:
        annotated_frame (np.ndarray): Frame with detections drawn
        num_detections (int):         Count of detected objects
        detections_info (list):       List of dicts with detection metadata
    """
    results = yolo_model.predict(frame, conf=confidence_threshold, verbose=False)
    annotated_frame = frame.copy()
    detections_info = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract bounding box coordinates
            xyxy = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = map(int, xyxy)

            # Extract class and confidence
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = result.names.get(cls_id, f"cls_{cls_id}")

            # Draw bounding box rectangle
            cv2.rectangle(
                annotated_frame,
                (xmin, ymin), (xmax, ymax),
                DETECTION_BOX_COLOR, 2
            )

            # Compose label string: "ClassName 0.XX"
            label = f"{cls_name} {conf:.2f}"

            # Calculate text size for background fill
            (text_w, text_h), baseline = cv2.getTextSize(
                label, HUD_FONT, 0.50, 1
            )
            # Draw filled rectangle behind text for readability
            cv2.rectangle(
                annotated_frame,
                (xmin, ymin - text_h - baseline - 4),
                (xmin + text_w + 4, ymin),
                DETECTION_TEXT_BG_COLOR, -1
            )
            # Draw label text
            cv2.putText(
                annotated_frame, label,
                (xmin + 2, ymin - baseline - 2),
                HUD_FONT, 0.50, DETECTION_TEXT_COLOR,
                1, cv2.LINE_AA
            )

            detections_info.append({
                'class_id': cls_id,
                'class_name': cls_name,
                'confidence': conf,
                'bbox': (xmin, ymin, xmax, ymax)
            })

    num_detections = len(detections_info)
    return annotated_frame, num_detections, detections_info


# =====================================================================================
#  Section 9: High-Fidelity Telemetry HUD Overlay
# =====================================================================================

def render_hud_overlay(
    frame: np.ndarray,
    weather_label: str,
    weather_confidence: float,
    severity_pct: float,
    route_mode: str,
    latency_ms: float,
    num_detections: int,
    confidence_profile: list,
    laplacian_var: float,
    mean_brightness: float,
    white_clip_ratio: float,
    frame_index: int
) -> np.ndarray:
    """
    Render a semi-transparent black rectangle bar along the right edge of the video
    pane displaying live telemetry data.

    Displayed fields:
      - Active Weather Profile
      - Analytical Severity Percentage (color-coded)
      - Pipeline Routing Track (BYPASS / RESTORED)
      - Full processing inference latency in milliseconds
      - Detection count
      - Confidence profile breakdown
      - Raw analytical metrics
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    panel_width = min(HUD_PANEL_WIDTH, w // 3)

    # Draw semi-transparent black panel on the right edge
    panel_x_start = w - panel_width
    cv2.rectangle(overlay, (panel_x_start, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, HUD_PANEL_ALPHA, frame, 1.0 - HUD_PANEL_ALPHA, 0, frame)

    # --- Text rendering helper ---
    x_margin = panel_x_start + 14
    y_cursor = 30
    line_spacing = 24

    def draw_text(text, color=HUD_TEXT_COLOR, scale=HUD_FONT_SCALE, bold=False):
        nonlocal y_cursor
        thickness = 2 if bold else HUD_LINE_THICKNESS
        cv2.putText(frame, text, (x_margin, y_cursor),
                    HUD_FONT, scale, color, thickness, cv2.LINE_AA)
        y_cursor += line_spacing

    def draw_separator():
        nonlocal y_cursor
        cv2.line(frame, (x_margin, y_cursor - 10),
                 (w - 14, y_cursor - 10), (80, 80, 80), 1)
        y_cursor += 6

    # === HUD Content ===

    # Header
    draw_text("TH OWL PERCEPTION HUD", HUD_HEADER_COLOR, 0.55, bold=True)
    draw_separator()

    # Frame counter
    draw_text(f"Frame: {frame_index}")
    draw_separator()

    # Weather Profile
    draw_text("WEATHER PROFILE", HUD_HEADER_COLOR, 0.48, bold=True)
    draw_text(f"  Class: {weather_label}")
    draw_text(f"  Confidence: {weather_confidence * 100.0:.1f}%")
    draw_separator()

    # Analytical Severity
    draw_text("ANALYTICAL SEVERITY", HUD_HEADER_COLOR, 0.48, bold=True)
    if severity_pct < 25.0:
        sev_color = HUD_SEVERITY_LOW_COLOR
    elif severity_pct < 60.0:
        sev_color = HUD_SEVERITY_MED_COLOR
    else:
        sev_color = HUD_SEVERITY_HIGH_COLOR
    draw_text(f"  Severity: {severity_pct:.1f}%", sev_color)

    # Severity bar visualization
    bar_x = x_margin
    bar_y = y_cursor - 18
    bar_w = panel_width - 28
    bar_h = 10
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * min(severity_pct / 100.0, 1.0))
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), sev_color, -1)
    y_cursor += 4
    draw_separator()

    # Pipeline Routing
    draw_text("PIPELINE ROUTING", HUD_HEADER_COLOR, 0.48, bold=True)
    route_color = HUD_BYPASS_COLOR if route_mode == "BYPASS" else HUD_RESTORED_COLOR
    draw_text(f"  Track: {route_mode}", route_color, bold=True)
    draw_separator()

    # Inference Latency
    draw_text("PERFORMANCE", HUD_HEADER_COLOR, 0.48, bold=True)
    draw_text(f"  Latency: {latency_ms:.1f} ms")
    draw_text(f"  Detections: {num_detections}")
    draw_separator()



    return frame


# =====================================================================================
#  Section 10: Main Pipeline Loop
# =====================================================================================

def main():
    # ---------------------------------------------------------------------------------
    #  Argument Parsing
    # ---------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "TH OWL Adverse Weather Perception Pipeline — Dataset Evaluation Mode.\n"
            "Streams frames from a video file or image directory, runs weather classification,\n"
            "analytical severity estimation, conditional image restoration, and YOLOv8 detection\n"
            "with a live telemetry HUD overlay."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'input_path',
        type=str,
        help=(
            "Absolute path to either:\n"
            "  - An .mp4 video file\n"
            "  - A directory containing sequentially numbered .png/.jpg images"
        )
    )
    parser.add_argument(
        '--restorer-weights', type=str, default=DEFAULT_RESTORER_PATH,
        help=f"Path to restorer weights (default: {DEFAULT_RESTORER_PATH})"
    )
    parser.add_argument(
        '--classifier-weights', type=str, default=DEFAULT_CLASSIFIER_PATH,
        help=f"Path to classifier weights (default: {DEFAULT_CLASSIFIER_PATH})"
    )
    parser.add_argument(
        '--yolo-weights', type=str, default=DEFAULT_YOLO_PATH,
        help=f"Path to YOLOv8 weights (default: {DEFAULT_YOLO_PATH})"
    )
    parser.add_argument(
        '--confidence', type=float, default=YOLO_CONFIDENCE_THRESHOLD,
        help=f"YOLO detection confidence threshold (default: {YOLO_CONFIDENCE_THRESHOLD})"
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------------------------
    #  Device Selection (CUDA with safe CPU fallback)
    # ---------------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INIT] Active compute device: {device}")
    if device.type == 'cuda':
        print(f"[INIT]   GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INIT]   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---------------------------------------------------------------------------------
    #  Model Initialization
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  Loading Neural Network Models")
    print("=" * 72)

    restorer = load_restorer(args.restorer_weights, device)
    classifier = load_classifier(args.classifier_weights, device)
    yolo_model = load_yolo(args.yolo_weights)

    yolo_enabled = yolo_model is not None
    if yolo_enabled:
        print("[INIT] All models loaded successfully. Entering evaluation loop.\n")
    else:
        print("[INIT] Classifier + Restorer loaded. YOLO disabled. Entering evaluation loop.\n")

    # ---------------------------------------------------------------------------------
    #  Dataset Streamer Initialization
    # ---------------------------------------------------------------------------------
    streamer = DatasetStreamer(args.input_path)

    # ---------------------------------------------------------------------------------
    #  Main Processing Loop
    # ---------------------------------------------------------------------------------
    frame_index = 0
    window_name = "TH OWL — Adverse Weather Perception Pipeline"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    print("=" * 72)
    print("  Pipeline Active — Press 'q' or ESC to exit")
    print("=" * 72 + "\n")

    try:
        while True:
            loop_start = time.perf_counter()

            # ----- Step 1: Read next frame from dataset source -----
            frame = streamer.read_frame()
            frame_index += 1

            # ----- Step 2: Prepare tensor for classifier/restorer -----
            # Normalize BGR frame (H,W,C) → Tensor (1,C,H,W) in [0.0, 1.0]
            # Preserved tensor convention from local perception nodes
            tensor_bgr = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            tensor_bgr = tensor_bgr.unsqueeze(0).to(device)

            # ----- Step 3: Weather Classification (every frame) -----
            weather_label, weather_idx, weather_conf, conf_profile = \
                classify_weather(tensor_bgr, classifier, device)

            # ----- Step 4: Gating Switch Logic -----
            route_to_restorer, route_mode = apply_gating_logic(weather_label)

            # ----- Step 5: Conditional Image Restoration -----
            if route_to_restorer:
                # Adverse condition detected — run through LightweightRestorer
                restored_tensor = restore_frame(tensor_bgr, restorer)
                processing_frame = tensor_to_bgr(restored_tensor)
            else:
                # Sunny / Clear — bypass restorer, forward raw frame
                processing_frame = frame.copy()

            # ----- Step 6: Analytical Severity Estimation -----
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            severity_pct, laplacian_var, mean_brightness, white_clip_ratio = \
                compute_analytical_severity(gray, weather_label)

            # ----- Step 7: YOLOv8 Object Detection (skipped if model unavailable) -----
            if yolo_enabled:
                annotated_frame, num_detections, detections_info = \
                    run_yolo_detection(processing_frame, yolo_model, args.confidence)
            else:
                annotated_frame = processing_frame.copy()
                num_detections = 0
                detections_info = []

            # ----- Step 8: Compute total pipeline latency -----
            latency_ms = (time.perf_counter() - loop_start) * 1000.0

            # ----- Step 9: Render Telemetry HUD -----
            display_frame = render_hud_overlay(
                frame=annotated_frame,
                weather_label=weather_label,
                weather_confidence=weather_conf,
                severity_pct=severity_pct,
                route_mode=route_mode,
                latency_ms=latency_ms,
                num_detections=num_detections,
                confidence_profile=conf_profile,
                laplacian_var=laplacian_var,
                mean_brightness=mean_brightness,
                white_clip_ratio=white_clip_ratio,
                frame_index=frame_index
            )

            # ----- Step 10: Display output window -----
            cv2.imshow(window_name, display_frame)

            # ----- Console telemetry (throttled to every frame) -----
            print(
                f"[F{frame_index:06d}] "
                f"Weather: {weather_label:6s} ({weather_conf * 100:.1f}%) | "
                f"Severity: {severity_pct:5.1f}% | "
                f"Route: {route_mode:8s} | "
                f"Detections: {num_detections:3d} | "
                f"Latency: {latency_ms:7.1f} ms"
            )

            # ----- Throttle image sequence to target FPS -----
            if streamer.is_image_sequence:
                elapsed = time.perf_counter() - loop_start
                sleep_time = streamer.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # ----- Exit on 'q' or ESC keypress -----
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[EXIT] User requested shutdown.")
                break

    except KeyboardInterrupt:
        print("\n[EXIT] Pipeline terminated via KeyboardInterrupt.")

    finally:
        # Cleanup
        streamer.release()
        cv2.destroyAllWindows()
        print("[EXIT] All resources released. Pipeline shutdown complete.")


# =====================================================================================
#  Entry Point
# =====================================================================================
if __name__ == '__main__':
    main()
