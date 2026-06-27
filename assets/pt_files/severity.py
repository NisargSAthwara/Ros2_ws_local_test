"""
=====================================================================================
  TH OWL — Analytical Weather Severity Estimation Module
=====================================================================================
  File: severity.py
  Location: ~/ros2_ws/assets/pt_files/

  Computes weather degradation severity as a percentage using deterministic
  OpenCV metrics. No ML model required — purely script-based calculations.

  Design pattern matches the existing gating.py module in this directory.

  Metrics Used:
      - Laplacian Variance:       Edge sharpness / texture detail indicator
      - Grayscale Mean Brightness: Overall scene illumination level
      - White Pixel Clipping Ratio: Fraction of pixels with intensity > 240

  Usage:
      from severity import compute_severity
      result = compute_severity(gray_frame, weather_label)
=====================================================================================
"""

import cv2
import numpy as np


# -------------------------------------------------------------------------------------
#  Baseline & Threshold Configuration
# -------------------------------------------------------------------------------------

# Clear-day Laplacian variance baseline.
# This is the expected Laplacian variance for a well-lit, sharp, clear-weather frame.
# Severity for fog/rain/snow/mist is derived by measuring how far the current
# frame's Laplacian variance has dropped below this reference point.
# Tune this value based on your specific camera sensor and dataset characteristics.
CLEAR_DAY_LAPLACIAN_BASELINE = 500.0

# Low-light brightness threshold (0-255 grayscale).
# Frames with mean brightness below this value are classified as night/low-light.
LOW_LIGHT_THRESHOLD = 50.0

# White pixel clipping threshold (ratio 0.0–1.0).
# Frames with more than this fraction of pixels above intensity 240 exhibit
# significant glare (e.g., headlights, sun flare, overexposure).
HIGH_GLARE_THRESHOLD = 0.05


# -------------------------------------------------------------------------------------
#  Core Metric Extraction
# -------------------------------------------------------------------------------------

def extract_metrics(gray_frame: np.ndarray) -> dict:
    """
    Extract all analytical image quality metrics from a single grayscale frame.

    Args:
        gray_frame (np.ndarray): Single-channel grayscale image (H, W), dtype uint8.

    Returns:
        dict with keys:
            'laplacian_var'   (float): Variance of the Laplacian operator output.
            'mean_brightness' (float): Average pixel intensity [0.0–255.0].
            'white_clip_ratio'(float): Fraction of pixels with intensity > 240 [0.0–1.0].
            'total_pixels'    (int):   Total number of pixels in the frame.
    """
    laplacian_var = float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray_frame))

    total_pixels = gray_frame.size
    white_clipped_pixels = int(np.sum(gray_frame > 240))
    white_clip_ratio = float(white_clipped_pixels) / float(total_pixels)

    return {
        'laplacian_var': laplacian_var,
        'mean_brightness': mean_brightness,
        'white_clip_ratio': white_clip_ratio,
        'total_pixels': total_pixels
    }


# -------------------------------------------------------------------------------------
#  Severity Calculation Logic
# -------------------------------------------------------------------------------------

def _severity_from_laplacian(laplacian_var: float) -> float:
    """
    Derive severity from Laplacian variance drop against the clear-day baseline.

    Logic:
        severity = (1 - current_var / baseline_var) × 100
        - If current_var == baseline_var → 0% (no degradation)
        - If current_var == 0           → 100% (complete blur / whiteout)

    Returns:
        Severity percentage clamped to [0.0, 100.0].
    """
    if CLEAR_DAY_LAPLACIAN_BASELINE <= 0.0:
        return 0.0
    ratio = laplacian_var / CLEAR_DAY_LAPLACIAN_BASELINE
    return max(0.0, min(100.0, (1.0 - ratio) * 100.0))


def _severity_from_darkness(mean_brightness: float) -> float:
    """
    Derive night severity from mean brightness level.

    Logic:
        - Brightness at LOW_LIGHT_THRESHOLD → 0% severity
        - Brightness at 0 (pitch black)     → 100% severity

    Returns:
        Severity percentage clamped to [0.0, 100.0].
    """
    if mean_brightness >= LOW_LIGHT_THRESHOLD:
        return 0.0
    return max(0.0, min(100.0, (1.0 - mean_brightness / LOW_LIGHT_THRESHOLD) * 100.0))


def _severity_from_glare(white_clip_ratio: float) -> float:
    """
    Derive glare severity from the white pixel clipping ratio.

    Logic:
        - Clipping at HIGH_GLARE_THRESHOLD → 100% severity
        - Clipping at 0                    → 0% severity

    Returns:
        Severity percentage clamped to [0.0, 100.0].
    """
    if HIGH_GLARE_THRESHOLD <= 0.0:
        return 0.0
    return max(0.0, min(100.0, (white_clip_ratio / HIGH_GLARE_THRESHOLD) * 100.0))


# -------------------------------------------------------------------------------------
#  Public API
# -------------------------------------------------------------------------------------

def compute_severity(gray_frame: np.ndarray, weather_label: str) -> dict:
    """
    Calculate analytical weather severity for a single frame.

    Severity Mapping by Weather Class:
        Fog / Rain / Snowy / Mist:
            → Derived from Laplacian variance drop vs. clear-day baseline.
              Lower variance = more blur/degradation = higher severity.

        Night:
            → Primary: darkness severity from mean brightness.
            → Secondary: glare severity from white clipping (headlights).
            → Final severity = max(darkness, glare).

        Sunny:
            → Hardcoded 0% severity (clear conditions by definition).

        Unknown / Other:
            → Fallback to Laplacian-based estimate.

    Args:
        gray_frame (np.ndarray):  Single-channel grayscale frame (H, W), dtype uint8.
        weather_label (str):      Predicted weather class string from the classifier.
                                  Expected: "Fog", "Mist", "Night", "Rain", "Snowy", "Sunny".

    Returns:
        dict with keys:
            'severity_pct'     (float): Final severity percentage [0.0–100.0].
            'laplacian_var'    (float): Raw Laplacian variance value.
            'mean_brightness'  (float): Mean grayscale intensity [0.0–255.0].
            'white_clip_ratio' (float): Fraction of pixels > 240 [0.0–1.0].
            'method'           (str):   Estimation method used ('laplacian', 'darkness+glare', 'clear', 'fallback').
    """
    # Step 1: Extract raw metrics
    metrics = extract_metrics(gray_frame)
    laplacian_var = metrics['laplacian_var']
    mean_brightness = metrics['mean_brightness']
    white_clip_ratio = metrics['white_clip_ratio']

    # Step 2: Route to the appropriate severity formula
    severity_pct = 0.0
    method = "clear"

    if weather_label in ("Fog", "Rain", "Snowy", "Mist"):
        severity_pct = _severity_from_laplacian(laplacian_var)
        method = "laplacian"

    elif weather_label == "Night":
        darkness_severity = _severity_from_darkness(mean_brightness)
        glare_severity = _severity_from_glare(white_clip_ratio)
        severity_pct = max(darkness_severity, glare_severity)
        method = "darkness+glare"

    elif weather_label == "Sunny":
        severity_pct = 0.0
        method = "clear"

    else:
        # Unknown class fallback — use Laplacian-based estimate
        severity_pct = _severity_from_laplacian(laplacian_var)
        method = "fallback"

    return {
        'severity_pct': severity_pct,
        'laplacian_var': laplacian_var,
        'mean_brightness': mean_brightness,
        'white_clip_ratio': white_clip_ratio,
        'method': method
    }
