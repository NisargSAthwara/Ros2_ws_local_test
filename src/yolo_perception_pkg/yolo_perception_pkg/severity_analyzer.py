"""
TH OWL — Camera Weather Quality Index (CWQI) Module
=====================================================
Computes a single-frame, no-reference image quality degradation score
(CWQI) using four deterministic OpenCV/NumPy metrics. Suitable for
real-time ROS2 execution without a reference frame or ML model.

EXCLUDED intentionally:
    SSIM, PSNR — require a reference (clean) frame; cannot be applied
    in online/live inference mode within the 33.3 ms per-frame budget.

Metrics used:
    Contrast     — Pixel intensity standard deviation
    Sharpness    — Laplacian variance (edge definition)
    Edge Density — Canny edge pixel fraction
    Entropy      — Shannon entropy of the grayscale histogram

CWQI = 1 − mean(quality_scores)   → [0.0 = perfect, 1.0 = fully degraded]

Severity Levels (CWQI → Level):
    < 0.15 → Level 1 (Excellent)
    0.15–0.35 → Level 2 (Good)
    0.35–0.55 → Level 3 (Moderate)
    0.55–0.75 → Level 4 (Severe)
    ≥ 0.75 → Level 5 (Critical)

Usage:
    from yolo_perception_pkg.severity_analyzer import compute_cwqi
    result = compute_cwqi(gray_frame)
    print(result['cwqi'], result['severity_level'], result['severity_label'])
"""

import cv2
import numpy as np
from typing import Dict

# -------------------------------------------------------------------------------------
#  Normalization Baselines
# -------------------------------------------------------------------------------------

# std(gray) at this level counts as full contrast (score = 1.0)
_CONTRAST_BASELINE: float = 64.0

# Laplacian variance at this level = clear sharp image (score = 1.0)
_SHARPNESS_BASELINE: float = 500.0

# Canny edge pixel fraction at this level = good texture (score = 1.0)
_EDGE_DENSITY_BASELINE: float = 0.10

# Max achievable Shannon entropy (log2 of 256 bins)
_MAX_ENTROPY: float = float(np.log2(256))

# CWQI severity breakpoints (ascending degradation thresholds)
_CWQI_THRESHOLDS = (0.15, 0.35, 0.55, 0.75)
_SEVERITY_LABELS = ('Excellent', 'Good', 'Moderate', 'Severe', 'Critical')


# -------------------------------------------------------------------------------------
#  Internal Helpers
# -------------------------------------------------------------------------------------

def _contrast_score(gray: np.ndarray) -> float:
    """Normalized pixel intensity standard deviation [0–1]. Higher = more contrast."""
    return min(float(np.std(gray)) / _CONTRAST_BASELINE, 1.0)


def _sharpness_score(gray: np.ndarray) -> float:
    """Normalized Laplacian variance [0–1]. Higher = sharper image."""
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return min(lap_var / _SHARPNESS_BASELINE, 1.0), lap_var


def _edge_density_score(gray: np.ndarray) -> float:
    """Normalized Canny edge pixel fraction [0–1]. Higher = more texture/edges."""
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    ratio = float(np.count_nonzero(edges)) / float(gray.size)
    return min(ratio / _EDGE_DENSITY_BASELINE, 1.0)


def _entropy_score(gray: np.ndarray) -> float:
    """Normalized Shannon entropy from grayscale histogram [0–1]. Higher = more info."""
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    total = float(gray.size)
    hist_norm = hist[hist > 0] / total
    raw_entropy = float(-np.sum(hist_norm * np.log2(hist_norm)))
    return min(raw_entropy / _MAX_ENTROPY, 1.0)


def _cwqi_to_severity(cwqi: float) -> tuple:
    """Map a CWQI value to an integer severity level and label string."""
    for level, threshold in enumerate(_CWQI_THRESHOLDS, start=1):
        if cwqi < threshold:
            return level, _SEVERITY_LABELS[level - 1]
    return 5, _SEVERITY_LABELS[4]


# -------------------------------------------------------------------------------------
#  Public API
# -------------------------------------------------------------------------------------

def compute_cwqi(gray_frame: np.ndarray) -> Dict:
    """
    Compute the Camera Weather Quality Index for a single grayscale frame.

    Args:
        gray_frame: Single-channel grayscale image (H, W), dtype uint8.

    Returns:
        dict containing:
            'cwqi'           (float): Degradation index [0.0–1.0].
            'severity_level' (int):   Severity level [1–5].
            'severity_label' (str):   Human-readable label string.
            'contrast'       (float): Normalized contrast quality score [0–1].
            'sharpness'      (float): Normalized sharpness quality score [0–1].
            'edge_density'   (float): Normalized edge density quality score [0–1].
            'entropy'        (float): Normalized entropy quality score [0–1].
            'laplacian_var'  (float): Raw Laplacian variance (for telemetry logging).
    """
    contrast      = _contrast_score(gray_frame)
    sharpness, lap_var = _sharpness_score(gray_frame)
    edge_density  = _edge_density_score(gray_frame)
    entropy       = _entropy_score(gray_frame)

    # CWQI = degradation = 1 − mean quality score
    quality_score = (contrast + sharpness + edge_density + entropy) / 4.0
    cwqi          = max(0.0, min(1.0, 1.0 - quality_score))

    severity_level, severity_label = _cwqi_to_severity(cwqi)

    return {
        'cwqi':           cwqi,
        'severity_level': severity_level,
        'severity_label': severity_label,
        'contrast':       contrast,
        'sharpness':      sharpness,
        'edge_density':   edge_density,
        'entropy':        entropy,
        'laplacian_var':  lap_var,
    }
