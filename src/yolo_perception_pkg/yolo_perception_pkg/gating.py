"""
TH OWL — Illumination & Image Quality Gating Module
====================================================
Provides metric-based degradation detection to complement the neural
network weather classifier. Used by both perception nodes as Stage B
of the dual-layer gating system.

Stage A: Classifier (EfficientNetV2-M) flags adverse weather labels.
Stage B: check_gating() validates frame metrics independently.
Routing: restore if Stage A OR Stage B raises an alarm (conservative union).

Usage:
    from yolo_perception_pkg.gating import check_gating
    route, reason = check_gating(mean_brightness, laplacian_var, white_clip_ratio)
"""


def check_gating(
    mean_brightness: float,
    laplacian_var: float,
    white_clipping: float,
) -> tuple:
    """
    Evaluate image quality metrics and decide if the frame requires restoration.

    Args:
        mean_brightness (float): Average grayscale pixel intensity [0–255].
        laplacian_var   (float): Variance of the Laplacian operator output.
                                 Higher = sharper image, lower = blur/fog.
        white_clipping  (float): Fraction of pixels with intensity > 240 [0.0–1.0].
                                 Higher = overexposure / glare.

    Returns:
        route_to_restorer (bool): True if frame should be restored, False to bypass.
        route_reason      (str):  Reason code for telemetry logging:
                                  'low_light' | 'glare' | 'blur' | 'bypass'
    """
    # Threshold configurations — tuned for CARLA camera characteristics
    LOW_LIGHT_THRESHOLD  = 50.0   # Mean brightness < 50  → night / low-light
    HIGH_GLARE_THRESHOLD = 0.05   # > 5% pixels clipped above 240 → significant glare
    BLUR_THRESHOLD       = 100.0  # Laplacian variance < 100 → heavy blur / fog

    # 1. Low illumination / Night condition
    if mean_brightness < LOW_LIGHT_THRESHOLD:
        return True, 'low_light'

    # 2. High glare / Illumination saturation clipping
    if white_clipping > HIGH_GLARE_THRESHOLD:
        return True, 'glare'

    # 3. Severe blur or poor edge definition (fog / rain / heavy mist)
    if laplacian_var < BLUR_THRESHOLD:
        return True, 'blur'

    # Default: raw frame is acceptable — forward directly to detector
    return False, 'bypass'
