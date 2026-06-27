def check_gating(mean_brightness: float, laplacian_var: float, white_clipping: float) -> (bool, str):
    """
    Evaluates calculated image metrics and decides if the frame is degraded.

    Args:
        mean_brightness (float): The average grayscale intensity (0-255).
        laplacian_var (float): Variance of the Laplacian, indicating texture/sharpness (blur).
        white_clipping (float): Ratio of pixels with intensity > 240 (0.0 - 1.0).

    Returns:
        route_to_restorer (bool): True if the frame should be restored, False to bypass.
        route_reason (str): Reason code for logging output (e.g., 'low_light', 'glare', 'blur', 'bypass').
    """
    # Threshold configurations for gating activation
    LOW_LIGHT_THRESHOLD = 50.0    # Mean brightness below 50 indicates night/low-light
    HIGH_GLARE_THRESHOLD = 0.05   # More than 5% of pixels clipped above 240 indicates significant glare
    BLUR_THRESHOLD = 100.0        # Laplacian variance below 100 indicates heavy blur/fog

    # 1. Low illumination / Night condition
    if mean_brightness < LOW_LIGHT_THRESHOLD:
        return True, "low_light"

    # 2. High glare / Illumination saturation clipping
    if white_clipping > HIGH_GLARE_THRESHOLD:
        return True, "glare"

    # 3. Severe blur or poor edge definition (e.g., heavy fog/rain)
    if laplacian_var < BLUR_THRESHOLD:
        return True, "blur"

    # Default: Pass the raw frame forward (Bypass)
    return False, "bypass"
