import numpy as np
import cv2
from .utils import normalize_diff_minmax

def brightness_propagation_with_cumulative_diff_to_original(
    img_gray_u8: np.ndarray,
    roi_mask_bool=None,
    threshold: int = 0,
    max_limit: int = 900
):
    """
    Brightness propagation (max-dilate update) + cumulative |original - current_t|.
    Returns: (final_img_u8, diff_sum_float32, used_iters)
    """
    original = img_gray_u8.copy().astype(np.uint8)
    current  = img_gray_u8.copy().astype(np.uint8)

    if roi_mask_bool is None:
        roi_mask_bool = np.ones_like(current, dtype=bool)
    else:
        roi_mask_bool = roi_mask_bool.astype(bool)

    kernel = np.ones((3, 3), np.uint8)

    diff_sum = np.zeros_like(current, dtype=np.float32)
    used_iters = 0

    for step in range(1, max_limit + 1):
        max_neighbors = cv2.dilate(current, kernel, iterations=1)
        diff = max_neighbors.astype(np.int16) - current.astype(np.int16)
        update_mask = (diff > threshold) & roi_mask_bool

        if not np.any(update_mask):
            used_iters = step - 1
            break

        next_img = current.copy()
        next_img[update_mask] = max_neighbors[update_mask]

        step_diff = cv2.absdiff(original, next_img).astype(np.float32)
        step_diff *= roi_mask_bool.astype(np.float32)
        diff_sum += step_diff

        current = next_img
        used_iters = step

    return current, diff_sum, used_iters

def build_diff_norm_map(
    original_gray_u8: np.ndarray,
    roi_mask_bool=None,
    gap_threshold: int = 0,
    max_safety_iter: int = 900,
    dynamic_clip_min: float = 30.0,
    dynamic_clip_mul: float = 2.0,
    fallback_clip: float = 80.0,
):
    """
    Your pipeline:
      propagation -> diff_sum -> diff_mean -> dynamic clip -> uint8 -> minmax normalize.
    Returns:
      diff_norm_u8, used_iters, dynamic_clip_limit
    """
    _, diff_sum, used_iters = brightness_propagation_with_cumulative_diff_to_original(
        original_gray_u8, roi_mask_bool,
        threshold=gap_threshold,
        max_limit=max_safety_iter
    )

    if used_iters <= 0:
        diff_mean = np.zeros_like(original_gray_u8, dtype=np.float32)
    else:
        diff_mean = diff_sum / float(used_iters)

    diff_mean = np.clip(diff_mean, 0, 255).astype(np.float32)

    valid_pixels = diff_mean[diff_mean > 0]
    if valid_pixels.size > 0:
        mean_val = float(np.mean(valid_pixels))
        dynamic_clip_limit = max(mean_val * dynamic_clip_mul, dynamic_clip_min)
    else:
        dynamic_clip_limit = float(fallback_clip)

    diff_clipped = np.clip(diff_mean, 0, dynamic_clip_limit)
    diff_clipped_u8 = np.clip(diff_clipped, 0, 255).astype(np.uint8)

    diff_norm_u8 = normalize_diff_minmax(diff_clipped_u8, roi_mask_bool=roi_mask_bool)
    return diff_norm_u8, used_iters, dynamic_clip_limit


