import numpy as np
import cv2

def imread_korean(path):
    """Read image path that may include Korean characters (Windows-friendly)."""
    try:
        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise FileNotFoundError(f"Error loading {path}: {e}")

def penalty_to_lambda(penalty: float) -> float:
    p = float(penalty)
    if p >= 1.0: return 1e6
    if p <= 0.0: return 0.0
    return p / (1.0 - p)

def normalize_diff_minmax(diff_u8: np.ndarray, roi_mask_bool=None) -> np.ndarray:
    """Min-max normalize uint8 map to [0,255] globally or within ROI."""
    diff = diff_u8.astype(np.float32)

    if roi_mask_bool is None:
        vals = diff.ravel()
        if vals.size == 0:
            return np.zeros_like(diff_u8, dtype=np.uint8)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax > vmin:
            norm = (diff - vmin) / (vmax - vmin) * 255.0
        else:
            norm = np.zeros_like(diff)
        return np.clip(norm, 0, 255).astype(np.uint8)

    roi = roi_mask_bool.astype(bool)
    if not np.any(roi):
        return np.zeros_like(diff_u8, dtype=np.uint8)

    vals = diff[roi]
    vmin, vmax = float(vals.min()), float(vals.max())
    norm = np.zeros_like(diff, dtype=np.float32)
    if vmax > vmin:
        norm_roi = (diff[roi] - vmin) / (vmax - vmin) * 255.0
        norm[roi] = norm_roi
    else:
        norm[roi] = 0.0
    return np.clip(norm, 0, 255).astype(np.uint8)

def sigmoid_weight_from_diff(diff_u8: np.ndarray, k=0.10, center=30.0, roi_mask_bool=None) -> np.ndarray:
    """Sigmoid mapping to [0,1] weights."""
    x = diff_u8.astype(np.float32)
    w = 1.0 / (1.0 + np.exp(-float(k) * (x - float(center))))
    if roi_mask_bool is not None:
        w = w * roi_mask_bool.astype(np.float32)
    return w.astype(np.float32)

def binarize_mask_u8(mask_u8: np.ndarray, thr: int = 127) -> np.ndarray:
    """Return boolean mask from uint8 mask."""
    return (mask_u8 > thr)


