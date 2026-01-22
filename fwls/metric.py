import numpy as np
from .utils import penalty_to_lambda, sigmoid_weight_from_diff, binarize_mask_u8
from .weights import build_diff_norm_map

def weighted_score(G_bool, P_bool, w):
    TP = G_bool & P_bool
    FN = G_bool & (~P_bool)
    FP = (~G_bool) & P_bool

    TP_w = float(w[TP].sum())
    FN_w = float(w[FN].sum())
    FP_w = float(w[FP].sum())

    denom = TP_w + FN_w + FP_w
    if denom == 0:
        return 1.0 if np.count_nonzero(G_bool) == 0 else 0.0
    return TP_w / denom


def fwls_score(
    pred_mask_u8, gt_mask_u8, original_gray_u8,
    *,
    roi_mode="all",
    gap_threshold=0,
    max_safety_iter=900,
    sigmoid_k=0.10,
    c_min=0, c_max=255,
    gamma=0.4,
):
    G = binarize_mask_u8(gt_mask_u8, 127)
    P = binarize_mask_u8(pred_mask_u8, 127)

    if roi_mode == "union":
        roi = (G | P)
    elif roi_mode == "gt_only":
        roi = G.copy()
    else:
        roi = None

    diff_norm_u8, used_iters, clip_limit = build_diff_norm_map(
        original_gray_u8,
        roi_mask_bool=roi,
        gap_threshold=gap_threshold,
        max_safety_iter=max_safety_iter,
    )

    scores = []
    for c in np.arange(c_min, c_max + 1, dtype=np.float32):
        w = sigmoid_weight_from_diff(diff_norm_u8, k=sigmoid_k, center=c, roi_mask_bool=roi)
        w = np.power(w, gamma)
        scores.append(weighted_score(G, P, w))

    return float(np.mean(scores))





