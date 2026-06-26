"""
severity/utils/instance_matching.py

Greedy one-to-one matching between GT and predicted instances.

IoU mode (auto-selected per instance pair):
  - "polygon": rasterise both polygons to binary masks, compute mask IoU
               used when both instances have >= 3 polygon vertices
  - "bbox":    fall back to bbox IoU when polygon is degenerate (< 3 pts)

The mode actually used is recorded per matched pair and in the summary.
"""

from typing import List, Tuple
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# polygon rasterisation (no cv2 dependency)
# ---------------------------------------------------------------------------

def _poly_mask_fast(pts_xy: List[Tuple[float, float]],
                    W: int, H: int) -> np.ndarray:
    """
    Rasterise a polygon to a boolean (H, W) mask using scanline fill.
    pts_xy: list of (x_pixel, y_pixel) tuples (already in pixel coords).
    """
    mask = np.zeros((H, W), dtype=np.bool_)
    n = len(pts_xy)
    if n < 3:
        return mask
    min_y = max(0, int(min(p[1] for p in pts_xy)))
    max_y = min(H - 1, int(max(p[1] for p in pts_xy)))
    for y in range(min_y, max_y + 1):
        xs_cross = []
        for i in range(n):
            x0, y0 = pts_xy[i]
            x1, y1 = pts_xy[(i + 1) % n]
            if (y0 <= y < y1) or (y1 <= y < y0):
                if y1 != y0:
                    xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                    xs_cross.append(xc)
        xs_cross.sort()
        for k in range(0, len(xs_cross) - 1, 2):
            xa = max(0, int(xs_cross[k]))
            xb = min(W - 1, int(xs_cross[k + 1]))
            mask[y, xa:xb + 1] = True
    return mask


def _norm_to_pixel(points_norm: List[float],
                   W: int, H: int) -> List[Tuple[float, float]]:
    """Convert flat normalised [x0,y0,x1,y1,...] to pixel (x,y) tuples."""
    return [(points_norm[i] * W, points_norm[i + 1] * H)
            for i in range(0, len(points_norm) - 1, 2)]


# ---------------------------------------------------------------------------
# IoU implementations
# ---------------------------------------------------------------------------

def polygon_iou(pts_a: List[float], pts_b: List[float],
                W: int, H: int) -> Tuple[float, str]:
    """
    Compute IoU between two normalised polygons.
    Returns (iou_value, iou_mode) where iou_mode is 'polygon' or 'bbox'.

    Falls back to bbox IoU if either polygon has < 3 vertices.
    """
    n_a = len(pts_a) // 2
    n_b = len(pts_b) // 2

    if n_a >= 3 and n_b >= 3:
        # polygon mask IoU
        pix_a = _norm_to_pixel(pts_a, W, H)
        pix_b = _norm_to_pixel(pts_b, W, H)
        mask_a = _poly_mask_fast(pix_a, W, H)
        mask_b = _poly_mask_fast(pix_b, W, H)
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        iou = float(inter) / (float(union) + 1e-9) if union > 0 else 0.0
        return iou, "polygon"
    else:
        # bbox fallback
        iou = _bbox_iou_from_norm(pts_a, pts_b)
        return iou, "bbox"


def _bbox_from_norm(pts: List[float]) -> Tuple[float, float, float, float]:
    xs = pts[0::2]
    ys = pts[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou_from_norm(pts_a: List[float], pts_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = _bbox_from_norm(pts_a)
    bx1, by1, bx2, by2 = _bbox_from_norm(pts_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


# ---------------------------------------------------------------------------
# Greedy matching
# ---------------------------------------------------------------------------

def greedy_match(
    gt_instances: List[dict],
    pred_instances: List[dict],
    iou_thresh: float = 0.5,
    class_aware: bool = True,
    img_W: int = 1024,
    img_H: int = 1024,
) -> Tuple[List[dict], List[int], List[int], dict]:
    """
    Greedy one-to-one matching.

    Each instance dict must have:
      'points_norm': flat list of normalised polygon coords
      'cls_id':      integer class id

    img_W / img_H are used for polygon rasterisation.
    If unknown, pass a large value (e.g. 1024) — IoU is scale-invariant
    for polygon mode as long as both use the same W/H.

    Returns:
      matched_pairs:     list of dicts (gt_idx, pred_idx, iou, iou_mode, ...)
      unmatched_gt_idx:  list of unmatched GT indices
      unmatched_pred_idx:list of unmatched pred indices
      iou_mode_counts:   {'polygon': N, 'bbox': M}  for logging
    """
    if not gt_instances or not pred_instances:
        return [], list(range(len(gt_instances))), list(range(len(pred_instances))), {}

    # build candidate list
    candidates = []
    for gi, g in enumerate(gt_instances):
        for pi, p in enumerate(pred_instances):
            if class_aware and g["cls_id"] != p["cls_id"]:
                continue
            iou, mode = polygon_iou(
                g["points_norm"], p["points_norm"], img_W, img_H)
            if iou >= iou_thresh:
                candidates.append((iou, gi, pi, mode))

    candidates.sort(key=lambda x: -x[0])

    matched_gt   = set()
    matched_pred = set()
    matched_pairs = []
    iou_mode_counts = defaultdict(int)

    for iou, gi, pi, mode in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        iou_mode_counts[mode] += 1
        matched_pairs.append({
            "gt_idx":   gi,
            "pred_idx": pi,
            "iou":      round(iou, 4),
            "iou_mode": mode,
            "gt_cls":   gt_instances[gi]["cls_id"],
            "pred_cls": pred_instances[pi]["cls_id"],
        })

    unmatched_gt   = [i for i in range(len(gt_instances))   if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_instances)) if i not in matched_pred]

    return matched_pairs, unmatched_gt, unmatched_pred, dict(iou_mode_counts)


# ---------------------------------------------------------------------------
# Per-image match statistics
# ---------------------------------------------------------------------------

def compute_match_stats(gt_instances, pred_instances, matched_pairs) -> dict:
    num_gt      = len(gt_instances)
    num_pred    = len(pred_instances)
    num_matched = len(matched_pairs)
    return {
        "num_gt":              num_gt,
        "num_pred":            num_pred,
        "num_matched":         num_matched,
        "matched_rate":        round(num_matched / num_gt,   4) if num_gt   > 0 else 0.0,
        "pred_precision_like": round(num_matched / num_pred, 4) if num_pred > 0 else 0.0,
    }
