"""
severity/scripts/eval_rule_based_predicted.py

E0 (rule-based scorer) on PREDICTED instances.

Pipeline:
  1. Load val split GT annotations
  2. Load YOLO predicted instances (result/labels_val/)
  3. Greedy IoU matching (polygon IoU, class-aware)
  4. For each matched pair: compute rule-based features from PREDICTED polygon
  5. Apply fixed-weight scoring -> severity grade
  6. Compare against GT severity label

Features (same as eval_rule_based.py):
  - defect mask area (Shoelace from predicted polygon)
  - bbox width / height
  - background gray contrast

Scoring (same thresholds as classify_grades.py):
  weighted_score <= 3.0  -> Acceptable
  weighted_score <= 4.5  -> Marginal NG
  weighted_score <= 7.5  -> NG
  else                   -> Gross NG

Usage:
  python severity/scripts/eval_rule_based_predicted.py \\
      --split_file severity/splits/split.json \\
      --prediction_dir result/labels_val \\
      --output_dir severity/results_predicted/formal_E0_rule \\
      --eval_mode formal
"""

import argparse
import json
import os
import sys
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from severity.datasets.severity_roi_dataset import collect_instances
from severity.datasets.predicted_instance_dataset import load_yolo_seg_predictions
from severity.utils.instance_matching import greedy_match, compute_match_stats
from severity.utils.metrics import compute_metrics, print_metrics
from severity.utils.grade_schema import GRADE_MAP, GRADE_NAMES

DEFAULT_IMG_DIR   = "Track2/NG_1154/images"
DEFAULT_LABEL_DIR = "Track2/NG_1154/level_labels"

# ── rule-based scoring config (mirrors classify_grades.py) ────────────────
BG_EXPAND_RATIO = 0.1
FACTORS_CONFIG = [
    {"key": "area",    "scores": [3,   5,   7,   9  ]},
    {"key": "rect_w",  "scores": [2.5, 4.5, 6.5, 8.5]},
    {"key": "rect_h",  "scores": [3,   5,   7,   9  ]},
    {"key": "bg_gray", "scores": [2,   4,   6,   8  ]},
]
WEIGHTS = [0.15, 0.25, 0.25, 0.35]


def score_to_grade(ws: float) -> str:
    if ws <= 3.0:
        return "Acceptable"
    elif ws <= 4.5:
        return "Marginal NG"
    elif ws <= 7.5:
        return "NG"
    else:
        return "Gross NG"


def _quantile_score(value: float, all_vals: list, scores: list) -> float:
    n = len(all_vals)
    if n < 4:
        return scores[0]
    bins = [
        all_vals[0],
        all_vals[n // 4],
        all_vals[2 * n // 4],
        all_vals[3 * n // 4],
        all_vals[-1] + 1e-9,
    ]
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return scores[i]
    return scores[-1]


def _poly_mask_scanline(pts_px, W, H):
    """Rasterise polygon to binary mask (no cv2)."""
    mask = np.zeros((H, W), dtype=np.uint8)
    n = len(pts_px)
    if n < 3:
        return mask
    min_y = max(0, int(min(p[1] for p in pts_px)))
    max_y = min(H - 1, int(max(p[1] for p in pts_px)))
    for y in range(min_y, max_y + 1):
        xs_cross = []
        for i in range(n):
            x0, y0 = pts_px[i]
            x1, y1 = pts_px[(i + 1) % n]
            if (y0 <= y < y1) or (y1 <= y < y0):
                if y1 != y0:
                    xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                    xs_cross.append(xc)
        xs_cross.sort()
        for k in range(0, len(xs_cross) - 1, 2):
            xa = max(0, int(xs_cross[k]))
            xb = min(W - 1, int(xs_cross[k + 1]))
            mask[y, xa:xb + 1] = 1
    return mask


def extract_rule_features(points_norm, W, H, img_gray):
    """
    Extract rule-based features from a predicted polygon.
    Returns dict with keys: area, rect_w, rect_h, bg_gray
    """
    xs = [points_norm[i] * W for i in range(0, len(points_norm), 2)]
    ys = [points_norm[i] * H for i in range(1, len(points_norm), 2)]
    x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
    x2, y2 = min(W - 1, int(max(xs))), min(H - 1, int(max(ys)))
    rect_w = max(x2 - x1, 1)
    rect_h = max(y2 - y1, 1)

    pts_px = [(int(round(x)), int(round(y))) for x, y in zip(xs, ys)]
    mask = _poly_mask_scanline(pts_px, W, H)

    # area via Shoelace
    if len(pts_px) >= 3:
        arr = np.array(pts_px, dtype=np.float64)
        xa, ya = arr[:, 0], arr[:, 1]
        area = 0.5 * abs(np.sum(xa * np.roll(ya, -1) - np.roll(xa, -1) * ya))
    else:
        area = float(rect_w * rect_h)

    # background gray
    ex = max(1, int(rect_w * BG_EXPAND_RATIO))
    ey = max(1, int(rect_h * BG_EXPAND_RATIO))
    bx1, by1 = max(0, x1 - ex), max(0, y1 - ey)
    bx2, by2 = min(W, x2 + ex), min(H, y2 + ey)
    bg_mask = np.zeros((H, W), dtype=np.uint8)
    bg_mask[by1:by2, bx1:bx2] = 1
    bg_mask[mask == 1] = 0
    bg_pixels = img_gray[bg_mask == 1]
    bg_gray = float(bg_pixels.mean()) if len(bg_pixels) >= 10 else 128.0

    return {"area": area, "rect_w": float(rect_w),
            "rect_h": float(rect_h), "bg_gray": bg_gray}


def rule_grade_batch(matched_pairs, img_dir):
    """
    Apply rule-based scoring to a list of matched pairs.
    Features are computed from PREDICTED polygon (not GT).
    Returns list of predicted grade ids.
    """
    # collect features
    features = []
    for pair in matched_pairs:
        img_path = pair["img_path"]
        pts = pair["pred_points"]
        try:
            img = PILImage.open(img_path).convert("L")
            W, H = img.size
            gray = np.array(img, dtype=np.uint8)
            feat = extract_rule_features(pts, W, H, gray)
        except Exception as e:
            print(f"  [warn] feature extraction failed for {img_path}: {e}")
            feat = {"area": 0.0, "rect_w": 0.0, "rect_h": 0.0, "bg_gray": 128.0}
        features.append(feat)

    # quantile scoring over the batch
    pred_grades = []
    for feat in features:
        ws = 0.0
        for cfg, w in zip(FACTORS_CONFIG, WEIGHTS):
            key = cfg["key"]
            all_vals = sorted(f[key] for f in features)
            s = _quantile_score(feat[key], all_vals, cfg["scores"])
            ws += s * w
        pred_grades.append(GRADE_MAP[score_to_grade(ws)])
    return pred_grades


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split_file",        default="severity/splits/split.json")
    p.add_argument("--img_dir",           default=DEFAULT_IMG_DIR)
    p.add_argument("--label_dir",         default=DEFAULT_LABEL_DIR)
    p.add_argument("--prediction_dir",    default="result/labels_val")
    p.add_argument("--output_dir",        default="severity/results_predicted/formal_E0_rule")
    p.add_argument("--iou_thresh",        type=float, default=0.5)
    p.add_argument("--class_aware_match", default="true")
    p.add_argument("--eval_mode",         default="formal",
                   choices=["smoke_test", "formal"])
    return p.parse_args()


def main():
    args = parse_args()
    class_aware = args.class_aware_match.lower() in ("true", "1", "yes")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.eval_mode == "smoke_test":
        print("\n[SMOKE TEST] results are NON-FORMAL\n")

    # ── load split ────────────────────────────────────────────────────────
    with open(args.split_file) as f:
        sp = json.load(f)
    val_stems = sp["val_stems"]
    print(f"val split: {len(val_stems)} images")

    # ── read image sizes ──────────────────────────────────────────────────
    img_size_by_stem = {}
    for stem in val_stems:
        p = os.path.join(args.img_dir, stem + ".bmp")
        if os.path.exists(p):
            with PILImage.open(p) as im:
                img_size_by_stem[stem] = im.size
        else:
            img_size_by_stem[stem] = (1024, 1024)

    # ── load GT ───────────────────────────────────────────────────────────
    print("loading GT annotations...")
    all_inst = collect_instances(args.img_dir, args.label_dir, compute_morph=False)
    val_stem_set = set(val_stems)
    gt_by_stem = defaultdict(list)
    for inst in all_inst:
        if inst["img_stem"] in val_stem_set:
            gt_by_stem[inst["img_stem"]].append({
                "cls_id":      inst["cls_id"],
                "grade_id":    inst["grade_id"],
                "points_norm": inst["points_norm"],
                "img_path":    inst["img_path"],
            })
    print(f"  GT instances in val: {sum(len(v) for v in gt_by_stem.values())}")

    # ── load predictions ──────────────────────────────────────────────────
    print("loading predictions...")
    pred_by_stem = load_yolo_seg_predictions(args.prediction_dir, val_stems)
    print(f"  pred instances in val: {sum(len(v) for v in pred_by_stem.values())}")

    # ── matching ──────────────────────────────────────────────────────────
    all_matched_pairs = []
    per_image_stats   = []
    iou_mode_counts   = defaultdict(int)

    for stem in val_stems:
        gt_insts   = gt_by_stem.get(stem, [])
        pred_insts = pred_by_stem.get(stem, [])
        img_W, img_H = img_size_by_stem.get(stem, (1024, 1024))

        pairs, _, _, mode_counts = greedy_match(
            gt_insts, pred_insts, args.iou_thresh, class_aware, img_W, img_H)
        for k, v in mode_counts.items():
            iou_mode_counts[k] += v

        for pair in pairs:
            gi, pi = pair["gt_idx"], pair["pred_idx"]
            g, p = gt_insts[gi], pred_insts[pi]
            all_matched_pairs.append({
                "img_stem":    stem,
                "gt_idx":      gi,
                "pred_idx":    pi,
                "gt_cls":      g["cls_id"],
                "pred_cls":    p["cls_id"],
                "gt_severity": GRADE_NAMES[g["grade_id"]],
                "gt_grade_id": g["grade_id"],
                "iou":         pair["iou"],
                "iou_mode":    pair["iou_mode"],
                "img_path":    g["img_path"],
                "pred_points": p["points_norm"],
            })

        stats = compute_match_stats(gt_insts, pred_insts, pairs)
        stats["img_stem"] = stem
        per_image_stats.append(stats)

    num_gt      = sum(s["num_gt"]      for s in per_image_stats)
    num_pred    = sum(s["num_pred"]    for s in per_image_stats)
    num_matched = len(all_matched_pairs)
    print(f"  GT={num_gt}  pred={num_pred}  matched={num_matched}")
    print(f"  IoU mode: {dict(iou_mode_counts)}")

    if num_matched == 0:
        print("[warn] no matched pairs")
        return

    # ── rule-based scoring on predicted polygons ──────────────────────────
    print(f"running rule-based scoring on {num_matched} matched instances...")
    pred_grades = rule_grade_batch(all_matched_pairs, args.img_dir)
    gt_grades   = [p["gt_grade_id"] for p in all_matched_pairs]

    # ── metrics ───────────────────────────────────────────────────────────
    metrics  = compute_metrics(gt_grades, pred_grades)
    macro_f1 = float(f1_score(gt_grades, pred_grades, average="macro", zero_division=0))

    print("\n" + "=" * 60)
    print("  E0 rule-based (predicted instances)")
    print("=" * 60)
    print_metrics(metrics, prefix="E0-Predicted")
    print(f"  Macro-F1 : {macro_f1:.4f}")

    # ── save ──────────────────────────────────────────────────────────────
    SMOKE_WARNING = "NON-FORMAL" if args.eval_mode == "smoke_test" else None

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({
            "exp": "E0_rule_based_predicted",
            "eval_mode": args.eval_mode,
            "smoke_test_warning": SMOKE_WARNING,
            "prediction_dir": args.prediction_dir,
            "iou_thresh": args.iou_thresh,
            "class_aware_match": class_aware,
            "iou_mode_counts": dict(iou_mode_counts),
            "roi_source": "pred_polygon",
            "morph_source": "pred_polygon",
            "timestamp": datetime.datetime.now().isoformat(),
        }, f, indent=2)

    with open(os.path.join(args.output_dir, "severity_metrics_matched.json"), "w") as f:
        json.dump({
            "eval_mode": args.eval_mode,
            "smoke_test_warning": SMOKE_WARNING,
            "accuracy":  round(metrics["overall_acc"], 4),
            "macro_f1":  round(macro_f1, 4),
            "qwk":       round(metrics["weighted_kappa"], 4),
            "num_matched_eval": num_matched,
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        }, f, indent=2)

    pd.DataFrame(metrics["confusion_matrix"],
                 index=GRADE_NAMES, columns=GRADE_NAMES).to_csv(
        os.path.join(args.output_dir, "confusion_matrix_matched.csv"))

    rows = []
    for pair, pred, gt in zip(all_matched_pairs, pred_grades, gt_grades):
        rows.append({
            "img_stem":     pair["img_stem"],
            "gt_class":     pair["gt_cls"],
            "pred_class":   pair["pred_cls"],
            "gt_severity":  pair["gt_severity"],
            "pred_severity":GRADE_NAMES[pred],
            "iou":          pair["iou"],
            "iou_mode":     pair["iou_mode"],
            "correct":      int(pred == gt),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(args.output_dir, "matched_predictions.csv"), index=False)

    pd.DataFrame(per_image_stats).to_csv(
        os.path.join(args.output_dir, "per_image_match_stats.csv"), index=False)

    print(f"\n  results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
