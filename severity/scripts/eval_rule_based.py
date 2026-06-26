"""
severity/scripts/eval_rule_based.py

第二组实验：用 classify_grades.py 中的规则打分（面积/外接矩形/灰度）
在 val split 上计算 Sgrade（QWK），与第一组 ResNet18 结果对比。

直接复用 baseline/classify_grades.py 中的特征计算和分级逻辑，不做任何修改。
用法:
    python severity/scripts/eval_rule_based.py
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# 复用 baseline 的特征计算类
sys.path.insert(0, "baseline")
from classify_grades import (
    YOLODefectAnalyzer,
    process_excel_to_graded,
    DefectResultExcelGenerator,
)

from severity.utils.grade_schema import GRADE_MAP, GRADE_NAMES, NUM_GRADES
from severity.utils.metrics import compute_metrics, print_metrics

IMG_DIR      = "Track2/NG_1154/images"
LABEL_DIR    = "Track2/NG_1154/level_labels"   # GT json（含 severity）
SPLIT_PATH   = "severity/splits/split.json"
SAVE_DIR     = "severity/results/rule_based"

import argparse as _ap
def _parse():
    p = _ap.ArgumentParser()
    p.add_argument("--split_file", default=SPLIT_PATH)
    p.add_argument("--save_dir",   default=SAVE_DIR)
    return p.parse_args()
_args = _parse()

# 与 classify_grades.py 保持一致
BG_EXPAND_RATIO = 0.1
FACTORS_CONFIG = [
    {"column": "损伤区面积（像素²）",   "scores": [3, 5, 7, 9]},
    {"column": "外接矩形长度（像素）",  "scores": [2.5, 4.5, 6.5, 8.5]},
    {"column": "外接矩形高度（像素）",  "scores": [3, 5, 7, 9]},
    {"column": "背景平均灰度值",        "scores": [2, 4, 6, 8]},
]
WEIGHTS = [0.15, 0.25, 0.25, 0.35]

# 规则打分 → 最终等级映射（与 classify_grades.py 一致）
def score_to_grade(weighted_score):
    if weighted_score <= 3:
        return "Acceptable"
    elif weighted_score <= 4.5:
        return "Marginal NG"
    elif weighted_score <= 7.5:
        return "NG"
    else:
        return "Gross NG"


def grade_value_quantile(value, sorted_values, scores):
    """按四分位分箱打分，与 classify_grades.process_excel_to_graded 逻辑一致"""
    n = len(sorted_values)
    if n < 4:
        return scores[0]
    bins = [
        sorted_values[0],
        sorted_values[n // 4],
        sorted_values[2 * n // 4],
        sorted_values[3 * n // 4],
        sorted_values[-1] + 1e-9,   # 右开
    ]
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return scores[i]
    return scores[-1]


def rule_grade_instances(instances, img_dir):
    """
    对一批实例用规则打分，返回 (gt_grades, pred_grades) 两个列表。
    instances: list of dict，每个含 img_path, bbox(x1,y1,x2,y2), W, H, grade_id
    """
    from PIL import Image as PILImage
    import cv2
    from scipy.spatial import distance_matrix as sp_dist

    # ── 1. 提取每个实例的特征 ──────────────────────────────────────────────
    features = []   # list of dict: area, rect_w, rect_h, bg_gray
    gt_grades = []

    for inst in instances:
        gt_grades.append(inst["grade_id"])
        try:
            img = PILImage.open(inst["img_path"]).convert("L")
            gray = np.array(img, dtype=np.uint8)
            W, H = inst["W"], inst["H"]
            x1, y1, x2, y2 = inst["bbox"]

            # 多边形点（归一化 → 像素）
            pts = inst["points_px"]   # 由调用方填入

            # 面积（Shoelace）
            if len(pts) >= 3:
                pts_arr = np.array(pts, dtype=np.float64)
                x_arr, y_arr = pts_arr[:, 0], pts_arr[:, 1]
                area = 0.5 * abs(np.sum(x_arr * np.roll(y_arr, -1) - np.roll(x_arr, -1) * y_arr))
            else:
                area = float((x2 - x1) * (y2 - y1))

            rect_w = float(x2 - x1)
            rect_h = float(y2 - y1)

            # 背景灰度：扩展 10% 后排除缺陷区域
            expand_w = max(1, int(rect_w * BG_EXPAND_RATIO))
            expand_h = max(1, int(rect_h * BG_EXPAND_RATIO))
            bx1 = max(0, x1 - expand_w)
            by1 = max(0, y1 - expand_h)
            bx2 = min(W, x2 + expand_w)
            by2 = min(H, y2 + expand_h)

            mask = np.zeros((H, W), dtype=np.uint8)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1)
            else:
                mask[y1:y2, x1:x2] = 1

            bg_mask = np.zeros((H, W), dtype=np.uint8)
            bg_mask[by1:by2, bx1:bx2] = 1
            bg_mask[mask == 1] = 0
            bg_pixels = gray[bg_mask == 1]
            bg_gray = float(np.mean(bg_pixels)) if len(bg_pixels) >= 10 else 128.0

            features.append({
                "损伤区面积（像素²）": area,
                "外接矩形长度（像素）": rect_w,
                "外接矩形高度（像素）": rect_h,
                "背景平均灰度值": bg_gray,
            })
        except Exception as e:
            print(f"  警告: 特征提取失败 {inst['img_path']}: {e}")
            features.append({
                "损伤区面积（像素²）": 0.0,
                "外接矩形长度（像素）": 0.0,
                "外接矩形高度（像素）": 0.0,
                "背景平均灰度值": 128.0,
            })

    # ── 2. 按四分位分箱打分（与 classify_grades 一致，用当前批次统计分箱）──
    pred_grades = []
    for i, feat in enumerate(features):
        weighted_score = 0.0
        for cfg, w in zip(FACTORS_CONFIG, WEIGHTS):
            col = cfg["column"]
            scores = cfg["scores"]
            all_vals = sorted([f[col] for f in features])
            s = grade_value_quantile(feat[col], all_vals, scores)
            weighted_score += s * w
        pred_grade_name = score_to_grade(weighted_score)
        pred_grades.append(GRADE_MAP[pred_grade_name])

    return gt_grades, pred_grades


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 加载划分 ──────────────────────────────────────────────────────────
    if not os.path.exists(SPLIT_PATH):
        print(f"找不到 {SPLIT_PATH}，请先运行 make_split.py")
        sys.exit(1)

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    val_stems = set(split["val_stems"])
    print(f"val split: {len(val_stems)} 张图")

    # ── 收集 val 实例（含多边形像素坐标）────────────────────────────────
    from PIL import Image as PILImage
    from severity.datasets.severity_roi_dataset import parse_points, points_to_bbox_pixel

    instances = []
    for fname in sorted(os.listdir(LABEL_DIR)):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]
        if stem not in val_stems:
            continue
        img_path = os.path.join(IMG_DIR, stem + ".bmp")
        if not os.path.exists(img_path):
            continue
        with PILImage.open(img_path) as im:
            W, H = im.size
        anns = json.load(open(os.path.join(LABEL_DIR, fname), encoding="utf-8"))
        for ann in anns:
            sev = ann.get("severity")
            if sev not in GRADE_MAP:
                continue
            points = parse_points(ann["points"])
            if len(points) < 4:
                continue
            x1, y1, x2, y2 = points_to_bbox_pixel(points, W, H)
            # 多边形像素坐标
            pts_px = [(round(points[i]*W), round(points[i+1]*H))
                      for i in range(0, len(points), 2)]
            instances.append({
                "img_path": img_path,
                "img_stem": stem,
                "grade_id": GRADE_MAP[sev],
                "bbox": (x1, y1, x2, y2),
                "W": W, "H": H,
                "points_px": pts_px,
            })

    print(f"val 实例数: {len(instances)}")

    # ── 规则打分 ──────────────────────────────────────────────────────────
    print("规则打分中...")
    gt_grades, pred_grades = rule_grade_instances(instances, IMG_DIR)

    # ── 计算指标 ──────────────────────────────────────────────────────────
    metrics = compute_metrics(gt_grades, pred_grades)
    print("\n" + "=" * 60)
    print("  规则打分（classify_grades 逻辑）在 val 集上的结果")
    print("=" * 60)
    print_metrics(metrics, prefix="Rule-based Val")
    print(f"\n  Sgrade (QWK) = {metrics['weighted_kappa']:.4f}")

    # 保存结果
    result = {
        "weighted_kappa": metrics["weighted_kappa"],
        "overall_acc": metrics["overall_acc"],
        "macro_acc": metrics["macro_acc"],
        "per_class_acc": metrics["per_class_acc"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    out_path = os.path.join(SAVE_DIR, "rule_based_val_metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  结果保存至: {out_path}")


if __name__ == "__main__":
    main()
