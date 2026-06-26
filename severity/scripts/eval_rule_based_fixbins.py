"""
severity/scripts/eval_rule_based_fixbins.py

规则打分（classify_grades 逻辑）的修正版：
  - 用 train 集特征分布计算四分位分箱阈值（固定）
  - 把固定阈值应用到 val 集打分
  - 与 eval_rule_based.py（val 集自身分箱）和 ResNet18 结果对比

用法:
    python severity/scripts/eval_rule_based_fixbins.py
"""

import os, sys, json
import numpy as np
import cv2
from PIL import Image as PILImage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from severity.datasets.severity_roi_dataset import parse_points, points_to_bbox_pixel
from severity.utils.grade_schema import GRADE_MAP, GRADE_NAMES
from severity.utils.metrics import compute_metrics, print_metrics

IMG_DIR    = "Track2/NG_1154/images"
LABEL_DIR  = "Track2/NG_1154/level_labels"
SPLIT_PATH = "severity/splits/split.json"
SAVE_DIR   = "severity/results/rule_based_fixbins"

BG_EXPAND_RATIO = 0.1
FACTOR_COLS = ["area", "rect_w", "rect_h", "bg_gray"]
FACTOR_SCORES = {
    "area":    [3, 5, 7, 9],
    "rect_w":  [2.5, 4.5, 6.5, 8.5],
    "rect_h":  [3, 5, 7, 9],
    "bg_gray": [2, 4, 6, 8],
}
WEIGHTS = [0.15, 0.25, 0.25, 0.35]

def extract_features(instances):
    """提取每个实例的 4 个特征，返回 list of dict"""
    feats = []
    for inst in instances:
        try:
            img = PILImage.open(inst["img_path"]).convert("L")
            gray = np.array(img, dtype=np.uint8)
            W, H = inst["W"], inst["H"]
            x1, y1, x2, y2 = inst["bbox"]
            pts = inst["points_px"]

            # 面积（Shoelace）
            if len(pts) >= 3:
                arr = np.array(pts, dtype=np.float64)
                x_a, y_a = arr[:, 0], arr[:, 1]
                area = 0.5 * abs(np.sum(x_a * np.roll(y_a, -1) - np.roll(x_a, -1) * y_a))
            else:
                area = float((x2 - x1) * (y2 - y1))

            rect_w = float(x2 - x1)
            rect_h = float(y2 - y1)

            # 背景灰度
            ew = max(1, int(rect_w * BG_EXPAND_RATIO))
            eh = max(1, int(rect_h * BG_EXPAND_RATIO))
            bx1, by1 = max(0, x1 - ew), max(0, y1 - eh)
            bx2, by2 = min(W, x2 + ew), min(H, y2 + eh)
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

            feats.append({"area": area, "rect_w": rect_w, "rect_h": rect_h, "bg_gray": bg_gray})
        except Exception as e:
            print(f"  警告: 特征提取失败 {inst['img_path']}: {e}")
            feats.append({"area": 0.0, "rect_w": 0.0, "rect_h": 0.0, "bg_gray": 128.0})
    return feats


def compute_bins_from(feats):
    """用给定特征集合计算四分位分箱阈值（与 classify_grades.py 完全一致）"""
    bins = {}
    for col in FACTOR_COLS:
        vals = sorted(f[col] for f in feats)
        n = len(vals)
        bins[col] = [vals[0], vals[n // 4], vals[2 * n // 4], vals[3 * n // 4], vals[-1] + 1e-9]
    return bins


def apply_bins(feats, bins):
    """用固定 bins 对每个实例打分，返回预测等级列表"""
    preds = []
    for feat in feats:
        weighted = 0.0
        for col, w in zip(FACTOR_COLS, WEIGHTS):
            scores = FACTOR_SCORES[col]
            b = bins[col]
            score = scores[-1]
            for i in range(len(b) - 1):
                if b[i] <= feat[col] < b[i + 1]:
                    score = scores[i]
                    break
            weighted += score * w
        if weighted <= 3:
            preds.append(GRADE_MAP["Acceptable"])
        elif weighted <= 4.5:
            preds.append(GRADE_MAP["Marginal NG"])
        elif weighted <= 7.5:
            preds.append(GRADE_MAP["NG"])
        else:
            preds.append(GRADE_MAP["Gross NG"])
    return preds


def load_instances(stems):
    """加载指定 stem 集合的实例（含多边形像素坐标）"""
    instances = []
    for fname in sorted(os.listdir(LABEL_DIR)):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]
        if stem not in stems:
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
            pts_px = [(round(points[i] * W), round(points[i + 1] * H))
                      for i in range(0, len(points), 2)]
            instances.append({
                "img_path": img_path, "img_stem": stem,
                "grade_id": GRADE_MAP[sev],
                "bbox": (x1, y1, x2, y2), "W": W, "H": H,
                "points_px": pts_px,
            })
    return instances


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="用全部图像（train+val）计算，分箱也用全量")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    train_stems = set(split["train_stems"])
    val_stems   = set(split["val_stems"])

    if args.all:
        all_stems = train_stems | val_stems
        print(f"加载全量实例（train+val，{len(all_stems)} 张图）...")
        eval_inst = load_instances(all_stems)
        print(f"  共 {len(eval_inst)} 个实例")
        eval_feats = extract_features(eval_inst)
        fixed_bins = compute_bins_from(eval_feats)   # 用全量自身分箱
        tag = "全量（train+val）"
        out_name = "rule_fixbins_all_metrics.json"
    else:
        print(f"加载 train 实例（用于计算分箱阈值）...")
        train_inst = load_instances(train_stems)
        print(f"  train: {len(train_inst)} 个实例")
        print(f"加载 val 实例...")
        eval_inst = load_instances(val_stems)
        print(f"  val  : {len(eval_inst)} 个实例")
        train_feats = extract_features(train_inst)
        eval_feats  = extract_features(eval_inst)
        fixed_bins  = compute_bins_from(train_feats)
        tag = "val（固定 train 分箱）"
        out_name = "rule_fixbins_val_metrics.json"

    print("提取特征...")
    eval_feats = extract_features(eval_inst)

    print(f"\n分箱阈值（来源：{'全量' if args.all else 'train 集'}）:")
    for col in FACTOR_COLS:
        b = fixed_bins[col]
        print(f"  {col:10s}: [{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}, {b[4]:.2f}]")

    gt_grades   = [i["grade_id"] for i in eval_inst]
    pred_grades = apply_bins(eval_feats, fixed_bins)

    metrics = compute_metrics(gt_grades, pred_grades)
    print("\n" + "=" * 60)
    print(f"  规则打分 — {tag}")
    print("=" * 60)
    print_metrics(metrics, prefix=f"Rule {tag}")
    print(f"\n  Sgrade (QWK) = {metrics['weighted_kappa']:.4f}")

    result = {
        "tag": tag,
        "weighted_kappa": metrics["weighted_kappa"],
        "overall_acc": metrics["overall_acc"],
        "macro_acc": metrics["macro_acc"],
        "per_class_acc": metrics["per_class_acc"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "fixed_bins": {k: list(v) for k, v in fixed_bins.items()},
    }
    out_path = os.path.join(SAVE_DIR, out_name)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  结果保存至: {out_path}")


if __name__ == "__main__":
    main()
