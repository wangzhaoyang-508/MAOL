"""Run MAOL severity grading on YOLO segmentation predictions.

Input:
  <labels_dir>/<stem>.txt in YOLO segmentation format:
    class_id x1 y1 x2 y2 ... xn yn

Output:
  <output_dir>/<stem>.json:
    [{"class": "4", "points": "0.67, 0.07, ...", "severity": "Acceptable"}, ...]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_LABELS_DIR = "result/predict_test/labels"
DEFAULT_IMG_DIR = "data/test/Track2_TestData_Fine-Grained-Severity-Grading"
DEFAULT_OUTPUT_DIR = "result/grading_E6"

CHECKPOINT_MAP = {
    "E1": "severity/results/E1_ce/best.pth",
    "E2": "severity/results/E2_distance_ce/best.pth",
    "E3": "severity/results/E3_coral/best.pth",
    "E4": "severity/results/E4_coral_morphology/best.pth",
    "E5": "severity/results/E5_coral_morph_classemb/best.pth",
    "E6": "severity/results/E6_coral_morph_adaptthres/best.pth",
    "E7": "severity/results/E7_corn/best.pth",
    "E8": "severity/results/E8_coral_morph_predaware/best.pth",
    "E9": "severity/results/E9_coral_morph_adaptthres_predaware/best.pth",
}

E6_HARDCODED_CONFIG = {
    "arch": "resnet18",
    "head_type": "coral",
    "use_morphology": True,
    "use_class_embedding": True,
    "use_adaptive_thresholds": True,
    "morph_hidden_dim": 32,
    "threshold_hidden_dim": 32,
    "class_emb_dim": 16,
    "num_defect_classes": 11,
    "roi_size": 64,
    "pad_ratio": 0.2,
    "morph_stats": None,
}

GRADE_NAMES = ["Acceptable", "Marginal NG", "NG", "Gross NG"]

BG_EXPAND_RATIO = 0.1
FACTORS_CONFIG = [
    {"key": "area", "scores": [3, 5, 7, 9]},
    {"key": "rect_w", "scores": [2.5, 4.5, 6.5, 8.5]},
    {"key": "rect_h", "scores": [3, 5, 7, 9]},
    {"key": "bg_gray", "scores": [2, 4, 6, 8]},
]
WEIGHTS = [0.15, 0.25, 0.25, 0.35]


def parse_yolo_seg_file(txt_path: str) -> list[dict]:
    """Parse one YOLO segmentation TXT file."""
    instances = []
    if not os.path.exists(txt_path):
        return instances

    with open(txt_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    cur_cls = None
    cur_coords: list[float] = []

    def commit():
        if cur_cls is None:
            return
        if len(cur_coords) >= 4 and len(cur_coords) % 2 == 0:
            instances.append({
                "cls_id": int(cur_cls),
                "points_norm": [max(0.0, min(1.0, float(v))) for v in cur_coords],
            })

    for line in raw_lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        starts_new_instance = tokens[0].lstrip("-").isdigit() and "." not in tokens[0]
        if starts_new_instance:
            commit()
            cur_cls = int(tokens[0])
            cur_coords = [float(t) for t in tokens[1:]]
        else:
            cur_coords.extend(float(t) for t in tokens)
    commit()

    return instances


def points_norm_to_str(points_norm: list[float]) -> str:
    return ", ".join(f"{v:.6f}" for v in points_norm)


def compute_rule_features(points_norm: list[float], img_w: int, img_h: int,
                          img_gray: np.ndarray) -> dict:
    xs = [points_norm[i] * img_w for i in range(0, len(points_norm), 2)]
    ys = [points_norm[i] * img_h for i in range(1, len(points_norm), 2)]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    area = (x2 - x1) * (y2 - y1)
    rect_w = x2 - x1
    rect_h = y2 - y1

    bx1 = max(0, int(x1 - BG_EXPAND_RATIO * rect_w))
    bx2 = min(img_w - 1, int(x2 + BG_EXPAND_RATIO * rect_w))
    by1 = max(0, int(y1 - BG_EXPAND_RATIO * rect_h))
    by2 = min(img_h - 1, int(y2 + BG_EXPAND_RATIO * rect_h))
    bg_region = img_gray[by1:by2 + 1, bx1:bx2 + 1]
    bg_gray = float(np.mean(bg_region)) if bg_region.size > 0 else 128.0
    return {"area": area, "rect_w": rect_w, "rect_h": rect_h, "bg_gray": bg_gray}


def rule_based_grade(features: dict) -> int:
    total_score = 0.0
    for cfg, weight in zip(FACTORS_CONFIG, WEIGHTS):
        value = features[cfg["key"]]
        scores = cfg["scores"]
        if value <= scores[0]:
            score = 1.0
        elif value <= scores[1]:
            score = 3.0
        elif value <= scores[2]:
            score = 5.0
        elif value <= scores[3]:
            score = 7.0
        else:
            score = 9.0
        total_score += weight * score

    if total_score <= 3.0:
        return 0
    if total_score <= 5.0:
        return 1
    if total_score <= 7.0:
        return 2
    return 3


def _read_checkpoint_config(checkpoint_path: str, method: str | None) -> dict:
    if method == "E6":
        return dict(E6_HARDCODED_CONFIG)

    cfg_path = Path(checkpoint_path).parent / "config.json"
    raw_cfg = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

    def as_bool(key: str, default: str = "false") -> bool:
        return str(raw_cfg.get(key, default)).lower() in {"true", "1", "yes"}

    cfg = {
        "arch": raw_cfg.get("arch", "resnet18"),
        "head_type": raw_cfg.get("head_type", "coral"),
        "use_morphology": as_bool("use_morphology", "true"),
        "use_class_embedding": as_bool("use_class_embedding", "false"),
        "use_adaptive_thresholds": as_bool("use_adaptive_thresholds", "false"),
        "morph_hidden_dim": int(raw_cfg.get("morph_hidden_dim", 32)),
        "threshold_hidden_dim": int(raw_cfg.get("threshold_hidden_dim", 32)),
        "class_emb_dim": int(raw_cfg.get("class_emb_dim", 16)),
        "num_defect_classes": int(raw_cfg.get("num_defect_classes", 11)),
        "roi_size": int(raw_cfg.get("roi_size", 64)),
        "pad_ratio": float(raw_cfg.get("pad_ratio", 0.2)),
        "morph_stats": None,
    }

    stats_path = Path(checkpoint_path).parent / "morph_stats.json"
    if cfg["use_morphology"] and stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as f:
            cfg["morph_stats"] = json.load(f)

    if cfg["use_adaptive_thresholds"]:
        cfg["use_class_embedding"] = True

    return cfg


def load_model_and_config(checkpoint_path: str, device, method: str | None = None):
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from severity.datasets.severity_roi_dataset import MORPH_DIM
    from severity.models.severity_classifier import build_model

    cfg = _read_checkpoint_config(checkpoint_path, method)
    if cfg["use_morphology"] and cfg["morph_stats"] is None:
        cfg["morph_stats"] = {"mean": [0.0] * MORPH_DIM, "std": [1.0] * MORPH_DIM}

    model = build_model(
        arch=cfg["arch"],
        num_classes=4,
        pretrained=False,
        dropout=0.3,
        head_type=cfg["head_type"],
        use_morphology=cfg["use_morphology"],
        morph_dim=MORPH_DIM,
        morph_hidden_dim=cfg["morph_hidden_dim"],
        use_class_embedding=cfg["use_class_embedding"],
        num_defect_classes=cfg["num_defect_classes"],
        class_emb_dim=cfg["class_emb_dim"],
        use_adaptive_thresholds=cfg.get("use_adaptive_thresholds", False),
        threshold_hidden_dim=cfg.get("threshold_hidden_dim", 32),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    remapped = {}
    for key, value in state.items():
        if key == "net.fc.1.weight":
            remapped["head.weight"] = value
        elif key == "net.fc.1.bias":
            remapped["head.bias"] = value
        else:
            remapped[key] = value
    model.load_state_dict(remapped, strict=True)
    model.eval()

    return model, cfg


def _bbox_from_points(points: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    xs = [points[i] * width for i in range(0, len(points), 2)]
    ys = [points[i] * height for i in range(1, len(points), 2)]
    x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
    x2, y2 = min(width - 1, int(max(xs))), min(height - 1, int(max(ys)))
    if x2 <= x1:
        x2 = min(x1 + 1, width - 1)
    if y2 <= y1:
        y2 = min(y1 + 1, height - 1)
    return x1, y1, x2, y2


def predict_instance(model, cfg: dict, inst: dict, img: Image.Image, device) -> int:
    import torch
    import torchvision.transforms as transforms

    from severity.datasets.severity_roi_dataset import (
        compute_morph_features,
        expand_bbox,
        normalize_morph,
    )
    from severity.utils.losses import coral_predict, corn_predict

    width, height = img.size
    points = inst["points_norm"]
    x1, y1, x2, y2 = _bbox_from_points(points, width, height)
    ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2, width, height, cfg["pad_ratio"])
    roi = img.crop((ex1, ey1, ex2, ey2))

    transform = transforms.Compose([
        transforms.Resize((cfg["roi_size"], cfg["roi_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(roi).unsqueeze(0).to(device)

    if cfg["use_morphology"]:
        img_gray = np.array(img.convert("L"), dtype=np.uint8)
        raw = compute_morph_features(points, width, height, img_gray)
        norm = normalize_morph(raw, cfg["morph_stats"])
        morph = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(device)
        if cfg["use_class_embedding"]:
            cls_id = torch.tensor([inst["cls_id"]], dtype=torch.long).to(device)
            logits = model(img_tensor, morph=morph, class_id=cls_id)
        else:
            logits = model(img_tensor, morph=morph)
    else:
        logits = model(img_tensor)

    if cfg["head_type"] == "coral":
        return int(coral_predict(logits).item())
    if cfg["head_type"] == "corn":
        return int(corn_predict(logits).item())
    return int(logits.argmax(1).item())


def parse_args():
    import torch

    parser = argparse.ArgumentParser(description="Run MAOL severity grading.")
    parser.add_argument("--method", default="E6",
                        choices=["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
                        help="Severity method. E0 is rule-based; E6 is MAOL adaptive CORAL.")
    parser.add_argument("--labels_dir", default=DEFAULT_LABELS_DIR,
                        help="Directory with YOLO segmentation TXT predictions.")
    parser.add_argument("--img_dir", default=DEFAULT_IMG_DIR,
                        help="Directory with corresponding images.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to severity model checkpoint.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory for per-image JSON outputs.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _find_image(img_dir: str, stem: str) -> str | None:
    for ext in (".bmp", ".jpg", ".jpeg", ".png"):
        path = os.path.join(img_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isdir(args.labels_dir):
        print(f"[error] labels_dir not found: {args.labels_dir}")
        sys.exit(1)

    model = None
    model_cfg = None
    device = None
    if args.method != "E0":
        import torch

        checkpoint_path = args.checkpoint or CHECKPOINT_MAP.get(args.method)
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"[error] checkpoint not found: {checkpoint_path}")
            print("Provide --checkpoint or place weights at the default checkpoint path.")
            sys.exit(1)

        device = torch.device(args.device)
        print(f"Loading severity model: {checkpoint_path}")
        model, model_cfg = load_model_and_config(checkpoint_path, device, method=args.method)
        print(
            "Model config: "
            f"head={model_cfg['head_type']} "
            f"morph={model_cfg['use_morphology']} "
            f"class_embedding={model_cfg['use_class_embedding']} "
            f"adaptive_thresholds={model_cfg.get('use_adaptive_thresholds', False)}"
        )

    txt_files = sorted(f for f in os.listdir(args.labels_dir) if f.endswith(".txt"))
    print(f"Processing {len(txt_files)} prediction files.")

    for txt_name in txt_files:
        stem = os.path.splitext(txt_name)[0]
        txt_path = os.path.join(args.labels_dir, txt_name)
        instances = parse_yolo_seg_file(txt_path)

        out_path = os.path.join(args.output_dir, stem + ".json")
        if not instances:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            continue

        img_path = _find_image(args.img_dir, stem)
        if img_path is None:
            print(f"[warn] image not found for {stem}; skipping.")
            continue

        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        img_gray = np.array(img.convert("L"), dtype=np.uint8)

        results = []
        for inst in instances:
            if args.method == "E0":
                features = compute_rule_features(inst["points_norm"], width, height, img_gray)
                grade_id = rule_based_grade(features)
            else:
                grade_id = predict_instance(model, model_cfg, inst, img, device)

            results.append({
                "class": str(inst["cls_id"]),
                "points": points_norm_to_str(inst["points_norm"]),
                "severity": GRADE_NAMES[grade_id],
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Severity grading completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
