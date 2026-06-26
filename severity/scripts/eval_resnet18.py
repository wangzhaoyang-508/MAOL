"""
severity/scripts/eval_resnet18.py

第一组实验：加载训练好的 ResNet18 checkpoint，
在与第二组相同的 val split 上推理，输出 Sgrade（QWK）。

用法:
    python severity/scripts/eval_resnet18.py
"""

import os
import sys
import json

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from severity.datasets.severity_roi_dataset import SeverityROIDataset, collect_instances
from severity.models.severity_classifier import build_model
from severity.utils.grade_schema import GRADE_MAP, GRADE_NAMES
from severity.utils.metrics import compute_metrics, print_metrics

IMG_DIR    = "Track2/NG_1154/images"
LABEL_DIR  = "Track2/NG_1154/level_labels"
SPLIT_PATH = "severity/splits/split.json"
CKPT_PATH  = "severity/checkpoints/best_resnet18.pth"
SAVE_DIR   = "severity/results/resnet18"

ROI_SIZE   = 64
PAD_RATIO  = 0.2
BATCH_SIZE = 64
ARCH       = "resnet18"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


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

    # ── 收集 val 实例 ─────────────────────────────────────────────────────
    all_inst = collect_instances(IMG_DIR, LABEL_DIR)
    val_inst = [i for i in all_inst if i["img_stem"] in val_stems]
    print(f"val 实例数: {len(val_inst)}")

    val_ds = SeverityROIDataset(val_inst, roi_size=ROI_SIZE, pad_ratio=PAD_RATIO, augment=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # ── 加载模型 ──────────────────────────────────────────────────────────
    device = torch.device(DEVICE)
    model = build_model(arch=ARCH, num_classes=4, dropout=0.3).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    print(f"加载权重: {CKPT_PATH}  (best epoch={ckpt.get('epoch','?')}, κ={ckpt.get('best_kappa','?')})")

    # ── 推理 ──────────────────────────────────────────────────────────────
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # ── 指标 ──────────────────────────────────────────────────────────────
    metrics = compute_metrics(all_labels, all_preds)
    print("\n" + "=" * 60)
    print("  ResNet18（GT ROI）在 val 集上的结果")
    print("=" * 60)
    print_metrics(metrics, prefix="ResNet18 Val")
    print(f"\n  Sgrade (QWK) = {metrics['weighted_kappa']:.4f}")

    result = {
        "weighted_kappa": metrics["weighted_kappa"],
        "overall_acc": metrics["overall_acc"],
        "macro_acc": metrics["macro_acc"],
        "per_class_acc": metrics["per_class_acc"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    out_path = os.path.join(SAVE_DIR, "resnet18_val_metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  结果保存至: {out_path}")


if __name__ == "__main__":
    main()
