"""
severity/scripts/make_split.py

按 image 80/20 划分，把 train/val stem 列表保存到 severity/splits/split.json。
两组实验（ResNet18 分类 vs 规则打分）共用同一份划分。

用法:
    python severity/scripts/make_split.py
"""

import os
import sys
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from severity.datasets.severity_roi_dataset import collect_instances, split_by_image

IMG_DIR   = "Track2/NG_1154/images"
LABEL_DIR = "Track2/NG_1154/level_labels"
SAVE_PATH = "severity/splits/split.json"
VAL_RATIO = 0.2
SEED      = 42


def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print("收集实例...")
    all_inst = collect_instances(IMG_DIR, LABEL_DIR)
    train_inst, val_inst = split_by_image(all_inst, VAL_RATIO, SEED)

    train_stems = sorted(set(i["img_stem"] for i in train_inst))
    val_stems   = sorted(set(i["img_stem"] for i in val_inst))

    split = {
        "val_ratio": VAL_RATIO,
        "seed": SEED,
        "train_stems": train_stems,
        "val_stems": val_stems,
    }
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    print(f"划分完成:")
    print(f"  train: {len(train_stems)} 张图, {len(train_inst)} 个实例")
    print(f"  val  : {len(val_stems)} 张图, {len(val_inst)} 个实例")
    print(f"  保存至: {SAVE_PATH}")


if __name__ == "__main__":
    main()
