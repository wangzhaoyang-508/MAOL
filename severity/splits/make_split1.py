"""
severity/splits/make_split1.py

按 baseline 规则生成 split1.json：
  - 每个 defect class 各抽 20 张图作为 test set
  - 剩余图按 8:2 分 train / val
  - 以图像 stem 为单位划分（同一张图的所有 instance 不跨 split）
  - 固定 seed=42 保证可复现

输出：severity/splits/split1.json
  {
    "train_stems": [...],
    "val_stems":   [...],
    "test_stems":  [...],
    "seed": 42,
    "test_per_class": 20,
    "val_ratio": 0.2
  }

用法：
  python severity/splits/make_split1.py
"""

import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

IMG_DIR   = "Track2/NG_1154/images"
LABEL_DIR = "Track2/NG_1154/level_labels"
OUT_PATH  = "severity/splits/split1.json"

SEED           = 42
TEST_PER_CLASS = 20
VAL_RATIO      = 0.2


def main():
    rng = random.Random(SEED)

    # ── 1. 收集每张图包含哪些 class ──────────────────────────────────────
    # stem -> set of class_ids
    stem_classes: dict[str, set] = defaultdict(set)

    for fname in sorted(os.listdir(LABEL_DIR)):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]
        img_path = os.path.join(IMG_DIR, stem + ".bmp")
        if not os.path.exists(img_path):
            continue
        with open(os.path.join(LABEL_DIR, fname), encoding="utf-8") as f:
            anns = json.load(f)
        for ann in anns:
            try:
                stem_classes[stem].add(int(ann["class"]))
            except (KeyError, ValueError):
                pass

    all_stems = sorted(stem_classes.keys())
    print(f"total images with labels: {len(all_stems)}")

    # ── 2. 按 class 分组（一张图可能属于多个 class，取第一个出现的 class） ──
    # 与 baseline split_data.py 逻辑一致：img_dict[class] = [stems...]
    class_to_stems: dict[int, list] = defaultdict(list)
    for stem in all_stems:
        for cls_id in stem_classes[stem]:
            class_to_stems[cls_id].append(stem)

    # 去重（同一 stem 可能被多个 class 收录）
    for cls_id in class_to_stems:
        class_to_stems[cls_id] = sorted(set(class_to_stems[cls_id]))

    print(f"defect classes found: {sorted(class_to_stems.keys())}")
    for cls_id in sorted(class_to_stems.keys()):
        print(f"  class {cls_id:>2}: {len(class_to_stems[cls_id])} images")

    # ── 3. 每个 class 抽 TEST_PER_CLASS 张作为 test ───────────────────────
    test_stems_set: set = set()
    remaining_by_class: dict[int, list] = {}

    for cls_id in sorted(class_to_stems.keys()):
        stems = list(class_to_stems[cls_id])
        rng.shuffle(stems)
        n_test = min(TEST_PER_CLASS, len(stems))
        test_stems_set.update(stems[:n_test])
        remaining_by_class[cls_id] = stems[n_test:]

    # ── 4. 剩余图（未被任何 class 选为 test 的）按 8:2 分 train/val ───────
    # 收集所有非 test stems，去重
    remaining_stems = sorted(set(all_stems) - test_stems_set)
    rng.shuffle(remaining_stems)

    n_val   = max(1, int(len(remaining_stems) * VAL_RATIO))
    val_stems   = sorted(remaining_stems[:n_val])
    train_stems = sorted(remaining_stems[n_val:])
    test_stems  = sorted(test_stems_set)

    print(f"\nsplit1 summary:")
    print(f"  train : {len(train_stems)} images")
    print(f"  val   : {len(val_stems)} images")
    print(f"  test  : {len(test_stems)} images")
    print(f"  total : {len(train_stems)+len(val_stems)+len(test_stems)} images")

    # sanity check: no overlap
    assert not (set(train_stems) & set(val_stems)),  "train/val overlap!"
    assert not (set(train_stems) & set(test_stems)), "train/test overlap!"
    assert not (set(val_stems)   & set(test_stems)), "val/test overlap!"
    print("  overlap check: OK")

    # ── 5. 保存 ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    result = {
        "seed":           SEED,
        "test_per_class": TEST_PER_CLASS,
        "val_ratio":      VAL_RATIO,
        "train_stems":    train_stems,
        "val_stems":      val_stems,
        "test_stems":     test_stems,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
