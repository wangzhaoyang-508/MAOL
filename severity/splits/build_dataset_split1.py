"""
severity/splits/build_dataset_split1.py

根据 split1.json 把图片和标签按 train/val/test 复制到 YOLO 格式目录：

  dataset_split1/
    images/
      train/   *.bmp
      val/     *.bmp
      test/    *.bmp
    labels/
      train/   *.txt   (YOLO seg 格式)
      val/     *.txt
      test/    *.txt
    data.yaml

JSON 标签 -> YOLO seg txt 转换规则：
  每行：<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
  坐标已经是归一化值，直接写入即可。

用法：
  python severity/splits/build_dataset_split1.py
"""

import json
import os
import shutil

SRC_IMG_DIR   = "Track2/NG_1154/images"
SRC_LABEL_DIR = "Track2/NG_1154/level_labels"
SPLIT_FILE    = "severity/splits/split1.json"
OUT_DIR       = "dataset_split1"
CLASS_FILE    = "Track2/class_name.txt"


def json_to_yolo_txt(json_path: str, txt_path: str):
    """Convert a level_labels JSON file to YOLO seg txt format."""
    with open(json_path, encoding="utf-8") as f:
        anns = json.load(f)

    lines = []
    for ann in anns:
        cls_id = int(ann["class"])
        # points: "x1, y1, x2, y2, ..." -> space-separated floats
        coords = [v.strip() for v in ann["points"].split(",")]
        line = str(cls_id) + " " + " ".join(coords)
        lines.append(line)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    with open(SPLIT_FILE, encoding="utf-8") as f:
        split = json.load(f)

    with open(CLASS_FILE, encoding="utf-8") as f:
        class_names = [l.strip() for l in f if l.strip()]

    splits = {
        "train": split["train_stems"],
        "val":   split["val_stems"],
        "test":  split["test_stems"],
    }

    # create output dirs
    for subset in splits:
        os.makedirs(os.path.join(OUT_DIR, "images", subset), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", subset), exist_ok=True)

    stats = {s: {"imgs": 0, "labels": 0, "missing_img": 0, "missing_label": 0}
             for s in splits}

    for subset, stems in splits.items():
        for stem in stems:
            img_src   = os.path.join(SRC_IMG_DIR,   stem + ".bmp")
            label_src = os.path.join(SRC_LABEL_DIR, stem + ".json")
            img_dst   = os.path.join(OUT_DIR, "images", subset, stem + ".bmp")
            label_dst = os.path.join(OUT_DIR, "labels", subset, stem + ".txt")

            # copy image
            if os.path.exists(img_src):
                shutil.copy2(img_src, img_dst)
                stats[subset]["imgs"] += 1
            else:
                print(f"  [warn] image not found: {img_src}")
                stats[subset]["missing_img"] += 1

            # convert label
            if os.path.exists(label_src):
                json_to_yolo_txt(label_src, label_dst)
                stats[subset]["labels"] += 1
            else:
                print(f"  [warn] label not found: {label_src}")
                stats[subset]["missing_label"] += 1

    # write data.yaml
    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    nc = len(class_names)
    # build names dict for yaml (index: name)
    names_lines = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    yaml_content = f"""path: {os.path.abspath(OUT_DIR)}
train: images/train
val:   images/val
test:  images/test
nc: {nc}
names:
{names_lines}
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # summary
    print("\ndataset_split1 build complete:")
    for subset, s in stats.items():
        print(f"  {subset:5s}: {s['imgs']} images, {s['labels']} labels"
              + (f"  [missing img={s['missing_img']}]" if s["missing_img"] else "")
              + (f"  [missing label={s['missing_label']}]" if s["missing_label"] else ""))
    print(f"  data.yaml -> {yaml_path}")
    print(f"  output dir -> {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
