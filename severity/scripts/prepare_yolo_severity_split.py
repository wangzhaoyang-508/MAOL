"""
severity/scripts/prepare_yolo_severity_split.py

Re-organise the existing dataset/ folder into a new YOLO dataset
that uses the severity split.json train/val division.

Source images/labels come from dataset/images/{train,val,test}/
and dataset/labels/{train,val,test}/.

Output layout:
  dataset_severity_split/
    images/
      train/   <- symlinks or copies of split.json train stems
      val/     <- symlinks or copies of split.json val stems
    labels/
      train/
      val/
    dataset.yaml

Usage:
  python severity/scripts/prepare_yolo_severity_split.py
  python severity/scripts/prepare_yolo_severity_split.py --copy  # copy instead of symlink
"""

import argparse
import json
import os
import shutil

SPLIT_FILE  = "severity/splits/split.json"
SRC_IMG_DIRS = [
    "dataset/images/train",
    "dataset/images/val",
    "dataset/images/test",
]
SRC_LBL_DIRS = [
    "dataset/labels/train",
    "dataset/labels/val",
    "dataset/labels/test",
]
OUT_DIR = "dataset_severity_split"


def find_file(stem: str, dirs: list, ext: str):
    """Return the first existing path for stem+ext across dirs, or None."""
    for d in dirs:
        p = os.path.join(d, stem + ext)
        if os.path.exists(p):
            return p
    return None


def link_or_copy(src: str, dst: str, do_copy: bool):
    if os.path.exists(dst):
        return
    if do_copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_file", default=SPLIT_FILE)
    p.add_argument("--out_dir",    default=OUT_DIR)
    p.add_argument("--copy",       action="store_true",
                   help="copy files instead of creating symlinks")
    args = p.parse_args()

    with open(args.split_file) as f:
        sp = json.load(f)

    splits = {"train": sp["train_stems"], "val": sp["val_stems"]}

    for split, stems in splits.items():
        img_out = os.path.join(args.out_dir, "images", split)
        lbl_out = os.path.join(args.out_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        n_img, n_lbl, n_missing_img, n_missing_lbl = 0, 0, 0, 0
        for stem in stems:
            img_src = find_file(stem, SRC_IMG_DIRS, ".bmp")
            lbl_src = find_file(stem, SRC_LBL_DIRS, ".txt")

            if img_src:
                link_or_copy(img_src, os.path.join(img_out, stem + ".bmp"), args.copy)
                n_img += 1
            else:
                print(f"  [warn] image not found: {stem}")
                n_missing_img += 1

            if lbl_src:
                link_or_copy(lbl_src, os.path.join(lbl_out, stem + ".txt"), args.copy)
                n_lbl += 1
            else:
                # no label = image with no defects; YOLO handles missing label files
                n_missing_lbl += 1

        print(f"  [{split}] {n_img} images, {n_lbl} labels "
              f"(missing img={n_missing_img}, missing lbl={n_missing_lbl})")

    # write dataset.yaml
    # read nc and names from original yaml
    import yaml
    with open("dataset/dataset.yaml") as f:
        orig = yaml.safe_load(f)

    out_yaml = {
        "path": os.path.abspath(args.out_dir),
        "train": "images/train",
        "val":   "images/val",
        "nc":    orig["nc"],
        "names": orig["names"],
    }
    yaml_path = os.path.join(args.out_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(out_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\n  dataset.yaml written to: {yaml_path}")
    print(f"  out_dir: {os.path.abspath(args.out_dir)}")
    print(f"\n  Next steps:")
    print(f"  1. Train YOLO:")
    print(f"     python severity/scripts/train_yolo_severity_split.py")
    print(f"  2. Predict on val split:")
    print(f"     python severity/scripts/predict_yolo_val.py --weights <best.pt>")


if __name__ == "__main__":
    main()
