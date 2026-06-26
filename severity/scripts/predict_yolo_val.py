"""
severity/scripts/predict_yolo_val.py

Run YOLO segmentation inference on the val split images and save
predictions in YOLO seg txt format (one file per image).

Output: result/labels_val/<stem>.txt
Each line: class_id x1 y1 x2 y2 ... xn yn  (normalised polygon)

Usage:
  python severity/scripts/predict_yolo_val.py \
      --weights models/yolo_severity_split/train/weights/best.pt \
      --split_file severity/splits/split.json \
      --img_dir Track2/NG_1154/images \
      --output_dir result/labels_val
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",    required=True,
                   help="path to trained YOLO best.pt")
    p.add_argument("--split_file", default="severity/splits/split.json")
    p.add_argument("--img_dir",    default="Track2/NG_1154/images")
    p.add_argument("--output_dir", default="result/labels_val")
    p.add_argument("--imgsz",      type=int, default=512)
    p.add_argument("--conf",       type=float, default=0.25)
    p.add_argument("--iou",        type=float, default=0.45)
    p.add_argument("--device",     default="0")
    p.add_argument("--batch",      type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # load val stems
    with open(args.split_file) as f:
        sp = json.load(f)
    val_stems = sp["val_stems"]
    print(f"val split: {len(val_stems)} images")

    # collect image paths
    img_paths = []
    missing = []
    for stem in val_stems:
        p = os.path.join(args.img_dir, stem + ".bmp")
        if os.path.exists(p):
            img_paths.append((stem, p))
        else:
            missing.append(stem)
    if missing:
        print(f"  [warn] {len(missing)} images not found, skipping")
    print(f"  running inference on {len(img_paths)} images...")

    from ultralytics import YOLO
    model = YOLO(args.weights)

    # run in batches
    batch_size = args.batch
    n_written = 0
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i:i + batch_size]
        stems_batch = [s for s, _ in batch]
        paths_batch = [p for _, p in batch]

        results = model.predict(
            source=paths_batch,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

        for stem, result in zip(stems_batch, results):
            out_path = os.path.join(args.output_dir, stem + ".txt")
            lines = []

            if result.masks is not None and len(result.masks) > 0:
                # normalised polygon segments: shape (N, num_pts, 2)
                segs = result.masks.xyn   # list of (num_pts, 2) arrays, normalised
                cls_ids = result.boxes.cls.int().tolist()

                for cls_id, seg in zip(cls_ids, segs):
                    # flatten to x0 y0 x1 y1 ...
                    coords = " ".join(f"{v:.6f}" for xy in seg for v in xy)
                    lines.append(f"{cls_id} {coords}")

            # write even if empty (empty file = no predictions for this image)
            with open(out_path, "w") as f:
                f.write("\n".join(lines))
            n_written += 1

        if (i // batch_size) % 10 == 0:
            print(f"  processed {min(i + batch_size, len(img_paths))}/{len(img_paths)}")

    print(f"\n  done. {n_written} prediction files written to: {args.output_dir}")
    print(f"\n  Next step — formal evaluation:")
    print(f"  python severity/scripts/eval_predicted_instances.py \\")
    print(f"      --split_file {args.split_file} \\")
    print(f"      --prediction_dir {args.output_dir} \\")
    print(f"      --checkpoint_path severity/results/E5_coral_morph_classemb/best.pth \\")
    print(f"      --output_dir severity/results_predicted/formal_E5 \\")
    print(f"      --eval_mode formal")


if __name__ == "__main__":
    main()
