"""
severity/scripts/train_yolo_severity_split.py

Train YOLO segmentation model on the severity-split dataset.
Dataset must be prepared first with prepare_yolo_severity_split.py.

Usage:
  python severity/scripts/train_yolo_severity_split.py
  python severity/scripts/train_yolo_severity_split.py --device 0 --epochs 300
"""

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",    default="baseline/yolov8l-seg.pt")
    p.add_argument("--data",       default="dataset_severity_split/dataset.yaml")
    p.add_argument("--project",    default="models/yolo_severity_split")
    p.add_argument("--name",       default="train")
    p.add_argument("--epochs",     type=int, default=300)
    p.add_argument("--imgsz",      type=int, default=512)
    p.add_argument("--batch",      type=int, default=48)
    p.add_argument("--device",     default="0")
    p.add_argument("--workers",    type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: dataset yaml not found: {args.data}")
        print("Run prepare_yolo_severity_split.py first.")
        return

    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        project=args.project,
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
