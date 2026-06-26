"""Train a YOLOv8 segmentation model for defect localization."""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model.")
    parser.add_argument("--weights", default="checkpoints/yolov8x-seg.pt",
                        help="Path to YOLO pretrained weights.")
    parser.add_argument("--data", default="configs/dataset.yaml",
                        help="Path to YOLO dataset YAML.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--device", default="0",
                        help="CUDA device id, e.g. '0' or '0,1'.")
    parser.add_argument("--project", default="checkpoints/yolo_runs",
                        help="Output project directory.")
    parser.add_argument("--name", default="yolov8x-seg-split6",
                        help="Experiment subdirectory name.")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
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

    print("YOLO training completed.")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
