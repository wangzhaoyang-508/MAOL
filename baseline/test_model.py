"""Run YOLOv8 segmentation inference and save YOLO TXT predictions."""

import argparse
import os

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO segmentation inference.")
    parser.add_argument("--weights", default="checkpoints/yolo_best.pt",
                        help="Path to trained YOLO weights.")
    parser.add_argument("--source", required=True,
                        help="Input image directory or single image path.")
    parser.add_argument("--output", default="result/predict_test",
                        help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=512,
                        help="Inference image size.")
    parser.add_argument("--device", default="0",
                        help="CUDA device id.")
    parser.add_argument("--save_vis", action="store_true",
                        help="Save visualization images.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save_vis,
        save_txt=True,
        save_conf=False,
        project=args.output,
        name="",
        exist_ok=True,
        vid_stride=1,
    )

    total_masks = 0
    total_area = 0.0
    for result in results:
        if result.masks is not None:
            areas = result.masks.data.sum(dim=(1, 2))
            total_masks += len(areas)
            total_area += areas.float().sum().item()

    print("YOLO inference completed.")
    print(f"Labels saved to: {args.output}/labels/")
    if total_masks > 0:
        print(f"Mask count: {total_masks}, average mask area: {total_area / total_masks:.1f} px")


if __name__ == "__main__":
    main()
