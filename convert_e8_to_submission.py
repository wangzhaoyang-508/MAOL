"""Convert per-image severity JSON files into a submission-style JSON file."""

import argparse
import glob
import json
import os
from pathlib import Path

from PIL import Image


DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512


def find_image_size(stem: str, search_dirs: list[str]) -> tuple[int, int]:
    for directory in search_dirs:
        for ext in (".bmp", ".jpg", ".jpeg", ".png"):
            path = Path(directory) / f"{stem}{ext}"
            if path.exists():
                try:
                    with Image.open(path) as img:
                        return img.width, img.height
                except OSError:
                    pass
    return DEFAULT_WIDTH, DEFAULT_HEIGHT


def points_str_to_list(points_str: str) -> list[float]:
    return [round(float(v.strip()), 6) for v in points_str.split(",") if v.strip()]


def points_to_bbox(points: list[float]) -> list[float]:
    xs = points[0::2]
    ys = points[1::2]
    return [round(min(xs), 6), round(min(ys), 6), round(max(xs), 6), round(max(ys), 6)]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MAOL grading JSON files.")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing per-image JSON files from grade_severity.py.")
    parser.add_argument("--output", required=True,
                        help="Output JSON path.")
    parser.add_argument("--img_search", nargs="+",
                        default=["data/Track2/NG_1154/images"],
                        help="Image directories used to recover width and height.")
    return parser.parse_args()


def main():
    args = parse_args()
    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    print(f"Found {len(json_files)} JSON files.")

    results = []
    for json_path in json_files:
        stem = Path(json_path).stem
        width, height = find_image_size(stem, args.img_search)

        with open(json_path, "r", encoding="utf-8") as f:
            try:
                detections = json.load(f)
            except json.JSONDecodeError as exc:
                print(f"[warn] failed to parse {json_path}: {exc}")
                detections = []

        defect_info = []
        for det in detections:
            segmentation = points_str_to_list(det.get("points", ""))
            bbox = points_to_bbox(segmentation) if len(segmentation) >= 4 else [0, 0, 0, 0]
            defect_info.append({
                "category_id": int(det.get("class", 0)),
                "severity": det.get("severity", "Acceptable"),
                "bbox": bbox,
                "segmentation": segmentation,
            })

        results.append({
            "file_name": f"{stem}.jpg",
            "width": width,
            "height": height,
            "defect_info": defect_info,
        })

    output = {"results": results}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Submission file saved to: {output_path}")


if __name__ == "__main__":
    main()
