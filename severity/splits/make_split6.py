"""Generate a train/validation split from image folders."""

import argparse
import json
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_stems(folder: Path) -> list[str]:
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    return sorted(p.stem for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def parse_args():
    parser = argparse.ArgumentParser(description="Build split JSON from YOLO image folders.")
    parser.add_argument("--dataset_dir", required=True,
                        help="Dataset root with images/train and images/val.")
    parser.add_argument("--output", default=Path(__file__).parent / "split6.json")
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output = Path(args.output)

    train_stems = get_stems(dataset_dir / "images" / "train")
    val_stems = get_stems(dataset_dir / "images" / "val")

    split = {
        "val_ratio": len(val_stems) / max(len(train_stems) + len(val_stems), 1),
        "seed": args.seed,
        "train_stems": train_stems,
        "val_stems": val_stems,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    print(f"split saved to {output}")
    print(f"train={len(train_stems)} val={len(val_stems)}")


if __name__ == "__main__":
    main()
