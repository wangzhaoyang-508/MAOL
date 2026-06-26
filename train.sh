#!/usr/bin/env bash
set -euo pipefail

# End-to-end training entry point for MAOL.
# Edit these paths for your local dataset and checkpoint locations.

DATA_ROOT="${DATA_ROOT:-data}"
YOLO_PRETRAIN="${YOLO_PRETRAIN:-checkpoints/yolov8x-seg.pt}"
YOLO_DATA_YAML="${YOLO_DATA_YAML:-configs/dataset.yaml}"
YOLO_OUTPUT_DIR="${YOLO_OUTPUT_DIR:-checkpoints/yolo_runs}"

SEV_IMG_DIR="${SEV_IMG_DIR:-${DATA_ROOT}/Track2/NG_1154/images}"
SEV_LABEL_DIR="${SEV_LABEL_DIR:-${DATA_ROOT}/Track2/NG_1154/level_labels}"
SEV_SPLIT_FILE="${SEV_SPLIT_FILE:-severity/splits/split.json}"
SEV_SAVE_DIR="${SEV_SAVE_DIR:-severity/results}"

DEVICE="${DEVICE:-0}"

echo "==> Training YOLO segmentation model"
python baseline/train_model.py \
  --weights "${YOLO_PRETRAIN}" \
  --data "${YOLO_DATA_YAML}" \
  --epochs 300 \
  --imgsz 512 \
  --batch 40 \
  --device "${DEVICE}" \
  --project "${YOLO_OUTPUT_DIR}" \
  --name "yolov8x-seg-split6"

echo "==> Training MAOL severity grader"
python severity/scripts/train_severity_baseline_ce.py \
  --img_dir "${SEV_IMG_DIR}" \
  --label_dir "${SEV_LABEL_DIR}" \
  --split_file "${SEV_SPLIT_FILE}" \
  --head_type coral \
  --use_morphology true \
  --use_class_embedding true \
  --use_adaptive_thresholds true \
  --use_pred_aware_roi true \
  --epochs 350 \
  --batch_size 64 \
  --lr 1e-3 \
  --device "${DEVICE}" \
  --save_dir "${SEV_SAVE_DIR}"

echo "Training completed."
