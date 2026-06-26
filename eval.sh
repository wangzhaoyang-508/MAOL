#!/usr/bin/env bash
set -euo pipefail

# Formal predicted-instance evaluation for a validation split.

IMG_DIR="${IMG_DIR:-data/Track2/NG_1154/images}"
LABEL_DIR="${LABEL_DIR:-data/Track2/NG_1154/level_labels}"
SPLIT_FILE="${SPLIT_FILE:-severity/splits/split.json}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-checkpoints/yolo_best.pt}"
SEV_CHECKPOINT="${SEV_CHECKPOINT:-checkpoints/severity_E6_best.pth}"

VAL_PRED_DIR="${VAL_PRED_DIR:-result/labels_val}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-severity/results_predicted/formal_E6}"

IMGSZ="${IMGSZ:-512}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.45}"
DEVICE="${DEVICE:-0}"
IOU_THRESH="${IOU_THRESH:-0.5}"

echo "==> Predicting validation instances"
python severity/scripts/predict_yolo_val.py \
  --weights "${YOLO_WEIGHTS}" \
  --split_file "${SPLIT_FILE}" \
  --img_dir "${IMG_DIR}" \
  --output_dir "${VAL_PRED_DIR}" \
  --imgsz "${IMGSZ}" \
  --conf "${CONF}" \
  --iou "${IOU}" \
  --device "${DEVICE}"

echo "==> Evaluating severity on matched predicted instances"
python severity/scripts/eval_predicted_instances.py \
  --split_file "${SPLIT_FILE}" \
  --img_dir "${IMG_DIR}" \
  --label_dir "${LABEL_DIR}" \
  --prediction_dir "${VAL_PRED_DIR}" \
  --checkpoint_path "${SEV_CHECKPOINT}" \
  --output_dir "${EVAL_OUTPUT_DIR}" \
  --iou_thresh "${IOU_THRESH}" \
  --eval_mode formal

echo "Evaluation completed: ${EVAL_OUTPUT_DIR}"
