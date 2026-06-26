#!/usr/bin/env bash
set -euo pipefail

# End-to-end inference entry point for MAOL.
# Provide trained checkpoints locally; this repository does not include weights.

TEST_IMG_DIR="${TEST_IMG_DIR:-data/test/Track2_TestData_Fine-Grained-Severity-Grading}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-checkpoints/yolo_best.pt}"
SEV_CHECKPOINT="${SEV_CHECKPOINT:-checkpoints/severity_E6_best.pth}"

YOLO_PRED_DIR="${YOLO_PRED_DIR:-result/predict_test}"
GRADING_DIR="${GRADING_DIR:-result/grading_E6}"
SUBMISSION_OUTPUT="${SUBMISSION_OUTPUT:-result/Track2_submission.json}"
IMG_SEARCH_DIR="${IMG_SEARCH_DIR:-${TEST_IMG_DIR}}"

CONF_THRESH="${CONF_THRESH:-0.1}"
IMGSZ="${IMGSZ:-512}"
DEVICE="${DEVICE:-0}"

echo "==> Running YOLO segmentation"
python baseline/test_model.py \
  --weights "${YOLO_WEIGHTS}" \
  --source "${TEST_IMG_DIR}" \
  --output "${YOLO_PRED_DIR}" \
  --conf "${CONF_THRESH}" \
  --imgsz "${IMGSZ}" \
  --device "${DEVICE}"

echo "==> Running MAOL severity grading"
python grade_severity.py \
  --method E6 \
  --labels_dir "${YOLO_PRED_DIR}/labels" \
  --img_dir "${TEST_IMG_DIR}" \
  --checkpoint "${SEV_CHECKPOINT}" \
  --output_dir "${GRADING_DIR}"

echo "==> Converting to submission JSON"
python convert_e8_to_submission.py \
  --input_dir "${GRADING_DIR}" \
  --output "${SUBMISSION_OUTPUT}" \
  --img_search "${IMG_SEARCH_DIR}"

echo "Inference completed: ${SUBMISSION_OUTPUT}"
