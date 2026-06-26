# MAOL

**MAOL: Morphology-Aware Ordinal Learning for Fine-Grained Industrial Defect Severity Grading** 的官方代码实现。

本仓库包含训练、推理和评估代码。数据集与模型权重不随仓库发布，请按 `README.md` 中的目录结构自行准备，或通过命令行参数传入本地路径。

主要流程：

```bash
bash train.sh
bash infer.sh
bash eval.sh
```

常用命令：

```bash
python severity/scripts/train_severity_baseline_ce.py \
  --img_dir data/Track2/NG_1154/images \
  --label_dir data/Track2/NG_1154/level_labels \
  --split_file severity/splits/split.json \
  --head_type coral \
  --use_morphology true \
  --use_class_embedding true \
  --use_adaptive_thresholds true \
  --use_pred_aware_roi true
```

推理：

```bash
python grade_severity.py \
  --method E6 \
  --labels_dir result/predict_test/labels \
  --img_dir data/test/Track2_TestData_Fine-Grained-Severity-Grading \
  --checkpoint checkpoints/severity_E6_best.pth \
  --output_dir result/grading_E6
```

许可证见 `LICENSE`。
