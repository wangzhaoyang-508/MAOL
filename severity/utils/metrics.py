"""
severity/utils/metrics.py

评估指标：
- per-class accuracy / macro accuracy
- weighted kappa（与 caculate_metric.py 中 severity_grading_from_confmat 对齐）
- 混淆矩阵打印
"""

import numpy as np
from collections import defaultdict
from typing import List

from severity.utils.grade_schema import GRADE_NAMES


def confusion_matrix(labels: List[int], preds: List[int], num_classes: int = 4) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=int)
    for g, p in zip(labels, preds):
        if 0 <= g < num_classes and 0 <= p < num_classes:
            mat[g, p] += 1
    return mat


def weighted_kappa(conf_mat: np.ndarray) -> float:
    """
    二次加权 kappa，与 baseline/caculate_metric.py::severity_grading_from_confmat 完全一致。
    conf_mat[i, j] = GT=i, Pred=j 的样本数
    """
    conf_mat = np.asarray(conf_mat, dtype=float)
    K = conf_mat.shape[0]
    N = conf_mat.sum()
    if N == 0:
        return float("nan")

    idx = np.arange(K)
    I, J = np.meshgrid(idx, idx, indexing="ij")
    W = ((I - J) ** 2) / float((K - 1) ** 2)

    num = N * np.sum(W * conf_mat)
    n_i = conf_mat.sum(axis=1)
    n_j = conf_mat.sum(axis=0)
    expected = np.outer(n_i, n_j)
    denom = np.sum(W * expected)
    if denom == 0:
        return 0.0
    return float(1.0 - num / denom)


def compute_metrics(labels: List[int], preds: List[int], num_classes: int = 4) -> dict:
    """
    返回:
        overall_acc, per_class_acc (list), macro_acc,
        weighted_kappa, confusion_matrix
    """
    mat = confusion_matrix(labels, preds, num_classes)
    total = mat.sum()
    overall_acc = float(mat.diagonal().sum()) / total if total > 0 else 0.0

    per_class_acc = []
    for c in range(num_classes):
        row_sum = mat[c].sum()
        per_class_acc.append(float(mat[c, c]) / row_sum if row_sum > 0 else 0.0)

    macro_acc = float(np.mean(per_class_acc))
    kappa = weighted_kappa(mat)

    return {
        "overall_acc": overall_acc,
        "per_class_acc": per_class_acc,
        "macro_acc": macro_acc,
        "weighted_kappa": kappa,
        "confusion_matrix": mat,
    }


def print_metrics(metrics: dict, prefix: str = ""):
    tag = f"[{prefix}] " if prefix else ""
    print(f"{tag}Overall Acc : {metrics['overall_acc']:.4f}")
    print(f"{tag}Macro   Acc : {metrics['macro_acc']:.4f}")
    print(f"{tag}Weighted κ  : {metrics['weighted_kappa']:.4f}")
    print(f"{tag}Per-class   :")
    for i, (name, acc) in enumerate(zip(GRADE_NAMES, metrics["per_class_acc"])):
        print(f"    {name:<14}: {acc:.4f}")
    print(f"{tag}Confusion Matrix (rows=GT, cols=Pred):")
    header = "         " + "  ".join(f"{n[:6]:>6}" for n in GRADE_NAMES)
    print(header)
    for i, row in enumerate(metrics["confusion_matrix"]):
        row_str = "  ".join(f"{v:>6}" for v in row)
        print(f"  {GRADE_NAMES[i][:6]:>6}  {row_str}")
