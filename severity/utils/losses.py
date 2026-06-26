"""
severity/utils/losses.py

Loss functions for ordinal severity classification.

E2: distance_aware_ce  — soft ordinal targets via exp(-alpha*|c-y|)
E3: coral_loss         — CORAL ordinal head (K-1 binary tasks)

Classes are ordered: Acceptable(0) < Marginal NG(1) < NG(2) < Gross NG(3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_distance_soft_targets(
    labels: torch.Tensor,
    num_classes: int = 4,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    For each sample with true label y, build a soft target distribution where
    class c gets weight proportional to exp(-alpha * |c - y|), then normalized.

    Args:
        labels:      (N,) integer tensor of ground-truth class indices
        num_classes: number of ordinal classes
        alpha:       decay rate; larger alpha → sharper peak at true class

    Returns:
        soft_targets: (N, num_classes) float tensor, each row sums to 1
    """
    device = labels.device
    # class indices: [0, 1, ..., num_classes-1]
    cls_idx = torch.arange(num_classes, dtype=torch.float32, device=device)  # (C,)
    # labels expanded: (N, 1) - cls_idx: (C,) → distances: (N, C)
    distances = torch.abs(labels.float().unsqueeze(1) - cls_idx.unsqueeze(0))  # (N, C)
    weights = torch.exp(-alpha * distances)  # (N, C)
    soft_targets = weights / weights.sum(dim=1, keepdim=True)  # normalize
    return soft_targets


def distance_aware_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
    num_classes: int = 4,
) -> torch.Tensor:
    """
    Distance-aware CrossEntropy with soft ordinal targets.

    loss = -sum_c [ soft_target(c) * log_softmax(logits)[c] ]

    Args:
        logits:      (N, num_classes) raw model outputs
        labels:      (N,) integer ground-truth class indices
        alpha:       ordinal decay rate for soft targets
        num_classes: number of classes

    Returns:
        scalar mean loss over the batch
    """
    soft_targets = build_distance_soft_targets(labels, num_classes=num_classes, alpha=alpha)
    log_probs = F.log_softmax(logits, dim=1)  # (N, C)
    # cross-entropy with soft targets: -sum(soft_target * log_prob)
    loss = -(soft_targets * log_probs).sum(dim=1)  # (N,)
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# CORAL ordinal loss (E3)
# ─────────────────────────────────────────────────────────────────────────────

def levels_from_label(labels: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """
    Convert integer class labels to CORAL binary level targets.

    For K classes, produce K-1 binary tasks:
        label 0 → [0, 0, 0]
        label 1 → [1, 0, 0]
        label 2 → [1, 1, 0]
        label 3 → [1, 1, 1]

    Args:
        labels:      (N,) integer tensor in [0, num_classes-1]
        num_classes: total number of ordinal classes K

    Returns:
        levels: (N, K-1) float tensor of 0/1
    """
    K = num_classes
    # threshold indices: 0, 1, ..., K-2
    thresholds = torch.arange(K - 1, device=labels.device).unsqueeze(0)  # (1, K-1)
    # label >= threshold+1  ↔  label > threshold
    levels = (labels.unsqueeze(1) > thresholds).float()  # (N, K-1)
    return levels


def coral_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 4,
) -> torch.Tensor:
    """
    CORAL loss: sum of K-1 binary cross-entropies.

    Args:
        logits:      (N, K-1) raw outputs from CoralHead
        labels:      (N,) integer ground-truth class indices
        num_classes: K

    Returns:
        scalar mean loss
    """
    levels = levels_from_label(labels, num_classes=num_classes)  # (N, K-1)
    # BCEWithLogitsLoss averaged over tasks and samples
    loss = F.binary_cross_entropy_with_logits(logits, levels, reduction="mean")
    return loss


def coral_predict(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Recover ordinal class predictions from CORAL logits.

    Predicted class = number of thresholds exceeded (sigmoid > threshold).

    Args:
        logits:    (N, K-1) raw CORAL outputs
        threshold: decision boundary (default 0.5)

    Returns:
        preds: (N,) integer tensor in [0, K-1]
    """
    probs = torch.sigmoid(logits)           # (N, K-1)
    preds = (probs > threshold).sum(dim=1)  # (N,)
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# CORN ordinal loss & predict (E7)
# ─────────────────────────────────────────────────────────────────────────────
#
# Key difference from CORAL:
#   CORAL: K-1 tasks share one linear weight; all samples used for every task.
#   CORN:  K-1 tasks have independent weights; each task k only trains on the
#          conditional subset {y >= k}, learning P(y > k | y >= k).
#          The unconditional rank probabilities are recovered via the chain rule:
#            P(y > k) = P(y>0) * P(y>1|y>0) * ... * P(y>k|y>=k)
#

def corn_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 4,
) -> torch.Tensor:
    """
    CORN loss: sum of K-1 conditional binary cross-entropies.

    For task k (k = 0 .. K-2):
      - Conditional subset: samples where label >= k  (i.e. y >= k)
      - Binary target:      1 if label > k, else 0
      - Loss:               BCEWithLogits on logits[:, k] for that subset

    If a task has no valid samples in the current batch, it is safely skipped.

    Args:
        logits:      (N, K-1) raw model outputs — one logit per ordinal task
        labels:      (N,) integer ground-truth class indices in [0, K-1]
        num_classes: K

    Returns:
        scalar mean loss (averaged over the K-1 tasks that had valid samples)
    """
    num_tasks = num_classes - 1
    total_loss = torch.zeros(1, device=logits.device, dtype=logits.dtype)
    valid_tasks = 0

    for k in range(num_tasks):
        # conditional subset: only samples with label >= k
        mask = labels >= k                          # (N,) bool
        if mask.sum() == 0:
            continue                                # no valid samples, skip safely

        logit_k = logits[mask, k]                  # (M,)
        # binary target: 1 if label > k (i.e. label >= k+1), else 0
        target_k = (labels[mask] > k).float()      # (M,)

        total_loss = total_loss + F.binary_cross_entropy_with_logits(
            logit_k, target_k, reduction="mean"
        )
        valid_tasks += 1

    if valid_tasks == 0:
        return total_loss.squeeze()

    return (total_loss / valid_tasks).squeeze()


def corn_predict(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Recover ordinal class predictions from CORN logits.

    Steps:
      1. sigmoid(logits) -> conditional probabilities q  (N, K-1)
         q[:, k] ≈ P(y > k | y >= k)
      2. cumulative product -> unconditional rank probabilities p
         p[:, 0] = q[:, 0]                    = P(y > 0)
         p[:, 1] = q[:, 0] * q[:, 1]          = P(y > 1)
         p[:, 2] = q[:, 0] * q[:, 1] * q[:, 2] = P(y > 2)
      3. threshold each p[:, k] at 0.5 -> binary indicators
      4. predicted class = number of thresholds exceeded (sum over K-1 tasks)

    This is the standard CORN decoding via the chain rule for conditional
    probabilities, which guarantees rank consistency.

    Args:
        logits:    (N, K-1) raw CORN outputs
        threshold: decision boundary (default 0.5)

    Returns:
        preds: (N,) integer tensor in [0, K-1]
    """
    q = torch.sigmoid(logits)                      # (N, K-1) conditional probs
    p = torch.cumprod(q, dim=1)                    # (N, K-1) unconditional probs
    preds = (p > threshold).sum(dim=1)             # (N,)
    return preds
