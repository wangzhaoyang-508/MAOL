"""
severity/models/severity_classifier.py

Severity classifier backbones + heads.

head_type="ce"    -> standard 4-class softmax head  (E1/E2)
head_type="coral" -> CORAL ordinal head, K-1 logits (E3/E4/E5/E6)

use_morphology=True         adds learnable MorphologyEncoder branch (E4)
use_class_embedding=True    adds learnable class embedding branch   (E5)
use_adaptive_thresholds=True  class-conditional ordinal threshold offsets (E6)

E6 design (class-conditional adaptive thresholds):
  logits = severity_score(h) + global_bias + delta_bias(class_id)
  where:
    severity_score: Linear(fused_dim, 1, bias=False)  -- sample-level severity
    global_bias:    nn.Parameter([K-1])               -- shared ordinal thresholds
    delta_bias:     Embedding + MLP -> [K-1]          -- per-class threshold offset
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# lightweight CNN backbone (kept for completeness)
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SeverityCNN(nn.Module):
    def __init__(self, num_classes=4, in_channels=3, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnRelu(in_channels, 32), nn.MaxPool2d(2),
            ConvBnRelu(32, 64),          nn.MaxPool2d(2),
            ConvBnRelu(64, 128),         nn.MaxPool2d(2),
            ConvBnRelu(128, 256),        nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# CORAL ordinal head
# ---------------------------------------------------------------------------

class CoralHead(nn.Module):
    """
    Shared-weight linear + K-1 learnable biases.
    Input:  (B, in_dim)
    Output: (B, K-1) logits
    """
    def __init__(self, in_dim: int, num_classes: int = 4):
        super().__init__()
        self.fc   = nn.Linear(in_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        return self.fc(x) + self.bias.unsqueeze(0)   # (B, K-1)


# ---------------------------------------------------------------------------
# Adaptive CORAL head  (E6)
# ---------------------------------------------------------------------------

class AdaptiveCoralHead(nn.Module):
    """
    Class-conditional adaptive ordinal thresholds for E6.

    logits = severity_score(h) + global_bias + delta_bias(class_id)

    - severity_score: Linear(fused_dim, 1, bias=False)
        captures sample-level severity from fused features
    - global_bias: nn.Parameter([K-1])
        shared ordinal thresholds across all classes
    - delta_bias: Embedding(num_defect_classes, emb_dim) + MLP -> [K-1]
        per-class threshold offset; lets each defect type shift its
        decision boundaries independently

    Input:  h (B, fused_dim),  class_id (B,) long
    Output: logits (B, K-1)
    """
    def __init__(
        self,
        fused_dim: int,
        num_classes: int = 4,
        num_defect_classes: int = 11,
        class_emb_dim: int = 16,
        threshold_hidden_dim: int = 32,
    ):
        super().__init__()
        num_thresholds = num_classes - 1

        # sample-level severity score (no bias; bias is handled by global_bias)
        self.severity_score = nn.Linear(fused_dim, 1, bias=False)

        # global shared ordinal thresholds (equivalent to CoralHead.bias)
        self.global_bias = nn.Parameter(torch.zeros(num_thresholds))

        # class-conditional threshold offset
        self.threshold_emb = nn.Embedding(num_defect_classes, class_emb_dim)
        self.threshold_mlp = nn.Sequential(
            nn.Linear(class_emb_dim, threshold_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(threshold_hidden_dim, num_thresholds),
        )
        # init MLP output near zero so E6 starts close to standard CORAL
        nn.init.zeros_(self.threshold_mlp[-1].weight)
        nn.init.zeros_(self.threshold_mlp[-1].bias)

    def forward(self, h: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:
        """
        h:        (B, fused_dim)
        class_id: (B,) long
        returns:  (B, K-1) logits
        """
        score     = self.severity_score(h)                    # (B, 1)
        delta     = self.threshold_mlp(self.threshold_emb(class_id))  # (B, K-1)
        return score + self.global_bias.unsqueeze(0) + delta  # (B, K-1)

    # ── analysis helper (optional, call during validation) ────────────────
    @torch.no_grad()
    def get_threshold_analysis(self, num_defect_classes: int = None) -> dict:
        """
        Return global_bias and per-class mean delta_bias for analysis.
        Useful for verifying that different defect classes learn distinct thresholds.
        """
        n = num_defect_classes or self.threshold_emb.num_embeddings
        ids = torch.arange(n, device=self.global_bias.device)
        delta_all = self.threshold_mlp(self.threshold_emb(ids))  # (n, K-1)
        return {
            "global_bias":       self.global_bias.cpu().tolist(),
            "per_class_delta":   delta_all.cpu().tolist(),   # list of [K-1] per class
        }


# ---------------------------------------------------------------------------
# Morphology encoder  (E4)
# ---------------------------------------------------------------------------

class MorphologyEncoder(nn.Module):
    """
    Small MLP that maps raw morphology features to an embedding.

    Design goal: learnable fusion of morphology priors with ROI features,
    replacing the fixed-weight rule-based scoring in the official baseline.

    Input:  (B, morph_dim)  -- z-score normalised
    Output: (B, hidden_dim)
    """
    def __init__(self, morph_dim: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(morph_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Main transfer model  (E1 / E2 / E3 / E4 / E5 / E6)
# ---------------------------------------------------------------------------

class SeverityTransferModel(nn.Module):
    """
    ResNet-18 / MobileNetV2 backbone with switchable head and optional
    morphology + class-embedding branches.

    use_morphology=False, use_class_embedding=False                    ->  E1/E2/E3
    use_morphology=True,  use_class_embedding=False                    ->  E4
    use_morphology=True,  use_class_embedding=True                     ->  E5
    use_morphology=True,  use_class_embedding=True,
      use_adaptive_thresholds=True                                     ->  E6

    E6 note: use_class_embedding=True is required so that the fused feature
    already contains class context; use_adaptive_thresholds then replaces
    the standard CoralHead with AdaptiveCoralHead for decision-level adaptation.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 4,
        pretrained: bool = False,
        dropout: float = 0.3,
        head_type: str = "ce",
        use_morphology: bool = False,
        morph_dim: int = 7,
        morph_hidden_dim: int = 32,
        use_class_embedding: bool = False,
        num_defect_classes: int = 11,
        class_emb_dim: int = 16,
        # E6
        use_adaptive_thresholds: bool = False,
        threshold_hidden_dim: int = 32,
    ):
        super().__init__()
        self.head_type               = head_type
        self.use_morphology          = use_morphology
        self.use_class_embedding     = use_class_embedding
        self.use_adaptive_thresholds = use_adaptive_thresholds

        if use_adaptive_thresholds and head_type != "coral":
            raise ValueError("use_adaptive_thresholds requires head_type='coral'")
        if use_adaptive_thresholds and not use_class_embedding:
            raise ValueError("use_adaptive_thresholds requires use_class_embedding=True "
                             "(class_id must be available at forward time)")

        # ── backbone ──────────────────────────────────────────────────────
        if backbone == "resnet18":
            net = tvm.resnet18(
                weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
            net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            roi_feat_dim = net.fc.in_features   # 512
            net.fc = nn.Identity()
            self.net = net
        elif backbone == "mobilenet_v2":
            net = tvm.mobilenet_v2(
                weights=tvm.MobileNet_V2_Weights.DEFAULT if pretrained else None)
            net.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
            roi_feat_dim = net.classifier[1].in_features   # 1280
            net.classifier = nn.Identity()
            self.net = net
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.dropout = nn.Dropout(dropout)

        # ── morphology branch (E4) ────────────────────────────────────────
        if use_morphology:
            self.morph_enc = MorphologyEncoder(morph_dim, morph_hidden_dim)

        # ── class embedding branch (E5 / E6) ─────────────────────────────
        if use_class_embedding:
            self.class_embedding = nn.Embedding(num_defect_classes, class_emb_dim)

        # ── fusion layer (dim depends on active branches) ─────────────────
        fused_dim = roi_feat_dim
        if use_morphology:
            fused_dim += morph_hidden_dim
        if use_class_embedding:
            fused_dim += class_emb_dim

        if fused_dim != roi_feat_dim:
            self.fusion = nn.Sequential(
                nn.Linear(fused_dim, roi_feat_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        else:
            self.fusion = None   # no extra branches, skip fusion

        # ── output head ───────────────────────────────────────────────────
        if use_adaptive_thresholds:
            # E6: AdaptiveCoralHead uses class_id for decision-level threshold adaptation.
            # Note: class_id is already fused into roi_feat_dim via class_embedding + fusion,
            # but AdaptiveCoralHead uses a *separate* threshold embedding so that the
            # feature-level and threshold-level class signals are decoupled.
            self.head = AdaptiveCoralHead(
                fused_dim=roi_feat_dim,
                num_classes=num_classes,
                num_defect_classes=num_defect_classes,
                class_emb_dim=class_emb_dim,
                threshold_hidden_dim=threshold_hidden_dim,
            )
        elif head_type == "ce":
            self.head = nn.Linear(roi_feat_dim, num_classes)
        elif head_type == "coral":
            self.head = CoralHead(roi_feat_dim, num_classes=num_classes)
        elif head_type == "corn":
            # E7: plain Linear head — CORN does NOT share weights across tasks.
            # Each of the K-1 output logits is an independent ordinal binary classifier.
            self.head = nn.Linear(roi_feat_dim, num_classes - 1)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x, morph=None, class_id=None):
        """
        x:        (B, 3, H, W)
        morph:    (B, morph_dim)   [E4/E5/E6]
        class_id: (B,) long        [E5/E6]

        E6 requires class_id; raises RuntimeError if missing.
        """
        if self.use_adaptive_thresholds and class_id is None:
            raise RuntimeError(
                "E6 (use_adaptive_thresholds=True) requires class_id in forward(). "
                "Make sure your DataLoader returns cls_id and it is passed here."
            )

        feat = self.dropout(self.net(x))   # (B, roi_feat_dim)

        parts = [feat]
        if self.use_morphology:
            assert morph is not None, "morph required when use_morphology=True"
            parts.append(self.morph_enc(morph))
        if self.use_class_embedding:
            assert class_id is not None, "class_id required when use_class_embedding=True"
            parts.append(self.class_embedding(class_id))

        if len(parts) > 1:
            feat = self.fusion(torch.cat(parts, dim=1))

        # E6: pass class_id to AdaptiveCoralHead for threshold adaptation
        if self.use_adaptive_thresholds:
            return self.head(feat, class_id)

        return self.head(feat)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def build_model(
    arch: str = "cnn",
    num_classes: int = 4,
    pretrained: bool = False,
    dropout: float = 0.3,
    head_type: str = "ce",
    use_morphology: bool = False,
    morph_dim: int = 7,
    morph_hidden_dim: int = 32,
    use_class_embedding: bool = False,
    num_defect_classes: int = 11,
    class_emb_dim: int = 16,
    # E6
    use_adaptive_thresholds: bool = False,
    threshold_hidden_dim: int = 32,
) -> nn.Module:
    if arch == "cnn":
        return SeverityCNN(num_classes=num_classes, dropout=dropout)
    return SeverityTransferModel(
        backbone=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        head_type=head_type,
        use_morphology=use_morphology,
        morph_dim=morph_dim,
        morph_hidden_dim=morph_hidden_dim,
        use_class_embedding=use_class_embedding,
        num_defect_classes=num_defect_classes,
        class_emb_dim=class_emb_dim,
        use_adaptive_thresholds=use_adaptive_thresholds,
        threshold_hidden_dim=threshold_hidden_dim,
    )
