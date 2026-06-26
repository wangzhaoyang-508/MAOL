"""
severity/scripts/train_severity_baseline_ce.py

Track2 severity baseline: E1 / E2 / E3 / E4 / E5 / E6 / E7 / E8 / E9

E1: plain CE                                  --loss_type ce
E2: distance-aware CE                         --loss_type distance_ce
E3: CORAL ordinal head                        --head_type coral
E4: CORAL + morphology features               --head_type coral --use_morphology true
E5: CORAL + morphology + class embedding      --head_type coral --use_morphology true --use_class_embedding true
E6: CORAL + morphology + adaptive thresholds  --head_type coral --use_morphology true --use_class_embedding true --use_adaptive_thresholds true
E7: CORN ordinal head (pure baseline)         --head_type corn
E8: E4 + prediction-aware ROI perturbation    --head_type coral --use_morphology true --use_pred_aware_roi true
E9-min:    E6 + pred-aware ROI perturbation   --head_type coral --use_morphology true --use_class_embedding true --use_adaptive_thresholds true --use_pred_aware_roi true --use_morphology_aware_perturbation false
"""

import argparse
import json
import os
import sys
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from severity.datasets.severity_roi_dataset import (
    SeverityROIDataset, collect_instances, split_by_image,
    fit_morph_stats, GRADE_NAMES, MORPH_DIM,
)
from severity.models.severity_classifier import build_model
from severity.utils.metrics import compute_metrics, print_metrics
from severity.utils.losses import distance_aware_ce, coral_loss, coral_predict, corn_loss, corn_predict

DEFAULT_IMG_DIR   = "Track2/NG_1154/images"
DEFAULT_LABEL_DIR = "Track2/NG_1154/level_labels"
DEFAULT_SAVE_DIR  = "severity/results"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_data_stats(all_inst, train_inst, val_inst):
    print("\n" + "=" * 60)
    print(f"  total={len(all_inst)}  train={len(train_inst)}  val={len(val_inst)}")
    for tag, lst in [("Train", train_inst), ("Val", val_inst)]:
        cnt = Counter(i["grade_id"] for i in lst)
        print(f"  [{tag}] " + "  ".join(f"{GRADE_NAMES[g]}={cnt.get(g,0)}" for g in range(4)))
    print("=" * 60 + "\n")


def visualize_roi_samples(all_inst, save_dir, n_per_grade=4, roi_size=64, pad_ratio=0.2):
    from PIL import Image, ImageDraw
    from severity.datasets.severity_roi_dataset import expand_bbox
    os.makedirs(save_dir, exist_ok=True)
    grade_insts = {g: [] for g in range(4)}
    for inst in all_inst:
        grade_insts[inst["grade_id"]].append(inst)
    cell, margin, lh = roi_size, 4, 14
    canvas_w = n_per_grade * (cell + margin) + margin
    canvas_h = 4 * (cell + margin + lh) + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    rng = random.Random(0)
    for row, gid in enumerate(range(4)):
        samples = rng.sample(grade_insts[gid], min(n_per_grade, len(grade_insts[gid])))
        for col, inst in enumerate(samples):
            try:
                img = Image.open(inst["img_path"]).convert("RGB")
                x1, y1, x2, y2 = expand_bbox(*inst["bbox"], inst["W"], inst["H"], pad_ratio)
                roi = img.crop((x1, y1, x2, y2)).resize((cell, cell))
            except Exception:
                roi = Image.new("RGB", (cell, cell), (200, 200, 200))
            px = margin + col * (cell + margin)
            py = margin + row * (cell + margin + lh) + lh
            canvas.paste(roi, (px, py))
            draw.text((px, py - lh), GRADE_NAMES[gid], fill=(50, 50, 200))
    out = os.path.join(save_dir, "roi_samples.png")
    canvas.save(out)
    print(f"  roi samples saved: {out}")


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def make_criterion(args):
    if args.head_type == "coral":
        return lambda logits, labels: coral_loss(logits, labels, num_classes=4)
    if args.head_type == "corn":
        return lambda logits, labels: corn_loss(logits, labels, num_classes=4)
    if args.loss_type == "distance_ce":
        a = args.distance_alpha
        return lambda logits, labels: distance_aware_ce(logits, labels, alpha=a, num_classes=4)
    ce = nn.CrossEntropyLoss()
    return lambda logits, labels: ce(logits, labels)


# ---------------------------------------------------------------------------
# train / eval  (morph-aware)
# ---------------------------------------------------------------------------

def _forward(model, batch, device, use_morphology, use_class_embedding, head_type,
             use_adaptive_thresholds=False):
    """Unpack batch and run forward pass. Returns (logits, labels)."""
    # E6: adaptive thresholds also needs class_id (same batch format as E5)
    if use_morphology and (use_class_embedding or use_adaptive_thresholds):
        imgs, morph, cls_id, labels = batch
        imgs, morph, cls_id, labels = (imgs.to(device), morph.to(device),
                                       cls_id.to(device), labels.to(device))
        logits = model(imgs, morph=morph, class_id=cls_id)
    elif use_morphology:
        imgs, morph, labels = batch
        imgs, morph, labels = imgs.to(device), morph.to(device), labels.to(device)
        logits = model(imgs, morph=morph)
    else:
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
    return logits, labels


def _ordinal_predict(head_type, logits):
    """Dispatch to the correct ordinal predict function."""
    if head_type == "coral":
        return coral_predict(logits)
    if head_type == "corn":
        return corn_predict(logits)
    return logits.argmax(1)


def train_one_epoch(model, loader, optimizer, criterion, device,
                    head_type, use_morphology, use_class_embedding,
                    use_adaptive_thresholds=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        logits, labels = _forward(model, batch, device,
                                  use_morphology, use_class_embedding, head_type,
                                  use_adaptive_thresholds)
        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        preds = _ordinal_predict(head_type, logits)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, head_type,
             use_morphology, use_class_embedding,
             use_adaptive_thresholds=False):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in loader:
        logits, labels = _forward(model, batch, device,
                                  use_morphology, use_class_embedding, head_type,
                                  use_adaptive_thresholds)
        total_loss += criterion(logits, labels).item() * len(labels)
        preds = _ordinal_predict(head_type, logits)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    metrics = compute_metrics(all_labels, all_preds)
    metrics["macro_f1"] = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    metrics["all_preds"] = all_preds
    metrics["all_labels"] = all_labels
    return total_loss / len(all_labels), metrics


# ---------------------------------------------------------------------------
# E8: debug visualization of clean vs perturbed ROI
# ---------------------------------------------------------------------------

def visualize_perturbation_samples(
    instances, save_dir, args,
    n_samples=8, roi_size=64, pad_ratio=0.2, seed=0,
):
    """
    Save a side-by-side comparison of clean ROI vs perturbed ROI for n_samples.
    Only called when use_pred_aware_roi=True.
    """
    from severity.datasets.severity_roi_dataset import perturb_bbox, expand_bbox
    import random as _random

    os.makedirs(save_dir, exist_ok=True)
    rng = _random.Random(seed)
    samples = rng.sample(instances, min(n_samples, len(instances)))

    cell = roi_size * 2   # display at 2x for clarity
    margin = 4
    label_h = 14
    canvas_w = 2 * (cell + margin) + margin
    canvas_h = len(samples) * (cell + margin + label_h) + margin

    from PIL import Image as _Image, ImageDraw as _Draw
    canvas = _Image.new("RGB", (canvas_w, canvas_h), (230, 230, 230))
    draw = _Draw.Draw(canvas)

    for row, inst in enumerate(samples):
        img = _Image.open(inst["img_path"]).convert("RGB")
        W, H = inst["W"], inst["H"]
        x1, y1, x2, y2 = inst["bbox"]

        # clean crop
        ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2, W, H, pad_ratio)
        clean_roi = img.crop((ex1, ey1, ex2, ey2)).resize((cell, cell))

        # perturbed crop
        pb, iou, fallback = perturb_bbox(
            x1, y1, x2, y2, W, H, rng,
            iou_min=args.roi_noise_iou_min,
            iou_max=args.roi_noise_iou_max,
            max_trials=args.roi_noise_max_trials,
            min_box_size=args.roi_noise_min_box_size,
        )
        px1, py1, px2, py2 = expand_bbox(*pb, W, H, pad_ratio)
        noisy_roi = img.crop((px1, py1, px2, py2)).resize((cell, cell))

        py_off = margin + row * (cell + margin + label_h) + label_h
        canvas.paste(clean_roi, (margin, py_off))
        canvas.paste(noisy_roi, (margin + cell + margin, py_off))
        draw.text((margin, py_off - label_h),
                  f"clean  grade={GRADE_NAMES[inst['grade_id']]}", fill=(30, 30, 180))
        draw.text((margin + cell + margin, py_off - label_h),
                  f"noisy  iou={iou:.2f}{'(fb)' if fallback else ''}", fill=(180, 30, 30))

    out = os.path.join(save_dir, "perturbation_samples.png")
    canvas.save(out)
    print(f"  [E8] perturbation debug image saved: {out}")


# ---------------------------------------------------------------------------
# E9-strict: morphology perturbation debug
# ---------------------------------------------------------------------------

def _debug_morph_perturbation(instances, save_dir, args, n_samples=8, seed=0):
    """
    E9-strict debug: print and save a CSV showing how gray_contrast changes
    when context_bbox switches from GT bbox to perturbed bbox.

    Columns: img_stem, gt_gray_contrast, perturbed_gray_contrast, delta, iou
    """
    from severity.datasets.severity_roi_dataset import perturb_bbox, compute_morph_features
    import random as _random

    os.makedirs(save_dir, exist_ok=True)
    rng = _random.Random(seed)
    samples = rng.sample(instances, min(n_samples, len(instances)))

    rows = []
    print(f"\n  [E9-strict] morphology perturbation debug ({len(samples)} samples):")
    print(f"  {'stem':40s}  {'gt_contrast':>12}  {'pert_contrast':>13}  {'delta':>8}  {'iou':>6}")
    for inst in samples:
        from PIL import Image as _Image
        img_gray = np.array(_Image.open(inst["img_path"]).convert("L"), dtype=np.uint8)
        W, H = inst["W"], inst["H"]
        x1, y1, x2, y2 = inst["bbox"]

        gt_feat = compute_morph_features(inst["points_norm"], W, H, img_gray)
        gt_contrast = float(gt_feat[6])

        pb, iou, fallback = perturb_bbox(
            x1, y1, x2, y2, W, H, rng,
            iou_min=args.roi_noise_iou_min,
            iou_max=args.roi_noise_iou_max,
            max_trials=args.roi_noise_max_trials,
            min_box_size=args.roi_noise_min_box_size,
        )
        pert_feat = compute_morph_features(
            inst["points_norm"], W, H, img_gray, context_bbox=pb
        )
        pert_contrast = float(pert_feat[6])
        delta = pert_contrast - gt_contrast

        stem_short = inst["img_stem"][-40:]
        print(f"  {stem_short:40s}  {gt_contrast:12.4f}  {pert_contrast:13.4f}  "
              f"{delta:+8.4f}  {iou:.3f}{'(fb)' if fallback else ''}")
        rows.append({
            "img_stem":           inst["img_stem"],
            "gt_gray_contrast":   round(gt_contrast, 6),
            "pert_gray_contrast": round(pert_contrast, 6),
            "delta":              round(delta, 6),
            "iou":                round(iou, 4),
            "fallback":           fallback,
        })

    out_csv = os.path.join(save_dir, "e9_strict_morph_debug.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  [E9-strict] morph debug saved: {out_csv}\n")


# ---------------------------------------------------------------------------
# save results
# ---------------------------------------------------------------------------

def save_results(save_dir, log_rows, metrics, val_inst, args, morph_stats=None):
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(os.path.join(save_dir, "train_log.csv"), index=False)
    pd.DataFrame(metrics["confusion_matrix"],
                 index=GRADE_NAMES, columns=GRADE_NAMES).to_csv(
        os.path.join(save_dir, "confusion_matrix_best.csv"))
    report = classification_report(
        metrics["all_labels"], metrics["all_preds"],
        target_names=GRADE_NAMES, output_dict=True, zero_division=0)
    with open(os.path.join(save_dir, "classification_report_best.json"), "w") as f:
        json.dump(report, f, indent=2)
    rows = [{"img_stem": inst["img_stem"], "cls_id": inst["cls_id"],
             "gt_grade": GRADE_NAMES[gt], "pred_grade": GRADE_NAMES[pred],
             "correct": int(pred == gt)}
            for inst, pred, gt in zip(val_inst, metrics["all_preds"], metrics["all_labels"])]
    pd.DataFrame(rows).to_csv(os.path.join(save_dir, "val_predictions_best.csv"), index=False)
    cfg = vars(args).copy()
    cfg["save_dir"] = save_dir
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    if morph_stats is not None:
        with open(os.path.join(save_dir, "morph_stats.json"), "w") as f:
            json.dump(morph_stats, f, indent=2)
    print(f"\n  results saved to: {save_dir}")


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir",          default=DEFAULT_IMG_DIR)
    p.add_argument("--label_dir",        default=DEFAULT_LABEL_DIR)
    p.add_argument("--save_dir",         default=DEFAULT_SAVE_DIR)
    p.add_argument("--exp_name",         default=None)
    p.add_argument("--arch",             default="resnet18",
                   choices=["cnn", "resnet18", "mobilenet_v2"])
    p.add_argument("--pretrained",       action="store_true")
    p.add_argument("--roi_size",         type=int,   default=64)
    p.add_argument("--pad_ratio",        type=float, default=0.2)
    p.add_argument("--val_ratio",        type=float, default=0.2)
    p.add_argument("--epochs",           type=int,   default=350)
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    p.add_argument("--lr_min",           type=float, default=1e-5,
                   help="CosineAnnealingLR eta_min (set to 0 to disable schedule)")
    p.add_argument("--no_lr_schedule",   action="store_true", default=True,
                   help="disable lr schedule, keep lr fixed throughout training")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers",          type=int,   default=4)
    p.add_argument("--split_file",       default="severity/splits/split2.json")
    p.add_argument("--vis_only",         action="store_true")
    # E2
    p.add_argument("--loss_type",        default="ce",
                   choices=["ce", "distance_ce"])
    p.add_argument("--distance_alpha",   type=float, default=1.0)
    # E3 / E7
    p.add_argument("--head_type",        default="ce",
                   choices=["ce", "coral", "corn"])
    # E4
    p.add_argument("--use_morphology",      default="false",
                   help="true/false — enable morphology branch (E4)")
    p.add_argument("--morph_feat_set",      default="basic7")
    p.add_argument("--morph_hidden_dim",    type=int, default=32)
    # E5
    p.add_argument("--use_class_embedding", default="false",
                   help="true/false — enable class embedding branch (E5)")
    p.add_argument("--class_emb_dim",       type=int, default=16)
    p.add_argument("--num_defect_classes",  type=int, default=11,
                   help="max(cls_id)+1 in the dataset")
    # E6
    p.add_argument("--use_adaptive_thresholds", default="false",
                   help="true/false — enable class-conditional adaptive thresholds (E6)")
    p.add_argument("--threshold_hidden_dim",    type=int, default=32,
                   help="hidden dim of threshold MLP in AdaptiveCoralHead (E6)")
    # E8: prediction-aware ROI perturbation
    p.add_argument("--use_pred_aware_roi",      default="false",
                   help="true/false — enable prediction-aware ROI perturbation training (E8/E9)")
    p.add_argument("--roi_noise_prob",          type=float, default=0.5,
                   help="probability of applying bbox perturbation per sample (E8/E9)")
    p.add_argument("--roi_noise_iou_min",       type=float, default=0.80,
                   help="minimum IoU between perturbed and original bbox (E8/E9)")
    p.add_argument("--roi_noise_iou_max",       type=float, default=0.95,
                   help="maximum IoU between perturbed and original bbox (E8/E9)")
    p.add_argument("--roi_noise_max_trials",    type=int,   default=20,
                   help="max re-sample attempts before fallback to original bbox (E8/E9)")
    p.add_argument("--roi_noise_min_box_size",  type=int,   default=8,
                   help="minimum perturbed bbox side length in pixels (E8/E9)")
    # E9-strict: align context-dependent morphology with perturbed bbox
    p.add_argument("--use_morphology_aware_perturbation", default="false",
                   help="true/false — E9-strict: recompute context-dependent morph "
                        "features (gray_contrast) using perturbed bbox as context window. "
                        "Intrinsic geometry (area, dims, aspect_ratio) stays GT-based.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    use_morph  = args.use_morphology.lower()      in ("true", "1", "yes")
    use_cls_emb = args.use_class_embedding.lower() in ("true", "1", "yes")
    use_adapt_thres = args.use_adaptive_thresholds.lower() in ("true", "1", "yes")
    use_pred_aware  = args.use_pred_aware_roi.lower()      in ("true", "1", "yes")
    use_morph_aware = args.use_morphology_aware_perturbation.lower() in ("true", "1", "yes")
    set_seed(args.seed)

    # auto exp name
    if args.exp_name is None:
        if use_adapt_thres and use_pred_aware:
            args.exp_name = "E9_coral_morph_adaptthres_predaware"
        elif use_adapt_thres:
            args.exp_name = "E6_coral_morph_adaptthres"
        elif use_cls_emb:
            args.exp_name = "E5_coral_morph_classemb"
        elif use_morph and use_pred_aware and args.head_type == "coral":
            args.exp_name = "E8_coral_morph_predaware"
        elif use_morph:
            args.exp_name = "E4_coral_morphology"
        elif use_pred_aware and args.head_type == "coral":
            args.exp_name = "E8_coral_predaware"  # fallback: E3+pred-aware
        elif args.head_type == "coral":
            args.exp_name = "E3_coral"
        elif args.head_type == "corn":
            args.exp_name = "E7_corn"
        elif args.loss_type == "distance_ce":
            args.exp_name = f"E2_distance_ce_alpha{args.distance_alpha}"
        else:
            args.exp_name = "E1_ce"

    exp_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ── collect instances ─────────────────────────────────────────────────
    print("collecting instances...")
    all_inst = collect_instances(args.img_dir, args.label_dir,
                                 compute_morph=use_morph)
    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file) as f:
            sp = json.load(f)
        tr_stems = set(sp["train_stems"])
        va_stems = set(sp["val_stems"])
        train_inst = [i for i in all_inst if i["img_stem"] in tr_stems]
        val_inst   = [i for i in all_inst if i["img_stem"] in va_stems]
        print(f"  using fixed split: {args.split_file}")
    else:
        train_inst, val_inst = split_by_image(all_inst, args.val_ratio, args.seed)

    print_data_stats(all_inst, train_inst, val_inst)
    visualize_roi_samples(all_inst, exp_dir, roi_size=args.roi_size, pad_ratio=args.pad_ratio)

    if args.vis_only:
        return

    # ── morphology stats (fit on train, reuse on val) ─────────────────────
    morph_stats = None
    if use_morph:
        print("fitting morphology stats on train set...")
        morph_stats = fit_morph_stats(train_inst)
        print(f"  morph mean: {[round(v,4) for v in morph_stats['mean']]}")
        print(f"  morph std : {[round(v,4) for v in morph_stats['std']]}")

    # ── dataloaders ───────────────────────────────────────────────────────
    # E6 needs class_id (same as E5), so use_class_id = use_cls_emb or use_adapt_thres
    need_class_id = use_cls_emb or use_adapt_thres

    # E8/E9 perturbation kwargs — only passed to train_ds (val always clean)
    pred_aware_kwargs = dict(
        use_pred_aware_roi                = use_pred_aware,
        roi_noise_prob                    = args.roi_noise_prob,
        roi_noise_iou_min                 = args.roi_noise_iou_min,
        roi_noise_iou_max                 = args.roi_noise_iou_max,
        roi_noise_max_trials              = args.roi_noise_max_trials,
        roi_noise_min_box_size            = args.roi_noise_min_box_size,
        use_morphology_aware_perturbation = use_morph_aware,
        seed                              = args.seed,
    ) if use_pred_aware else {}

    train_ds = SeverityROIDataset(
        train_inst, roi_size=args.roi_size, pad_ratio=args.pad_ratio,
        augment=True, morph_stats=morph_stats, use_class_id=need_class_id,
        **pred_aware_kwargs)
    val_ds = SeverityROIDataset(
        val_inst, roi_size=args.roi_size, pad_ratio=args.pad_ratio,
        augment=False, morph_stats=morph_stats, use_class_id=need_class_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────────────
    device = torch.device(args.device)
    model = build_model(
        arch=args.arch, num_classes=4, pretrained=args.pretrained,
        dropout=0.3, head_type=args.head_type,
        use_morphology=use_morph,
        morph_dim=MORPH_DIM,
        morph_hidden_dim=args.morph_hidden_dim,
        use_class_embedding=use_cls_emb or use_adapt_thres,  # E6 also needs class_emb
        num_defect_classes=args.num_defect_classes,
        class_emb_dim=args.class_emb_dim,
        use_adaptive_thresholds=use_adapt_thres,
        threshold_hidden_dim=args.threshold_hidden_dim,
    ).to(device)

    criterion = make_criterion(args)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    # CosineAnnealingLR: lr decays from args.lr to args.lr_min over all epochs.
    # Set --lr_min equal to --lr (or use --no_lr_schedule) to keep fixed lr.
    if args.no_lr_schedule or args.lr_min == args.lr:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_tag = "coral" if args.head_type == "coral" else args.loss_type
    print(f"\n  arch={args.arch}  head={args.head_type}  loss={loss_tag}"
          f"  morph={use_morph}  class_emb={use_cls_emb or use_adapt_thres}"
          f"  adapt_thres={use_adapt_thres}  pred_aware_roi={use_pred_aware}"
          f"  morph_aware_perturb={use_morph_aware}"
          f"  params={n_params:,}  device={device}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr} -> {args.lr_min} (cosine)\n")
    if use_pred_aware:
        mode_tag = "E9-strict" if use_morph_aware else ("E9-min" if use_adapt_thres else "E8")
        print(f"  [{mode_tag}] roi_noise_prob={args.roi_noise_prob}"
              f"  iou=[{args.roi_noise_iou_min},{args.roi_noise_iou_max}]"
              f"  max_trials={args.roi_noise_max_trials}")
        if use_morph_aware:
            print(f"  [E9-strict] context-dependent morph (gray_contrast) will use "
                  f"perturbed bbox as background window.")
            print(f"  [E9-strict] intrinsic geometry (area, dims, aspect_ratio) "
                  f"remains GT-based.")
        elif use_adapt_thres:
            print(f"  [E9-min] morphology branch uses clean GT prior; "
                  f"only ROI image branch is prediction-aware.")
        print()

    # ── training loop ─────────────────────────────────────────────────────
    best_ckpt = os.path.join(exp_dir, "best.pth")
    last_ckpt = os.path.join(exp_dir, "last.pth")
    best_kappa, log_rows = -1.0, []

    for epoch in range(1, args.epochs + 1):
        if use_pred_aware:
            train_ds.reset_perturbation_stats()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            args.head_type, use_morph, use_cls_emb or use_adapt_thres,
            use_adapt_thres)
        val_loss, vm = evaluate(
            model, val_loader, criterion, device,
            args.head_type, use_morph, use_cls_emb or use_adapt_thres,
            use_adapt_thres)

        kappa = vm["weighted_kappa"]
        cur_lr = optimizer.param_groups[0]["lr"]
        log_row = {"epoch": epoch,
                   "lr":          round(cur_lr, 8),
                   "train_loss": round(tr_loss, 6),
                   "val_loss":   round(val_loss, 6),
                   "acc":        round(vm["overall_acc"], 6),
                   "macro_f1":   round(vm["macro_f1"], 6),
                   "qwk":        round(kappa, 6)}

        # E8: append perturbation stats to log row
        if use_pred_aware:
            ps = train_ds.get_perturbation_stats()
            log_row.update({
                "noise_ratio":   round(ps["noised_ratio"], 4),
                "noise_mean_iou": round(ps["mean_iou"], 4),
                "noise_fallback": ps["fallback_count"],
            })

        log_rows.append(log_row)

        flag = ""
        if kappa > best_kappa:
            best_kappa = kappa
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "best_kappa": best_kappa}, best_ckpt)
            flag = "  <- best"

        noise_tag = ""
        if use_pred_aware:
            ps = train_ds.get_perturbation_stats()
            noise_tag = f"  noise={ps['noised_ratio']:.2f} iou={ps['mean_iou']:.3f} fb={ps['fallback_count']}"

        print(f"Epoch {epoch:>3}/{args.epochs}  "
              f"lr={cur_lr:.2e}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"acc={vm['overall_acc']:.4f}  "
              f"macro_f1={vm['macro_f1']:.4f}  "
              f"qwk={kappa:.4f}{noise_tag}{flag}")

        scheduler.step()

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": args.epochs, "best_kappa": best_kappa}, last_ckpt)

    # ── final eval ────────────────────────────────────────────────────────
    print(f"\nloading best weights: {best_ckpt}")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    _, final_metrics = evaluate(model, val_loader, criterion, device,
                                args.head_type, use_morph,
                                use_cls_emb or use_adapt_thres,
                                use_adapt_thres)

    print("\n" + "=" * 60)
    print_metrics(final_metrics, prefix="Val")
    print(f"  Macro-F1 : {final_metrics['macro_f1']:.4f}")
    print(f"  Best QWK : {best_kappa:.4f}")
    print(f"  exp_dir  : {exp_dir}")

    save_results(exp_dir, log_rows, final_metrics, val_inst, args,
                 morph_stats=morph_stats)

    # E8/E9: save perturbation debug visualization
    if use_pred_aware:
        visualize_perturbation_samples(
            train_inst, exp_dir, args,
            n_samples=8, roi_size=args.roi_size, pad_ratio=args.pad_ratio,
        )
    # E9-strict: additional debug — show how gray_contrast changes with perturbed bbox
    if use_pred_aware and use_morph_aware:
        _debug_morph_perturbation(train_inst, exp_dir, args)


if __name__ == "__main__":
    main()
