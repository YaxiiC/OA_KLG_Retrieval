"""
Image-only baseline training for KLGrade (multiclass or binary OA).

Example (multiclass):
python train_image_baseline.py \
    --images-tr "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTr" \
    --radiomics-train-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/output_train/radiomics_results_wide.csv" \
    --klgrade-train-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/subInfo_train.xlsx" \
    --outdir "training_logs_image_baseline" \
    --arch resnet3d \
    --epochs 200\
    --device cuda:0 \
    --label-mode binary_oa 

Example (binary OA, EfficientNet):
python train_image_baseline.py \
    --images-tr "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTr" \
    --radiomics-train-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/output_train/radiomics_results_wide.csv" \
    --klgrade-train-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/subInfo_train.xlsx" \
    --outdir "training_logs_image_baseline_eff_b"  \
    --arch efficientnet_b0 \
    --device cuda:0 \
    --label-mode binary_oa \


python train_image_baseline.py \
  --images-tr "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTr" \
  --radiomics-train-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/output_train/radiomics_results_wide.csv" \
  --klgrade-train-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/subInfo_train.xlsx" \
  --outdir "training_logs_image_baseline_ordinal_eff_b" \
  --arch efficientnet_b0 \
  --label-mode multiclass \
  --epochs 500 --batch-size 4 --lr 1e-4 --weight-decay 1e-4 \
  --use-class-weights \
  --device cuda:1 \
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from models import EfficientNet3DBackbone
from data_loader import (
    load_radiomics_wide_format,
    load_klgrade_labels,
    KLGradeDataset
)
from training_utils import compute_metrics
from training_utils import is_ordinal_kl_setting, ordinal_cross_entropy_loss, ordinal_logits_to_pred

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import torchvision.models.video as video_models
    import torchvision.models as tv_models
    try:
        from torchvision.models.video import R3D_18_Weights
    except Exception:
        R3D_18_Weights = None
    try:
        from torchvision.models import EfficientNet_B0_Weights
    except Exception:
        EfficientNet_B0_Weights = None
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    logger.warning("torchvision not available; only simple CNN backbone is supported")


class Simple3DCNN(nn.Module):
    def __init__(self, out_dim: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
        self.classifier = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class Simple2DCNN(nn.Module):
    def __init__(self, out_dim: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class ImageOnlyModel(nn.Module):
    def __init__(
        self,
        arch: str,
        out_dim: int,
        pretrained: bool = False
    ):
        super().__init__()
        self.arch = arch
        self.pretrained = pretrained
        # Only the torchvision EfficientNet path is 2D in this repo.
        self.uses_2d = (arch == "efficientnet_b0")

        if arch == "resnet3d":
            if HAS_TORCHVISION:
                weights = None
                if pretrained and R3D_18_Weights is not None:
                    weights = R3D_18_Weights.DEFAULT
                backbone = video_models.r3d_18(weights=weights)
                old_conv = backbone.stem[0]
                if old_conv.in_channels != 1:
                    new_conv = nn.Conv3d(
                        1, old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=old_conv.bias is not None
                    )
                    if pretrained:
                        with torch.no_grad():
                            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
                            if old_conv.bias is not None:
                                new_conv.bias.data = old_conv.bias.data
                    backbone.stem[0] = new_conv
                backbone.fc = nn.Linear(backbone.fc.in_features, out_dim)
                self.backbone = backbone
            else:
                self.backbone = Simple3DCNN(out_dim=out_dim)
        elif arch == "efficientnet_b0":
            if not HAS_TORCHVISION:
                self.backbone = Simple2DCNN(out_dim=out_dim)
            else:
                weights = None
                if pretrained and EfficientNet_B0_Weights is not None:
                    weights = EfficientNet_B0_Weights.DEFAULT
                backbone = tv_models.efficientnet_b0(weights=weights)
                first_conv = backbone.features[0][0]
                if first_conv.in_channels != 1:
                    new_conv = nn.Conv2d(
                        1, first_conv.out_channels,
                        kernel_size=first_conv.kernel_size,
                        stride=first_conv.stride,
                        padding=first_conv.padding,
                        bias=first_conv.bias is not None
                    )
                    if pretrained:
                        with torch.no_grad():
                            new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
                            if first_conv.bias is not None:
                                new_conv.bias.data = first_conv.bias.data
                    backbone.features[0][0] = new_conv
                backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, out_dim)
                self.backbone = backbone
        elif arch == "efficientnet3d_b0":
            # Custom 3D EfficientNet-style backbone (no pretrained weights).
            self.backbone = EfficientNet3DBackbone(in_channels=1, out_dim=out_dim)
        else:
            raise ValueError(f"Unsupported arch: {arch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.uses_2d:
            # Use full volume by averaging across depth
            x = x.mean(dim=2)
        return self.backbone(x)


def main():
    parser = argparse.ArgumentParser(
        description="Image-only baseline training for KLGrade",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--images-tr", type=str, required=True, help="Training images directory")
    parser.add_argument("--radiomics-train-csv", type=str, required=True, help="Radiomics CSV (wide format)")
    parser.add_argument("--klgrade-train-csv", type=str, required=True, help="Training KLGrade labels file")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")

    parser.add_argument("--arch", type=str, default="resnet3d",
                        choices=["resnet3d", "efficientnet_b0", "efficientnet3d_b0"])
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights (if available)")
    parser.add_argument("--label-mode", type=str, default="multiclass",
                        choices=["multiclass", "binary_oa"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--class-weight-method", type=str, default="balanced",
                        choices=["balanced", "balanced_subsample"])
    parser.add_argument("--target-shape", type=int, nargs=3, default=[32, 128, 128], help="D H W")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(exist_ok=True)
    (outdir / "checkpoints").mkdir(exist_ok=True)

    config_dict = vars(args)
    config_dict["timestamp"] = datetime.now().isoformat()
    with open(outdir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Loading radiomics and labels...")
    radiomics_train, _, _, _ = load_radiomics_wide_format(Path(args.radiomics_train_csv))
    labels_raw = load_klgrade_labels(Path(args.klgrade_train_csv))
    if args.label_mode == "binary_oa":
        labels = {cid: (0 if lab <= 1 else 1) for cid, lab in labels_raw.items()}
        label_num_classes = 2
    else:
        labels = labels_raw
        label_num_classes = 5
    use_ordinal = (args.label_mode == "multiclass") and is_ordinal_kl_setting(label_num_classes)
    out_dim = (label_num_classes - 1) if use_ordinal else label_num_classes
    logger.info(
        f"Label mode: {args.label_mode} | label_num_classes={label_num_classes} | "
        f"ordinal={use_ordinal} | out_dim={out_dim}"
    )

    case_ids = list(set(radiomics_train.keys()) & set(labels.keys()))
    logger.info(f"Cases with both radiomics and labels: {len(case_ids)}")

    train_ids, val_ids = train_test_split(
        case_ids,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=[labels[cid] for cid in case_ids]
    )

    train_dataset = KLGradeDataset(
        train_ids,
        Path(args.images_tr),
        radiomics_train,
        labels,
        target_shape=tuple(args.target_shape)
    )
    val_dataset = KLGradeDataset(
        val_ids,
        Path(args.images_tr),
        radiomics_train,
        labels,
        target_shape=tuple(args.target_shape)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )

    model = ImageOnlyModel(
        arch=args.arch,
        out_dim=out_dim,
        pretrained=args.pretrained
    ).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    class_weights = None
    if args.use_class_weights:
        train_labels_array = np.array([labels[cid] for cid in train_ids])
        classes = np.array([0, 1] if label_num_classes == 2 else [0, 1, 2, 3, 4])
        weights = compute_class_weight(
            class_weight=args.class_weight_method,
            classes=classes,
            y=train_labels_array
        )
        class_weights = torch.FloatTensor(weights).to(device)
        logger.info(f"Using class weights: {weights}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_macro_f1 = -1.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []

        for batch in train_loader:
            images = batch["image"].to(device)
            labels_batch = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(images)
            if use_ordinal:
                loss = ordinal_cross_entropy_loss(
                    logits, labels_batch.view(-1), label_num_classes=label_num_classes, class_weights=class_weights
                )
                preds_batch = ordinal_logits_to_pred(logits)
            else:
                loss = F.cross_entropy(logits, labels_batch.view(-1), weight=class_weights)
                preds_batch = logits.argmax(dim=1)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_preds.extend(preds_batch.cpu().numpy())
            train_labels.extend(labels_batch.cpu().numpy())

        train_metrics = compute_metrics(
            np.array(train_labels),
            np.array(train_preds),
            y_proba=None
        )

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels_batch = batch["label"].to(device)
                logits = model(images)
                if use_ordinal:
                    loss = ordinal_cross_entropy_loss(
                        logits, labels_batch.view(-1), label_num_classes=label_num_classes, class_weights=class_weights
                    )
                    preds_batch = ordinal_logits_to_pred(logits)
                else:
                    loss = F.cross_entropy(logits, labels_batch.view(-1), weight=class_weights)
                    preds_batch = logits.argmax(dim=1)
                val_losses.append(loss.item())
                val_preds.extend(preds_batch.cpu().numpy())
                val_labels.extend(labels_batch.cpu().numpy())

        val_metrics = compute_metrics(
            np.array(val_labels),
            np.array(val_preds),
            y_proba=None
        )

        epoch_time = time.time() - start
        is_best = val_metrics["macro_f1"] > best_macro_f1
        if is_best:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {np.mean(train_losses):.4f} | "
            f"Val Loss: {np.mean(val_losses):.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Val QWK: {val_metrics['qwk']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        history_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_acc": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_qwk": train_metrics["qwk"],
            "val_loss": float(np.mean(val_losses)),
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_qwk": val_metrics["qwk"],
            "time_sec": epoch_time
        }
        history.append(history_row)
        # Write full history each epoch for easy tracking
        import pandas as pd
        pd.DataFrame(history).to_csv(outdir / "logs" / "metrics.csv", index=False)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_macro_f1": val_metrics["macro_f1"],
            "args": vars(args),
            "num_classes": label_num_classes,
            "ordinal": bool(use_ordinal),
            "logit_dim": int(out_dim),
            "label_num_classes": int(label_num_classes),
            "arch": args.arch
        }
        torch.save(checkpoint, outdir / "checkpoints" / "last.pth")
        if is_best:
            torch.save(checkpoint, outdir / "checkpoints" / "best.pth")
            logger.info(f"  → Best checkpoint saved (Val Macro-F1: {best_macro_f1:.4f})")

    logger.info(f"Training complete. Best epoch: {best_epoch}, best Macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()

