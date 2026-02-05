"""
Joint Training: Subset Scorer + Token-Set KL Classifier

python train_joint_scoring_kl.py \
    --images-tr "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTr" \
    --radiomics-train-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/output_train/radiomics_results_wide.csv" \
    --klgrade-train-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/subInfo_train.xlsx" \
    --outdir "training_logs_joint_scoring_k50b" \
    --k 50 \
    --n-subsets 50 \
    --top-m 5 \
    --pool-size 1000 \
    --warmup-epochs 80 \
    --epochs 800 \
    --lambda-rank 0.5 \
    --exploration-ratio 0.6 \
    --probe-support 20 \
    --probe-query 20 \
    --probe-steps 10 \
    --probe-lr 5e-3 \
    --emb-dim 128 \
    --weight-decay 5e-3 \
    --device cuda:0 \
    --use-class-weights \
    --label-mode binary_oa

python train_joint_scoring_kl.py \
  --images-tr "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTr" \
  --radiomics-train-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/output_train/radiomics_results_wide.csv" \
  --klgrade-train-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/subInfo_train.xlsx" \
  --outdir "training_logs_joint_scoring_ordinal_k25" \
  --label-mode multiclass \
  --k 25 --n-subsets 50 --top-m 5 --pool-size 2000 \
  --warmup-epochs 80 --epochs 800 \
  --lambda-rank 0.5 --exploration-ratio 0.6 \
  --probe-support 20 --probe-query 20 --probe-steps 10 --probe-lr 5e-3 \
  --emb-dim 128 --lr 1e-4 --weight-decay 5e-3 \
  --batch-size 4 --num-workers 4 \
  --use-class-weights \
  --device cuda:1


"""

import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

from models import JointScoringModel
from data_loader import (
    load_radiomics_wide_format,
    load_klgrade_labels,
    KLGradeDataset
)
from training_utils import train_epoch_subset, validate_subset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train subset scorer + token-set KL classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--images-tr", type=str, required=True,
                        help="Training images directory")
    parser.add_argument("--radiomics-train-csv", type=str, required=True,
                        help="Training radiomics CSV (wide format)")
    parser.add_argument("--klgrade-train-csv", type=str, required=True,
                        help="Training KLGrade labels file (CSV or Excel .xlsx)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory")

    # Subset hyperparameters
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--n-subsets", type=int, default=32)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=320)
    parser.add_argument("--exploration-ratio", type=float, default=0.2)

    # Training hyperparameters
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--lambda-rank", type=float, default=0.1)
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
    parser.add_argument("--label-mode", type=str, default="multiclass",
                        choices=["multiclass", "binary_oa"],
                        help="Label mode: 5-class KL (multiclass) or binary OA")
    parser.add_argument("--emb-dim", type=int, default=256,
                        help="Embedding dim for image/token encoders and scorer/classifier")

    # Probe hyperparameters
    parser.add_argument("--probe-support", type=int, default=16)
    parser.add_argument("--probe-query", type=int, default=16)
    parser.add_argument("--probe-steps", type=int, default=3)
    parser.add_argument("--probe-lr", type=float, default=1e-2)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)
    (outdir / "checkpoints").mkdir(exist_ok=True)

    config_dict = vars(args)
    config_dict['timestamp'] = datetime.now().isoformat()
    with open(outdir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Loading radiomics and labels...")
    radiomics_train, feature_names, _, feature_meta = load_radiomics_wide_format(
        Path(args.radiomics_train_csv)
    )
    labels_train_raw = load_klgrade_labels(Path(args.klgrade_train_csv))
    if args.label_mode == "binary_oa":
        labels_train = {cid: (0 if lab <= 1 else 1) for cid, lab in labels_train_raw.items()}
    else:
        labels_train = labels_train_raw
    label_num_classes = 2 if args.label_mode == "binary_oa" else 5
    use_ordinal = (args.label_mode == "multiclass") and (label_num_classes == 5)
    logit_dim = (label_num_classes - 1) if use_ordinal else label_num_classes
    logger.info(
        f"Label mode: {args.label_mode} | label_num_classes={label_num_classes} | "
        f"ordinal={use_ordinal} | logit_dim={logit_dim}"
    )

    train_case_ids = list(set(radiomics_train.keys()) & set(labels_train.keys()))
    logger.info(f"Cases with both radiomics and labels: {len(train_case_ids)}")

    train_ids, val_ids = train_test_split(
        train_case_ids,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=[labels_train[cid] for cid in train_case_ids]
    )

    scaler = StandardScaler()
    train_array = np.array([radiomics_train[cid] for cid in train_ids])
    scaler.fit(train_array)

    for cid in train_ids:
        radiomics_train[cid] = scaler.transform(radiomics_train[cid].reshape(1, -1))[0]
    for cid in val_ids:
        radiomics_train[cid] = scaler.transform(radiomics_train[cid].reshape(1, -1))[0]

    joblib.dump(scaler, outdir / "checkpoints" / "scaler.joblib")
    with open(outdir / "checkpoints" / "feature_names.json", "w") as f:
        json.dump({i: name for i, name in enumerate(feature_names)}, f, indent=2)
    with open(outdir / "checkpoints" / "feature_meta.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    train_dataset = KLGradeDataset(
        train_ids,
        Path(args.images_tr),
        radiomics_train,
        labels_train,
        target_shape=(32, 128, 128)
    )
    val_dataset = KLGradeDataset(
        val_ids,
        Path(args.images_tr),
        radiomics_train,
        labels_train,
        target_shape=(32, 128, 128)
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

    num_features = len(feature_names)
    feature_to_roi = np.array(feature_meta["feature_to_roi"], dtype=np.int64)
    feature_to_type = np.array(feature_meta["feature_to_type"], dtype=np.int64)
    num_rois = len(feature_meta["roi_names"])
    num_types = len(feature_meta["type_names"])
    model = JointScoringModel(
        num_features=num_features,
        num_rois=num_rois,
        num_types=num_types,
        emb_dim=args.emb_dim,
        num_classes=logit_dim
    ).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    class_weights = None
    if args.use_class_weights:
        train_labels_array = np.array([labels_train[cid] for cid in train_ids])
        classes = np.array([0, 1] if label_num_classes == 2 else [0, 1, 2, 3, 4])
        weights = compute_class_weight(
            class_weight=args.class_weight_method,
            classes=classes,
            y=train_labels_array
        )
        class_weights = torch.FloatTensor(weights).to(device)
        logger.info(f"Using class weights: {weights}")

    best_macro_f1 = -1.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_metrics = train_epoch_subset(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            warmup_epochs=args.warmup_epochs,
            train_case_ids=train_ids,
            radiomics_dict=radiomics_train,
            labels_dict=labels_train,
            feature_to_roi=feature_to_roi,
            feature_to_type=feature_to_type,
            k=args.k,
            n_subsets=args.n_subsets,
            top_m=args.top_m,
            pool_size=args.pool_size,
            num_classes=label_num_classes,
            lambda_rank=args.lambda_rank,
            exploration_ratio=args.exploration_ratio,
            probe_support=args.probe_support,
            probe_query=args.probe_query,
            probe_steps=args.probe_steps,
            probe_lr=args.probe_lr,
            class_weights=class_weights
        )

        val_loss, val_metrics = validate_subset(
            model=model,
            dataloader=val_loader,
            device=device,
            radiomics_dict=radiomics_train,
            feature_to_roi=feature_to_roi,
            feature_to_type=feature_to_type,
            k=args.k,
            top_m=args.top_m,
            pool_size=args.pool_size
        )

        epoch_time = time.time() - start
        is_best = val_metrics["macro_f1"] > best_macro_f1
        if is_best:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Val QWK: {val_metrics['qwk']:.4f} | Time: {epoch_time:.1f}s"
        )
        if "probe_reward_mean" in train_metrics:
            logger.info(
                "Probe Reward | "
                f"mean: {train_metrics['probe_reward_mean']:.4f} | "
                f"std: {train_metrics['probe_reward_std']:.4f} | "
                f"min: {train_metrics['probe_reward_min']:.4f} | "
                f"max: {train_metrics['probe_reward_max']:.4f} | "
                f"score_mean: {train_metrics['scorer_score_mean']:.4f} | "
                f"score_std: {train_metrics['scorer_score_std']:.4f}"
            )

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_qwk": train_metrics["qwk"],
            "train_cls_loss": train_metrics.get("cls_loss", None),
            "train_rank_loss": train_metrics.get("rank_loss", None),
            "probe_reward_mean": train_metrics.get("probe_reward_mean", None),
            "probe_reward_std": train_metrics.get("probe_reward_std", None),
            "probe_reward_min": train_metrics.get("probe_reward_min", None),
            "probe_reward_max": train_metrics.get("probe_reward_max", None),
            "scorer_score_mean": train_metrics.get("scorer_score_mean", None),
            "scorer_score_std": train_metrics.get("scorer_score_std", None),
            "val_loss": val_loss,
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_qwk": val_metrics["qwk"],
            "time_sec": epoch_time
        }
        history.append(history_row)
        pd.DataFrame(history).to_csv(outdir / "logs" / "metrics.csv", index=False)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_qwk": val_metrics["qwk"],
            "val_macro_f1": val_metrics["macro_f1"],
            "args": vars(args),
            "num_features": num_features,
            "num_classes": label_num_classes,
            "ordinal": bool(use_ordinal),
            "logit_dim": int(logit_dim),
            "label_num_classes": int(label_num_classes),
        }
        torch.save(checkpoint, outdir / "checkpoints" / "last.pth")
        if is_best:
            torch.save(checkpoint, outdir / "checkpoints" / "best.pth")
            logger.info(f"  → Best checkpoint saved (Val Macro-F1: {best_macro_f1:.4f})")

    logger.info(f"Training complete. Best epoch: {best_epoch}, best Macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()

