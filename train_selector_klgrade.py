r"""
Joint Training: Image-Conditioned Feature Selector + KLGrade Classifier

Main training script that orchestrates:
1. Data loading and preprocessing
2. Model initialization
3. Training loop with two-stage gating
4. Inference and saving results

python train_selector_klgrade.py `
    --images-tr "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTr" `
    --images-ts "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTs" `
    --radiomics-train-csv "C:\Users\chris\MICCAI2026\OA_KLG_TopK\output_train\radiomics_results.csv" `
    --radiomics-test-csv "C:\Users\chris\MICCAI2026\OA_KLG_TopK\output_test\radiomics_results.csv" `
    --klgrade-train-csv "C:\Users\chris\MICCAI2026\OA_KLG_TopK\subInfo_train.xlsx" `
    --outdir "training_logs" `
    --k 15 `
    --warmup-epochs 30 `
    --epochs 500 `
    --early-stopping-patience 100 `
    --batch-size 8 `
    --lr 1e-4 `
    --use-class-weights
    --variance-threshold 1e-6

python train_selector_klgrade.py \
    --images-tr "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTr" \
    --images-ts "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTs" \
    --radiomics-train-csv "/home/yaxi/OA_KLG_TopK/output_train/radiomics_results.csv" \
    --radiomics-test-csv "/home/yaxi/OA_KLG_TopK/output_test/radiomics_results.csv" \
    --klgrade-train-csv "/home/yaxi/OA_KLG_TopK/subInfo_train.xlsx" \
    --outdir "training_logs_k50" \
    --k 50 \
    --warmup-epochs 100 \
    --epochs 500 \
    --early-stopping-patience 300 \
    --batch-size 8 \
    --lr 1e-4 \
    --use-class-weights \
    --lambda-diversity 0.3 \
    --device cuda:0 \
    --variance-threshold 1e-5

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import from separate modules
from models import JointModel
from data_loader import (
    load_radiomics_long_format,
    load_klgrade_labels,
    KLGradeDataset
)
from training_utils import train_epoch, validate, compute_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_loss_curve(history_df: pd.DataFrame, save_path: Path):
    """Plot train vs val total loss."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history_df['epoch'], history_df['train_loss'], label='Train', linewidth=2)
    ax.plot(history_df['epoch'], history_df['val_loss'], label='Val', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_components(history_df: pd.DataFrame, save_path: Path):
    """Plot CE loss and loss_k components."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # CE loss
    ax1.plot(history_df['epoch'], history_df['train_ce_loss'], label='Train CE', linewidth=2)
    ax1.plot(history_df['epoch'], history_df['val_ce_loss'], label='Val CE', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('CE Loss', fontsize=12)
    ax1.set_title('Classification Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Loss K
    ax2.plot(history_df['epoch'], history_df['train_loss_k'], label='Train Loss_K', linewidth=2, color='orange')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss_K', fontsize=12)
    ax2.set_title('Sparsity Regularization Loss', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_curve(history_df: pd.DataFrame, save_path: Path):
    """Plot validation metrics: acc, macro-F1, QWK."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history_df['epoch'], history_df['val_acc'], label='Val Accuracy', linewidth=2, marker='o', markersize=3)
    ax.plot(history_df['epoch'], history_df['val_macro_f1'], label='Val Macro F1', linewidth=2, marker='s', markersize=3)
    ax.plot(history_df['epoch'], history_df['val_qwk'], label='Val QWK', linewidth=2, marker='^', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Validation Metrics', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_gate_stats(history_df: pd.DataFrame, save_path: Path, target_k: int = 15):
    """Plot mean p.sum() vs epoch."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history_df['epoch'], history_df['mean_p_sum'], label='Mean p.sum()', linewidth=2, color='purple', marker='o', markersize=3)
    ax.axhline(y=target_k, 
               color='r', linestyle='--', label=f'Target k={target_k}', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean p.sum()', fontsize=12)
    ax.set_title('Gate Statistics: Mean Sum of Gate Probabilities', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, normalize: bool = False):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=[0, 1, 2, 3, 4],
           yticklabels=[0, 1, 2, 3, 4],
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train image-conditioned feature selector + KLGrade classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument("--images-tr", type=str, required=True,
                        help="Training images directory")
    parser.add_argument("--images-ts", type=str, required=True,
                        help="Test images directory")
    parser.add_argument("--radiomics-train-csv", type=str, required=True,
                        help="Training radiomics CSV (long format)")
    parser.add_argument("--radiomics-test-csv", type=str, required=True,
                        help="Test radiomics CSV (long format)")
    parser.add_argument("--klgrade-train-csv", type=str, required=True,
                        help="Training KLGrade labels file (CSV or Excel .xlsx)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory")
    
    # Model hyperparameters
    parser.add_argument("--k", type=int, default=15,
                        help="Top-k features to select")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained backbone")
    
    # Training hyperparameters
    parser.add_argument("--warmup-epochs", type=int, default=20,
                        help="Number of warmup epochs")
    parser.add_argument("--warmup-thr-start", type=float, default=0.0,
                        help="Warmup threshold start")
    parser.add_argument("--warmup-thr-end", type=float, default=0.5,
                        help="Warmup threshold end")
    parser.add_argument("--lambda-k-start", type=float, default=0.005,
                        help="Lambda_k start value")
    parser.add_argument("--lambda-k", type=float, default=0.05,
                        help="Lambda_k end value")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation split ratio")
    
    # Other
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda:0, cpu, etc.)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--use-class-weights", action="store_true",
                        help="Use class weights to handle class imbalance")
    parser.add_argument("--class-weight-method", type=str, default="balanced",
                        choices=["balanced", "balanced_subsample"],
                        help="Method for computing class weights (balanced: inverse frequency)")
    parser.add_argument("--lambda-diversity", type=float, default=0.1,
                        help="Weight for diversity loss (encourages different feature selections)")
    parser.add_argument("--variance-threshold", type=float, default=1e-6,
                        help="Variance threshold for feature filtering (features with variance < threshold will be removed)")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directories
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)
    (outdir / "checkpoints").mkdir(exist_ok=True)
    
    # Save config.json
    config_dict = vars(args)
    config_dict['timestamp'] = datetime.now().isoformat()
    with open(outdir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Saved config to {outdir / 'config.json'}")
    
    # Load data
    logger.info("=" * 80)
    logger.info("Loading data...")
    logger.info("=" * 80)
    
    # Load radiomics
    radiomics_train, roi_names, feature_names, missing_stats_train = load_radiomics_long_format(
        Path(args.radiomics_train_csv)
    )
    radiomics_test, _, _, missing_stats_test = load_radiomics_long_format(
        Path(args.radiomics_test_csv),
        expected_rois=roi_names,
        expected_features=feature_names
    )
    
    # Load labels (needed to filter to cases with labels)
    labels_train = load_klgrade_labels(Path(args.klgrade_train_csv))
    
    # Get common case IDs (cases with both radiomics and labels)
    train_case_ids_with_labels = list(set(radiomics_train.keys()) & set(labels_train.keys()))
    logger.info(f"Cases with both radiomics and labels: {len(train_case_ids_with_labels)}")
    
    # Variance-based feature filtering
    logger.info("=" * 80)
    logger.info("Variance-based Feature Filtering")
    logger.info("=" * 80)
    
    # Calculate variance for each feature across training samples (only cases with labels)
    train_radiomics_array = np.array([radiomics_train[cid] for cid in train_case_ids_with_labels])
    feature_variances = np.var(train_radiomics_array, axis=0)
    
    # Log variance statistics
    logger.info(f"Total features before filtering: {len(feature_variances)}")
    logger.info(f"Variance statistics:")
    logger.info(f"  Min:    {np.min(feature_variances):.6e}")
    logger.info(f"  Max:    {np.max(feature_variances):.6e}")
    logger.info(f"  Mean:   {np.mean(feature_variances):.6e}")
    logger.info(f"  Median: {np.median(feature_variances):.6e}")
    logger.info(f"  Std:    {np.std(feature_variances):.6e}")
    
    # Filter features based on variance threshold
    valid_feature_indices = np.where(feature_variances >= args.variance_threshold)[0]
    removed_feature_indices = np.where(feature_variances < args.variance_threshold)[0]
    
    logger.info(f"Variance threshold: {args.variance_threshold:.6e}")
    logger.info(f"Features kept: {len(valid_feature_indices)}")
    logger.info(f"Features removed: {len(removed_feature_indices)} ({100*len(removed_feature_indices)/len(feature_variances):.2f}%)")
    
    # Build feature mapping for removed features (for logging)
    original_feature_mapping = {}
    for idx, (roi, feat) in enumerate([(r, f) for r in roi_names for f in feature_names]):
        original_feature_mapping[idx] = f"{roi}:{feat}"
    
    if len(removed_feature_indices) > 0:
        removed_features_info = []
        for idx in removed_feature_indices:
            removed_features_info.append({
                "index": int(idx),
                "name": original_feature_mapping[int(idx)],
                "variance": float(feature_variances[idx])
            })
        
        # Save removed features info
        removed_features_path = outdir / "logs" / "removed_features_variance_filter.json"
        with open(removed_features_path, "w") as f:
            json.dump({
                "variance_threshold": args.variance_threshold,
                "total_features_before": len(feature_variances),
                "features_kept": len(valid_feature_indices),
                "features_removed": len(removed_feature_indices),
                "removed_features": removed_features_info
            }, f, indent=2)
        logger.info(f"Saved removed features info to {removed_features_path}")
    
    # Filter radiomics data
    logger.info("Filtering radiomics data...")
    for cid in list(radiomics_train.keys()):
        radiomics_train[cid] = radiomics_train[cid][valid_feature_indices]
    for cid in list(radiomics_test.keys()):
        radiomics_test[cid] = radiomics_test[cid][valid_feature_indices]
    
    logger.info("=" * 80)
    
    # Save feature mapping (only for kept features)
    feature_mapping = {}
    for new_idx, old_idx in enumerate(valid_feature_indices):
        feature_mapping[new_idx] = original_feature_mapping[int(old_idx)]
    
    with open(outdir / "checkpoints" / "feature_names.json", "w") as f:
        json.dump(feature_mapping, f, indent=2)
    logger.info(f"Saved feature mapping to {outdir / 'checkpoints/feature_names.json'}")
    
    # Get common case IDs (already computed above, but update variable name)
    train_case_ids = train_case_ids_with_labels
    logger.info(f"Training cases with both radiomics and labels: {len(train_case_ids)}")
    
    # Stratified train/val split
    train_ids, val_ids = train_test_split(
        train_case_ids,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=[labels_train[cid] for cid in train_case_ids]
    )
    logger.info(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    
    # Compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        train_labels_array = np.array([labels_train[cid] for cid in train_ids])
        classes = np.array([0, 1, 2, 3, 4])
        
        # Compute class weights
        weights = compute_class_weight(
            class_weight=args.class_weight_method,
            classes=classes,
            y=train_labels_array
        )
        class_weights = torch.FloatTensor(weights).to(device)
        
        # Log class distribution and weights
        from collections import Counter
        label_counts = Counter(train_labels_array)
        logger.info("=" * 80)
        logger.info("Class Distribution (Training Set):")
        logger.info("-" * 80)
        logger.info(f"{'Class':<10} {'Count':<10} {'Weight':<10}")
        logger.info("-" * 80)
        for cls in classes:
            count = label_counts.get(cls, 0)
            weight = weights[cls]
            logger.info(f"{cls:<10} {count:<10} {weight:<10.4f}")
        logger.info("=" * 80)
    else:
        logger.info("Class weights disabled (using uniform weights)")
    
    # Normalize radiomics (fit on train only)
    logger.info("Fitting radiomics scaler on training set...")
    train_radiomics_array = np.array([radiomics_train[cid] for cid in train_ids])
    scaler = StandardScaler()
    scaler.fit(train_radiomics_array)
    
    # Apply scaler
    for cid in train_ids:
        radiomics_train[cid] = scaler.transform(radiomics_train[cid].reshape(1, -1))[0]
    for cid in val_ids:
        if cid in radiomics_train:
            radiomics_train[cid] = scaler.transform(radiomics_train[cid].reshape(1, -1))[0]
    for cid in radiomics_test.keys():
        radiomics_test[cid] = scaler.transform(radiomics_test[cid].reshape(1, -1))[0]
    
    # Save scaler
    joblib.dump(scaler, outdir / "checkpoints" / "scaler.joblib")
    logger.info(f"Saved scaler to {outdir / 'checkpoints/scaler.joblib'}")
    
    # Create datasets
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
    test_dataset = KLGradeDataset(
        list(radiomics_test.keys()),
        Path(args.images_ts),
        radiomics_test,
        labels_dict=None,
        target_shape=(32, 128, 128)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Create model
    n_features = len(feature_mapping)
    logger.info(f"Model will gate {n_features} features, selecting top-{args.k}")
    
    model = JointModel(
        n_features=n_features,
        n_classes=5,
        pretrained=args.pretrained,
        k=args.k,
        warmup_threshold=args.warmup_thr_start,
        use_hard_topk=False
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Criterion
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Using weighted CrossEntropyLoss with class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using standard CrossEntropyLoss (no class weights)")
    
    # Training loop
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    train_history = []
    best_confusion_matrix_data = None
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # Determine stage
        is_warmup = epoch <= args.warmup_epochs
        stage = "warmup" if is_warmup else "hard-topk"
        
        # Train with tqdm progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [{stage}]", 
                   ncols=120, leave=False)
        train_loss, train_metrics, train_loss_components = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            args.warmup_epochs,
            args.lambda_k,
            args.lambda_k_start,
            args.warmup_thr_start,
            args.warmup_thr_end,
            args.lambda_diversity,
            progress_bar=pbar
        )
        pbar.close()
        
        # Validate
        val_loss, val_metrics, val_results, val_loss_components = validate(
            model, val_loader, criterion, device, args.lambda_k
        )
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check if best model
        is_best = val_metrics["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_confusion_matrix_data = (val_results["labels"], val_results["preds"])
        
        # Log per-epoch summary
        best_flag = " [BEST]" if is_best else ""
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} [{stage.upper()}]{best_flag} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Val QWK: {val_metrics['qwk']:.4f} | Time: {epoch_time:.1f}s"
        )
        
        # Print per-class metrics every 10 epochs
        if epoch % 10 == 0:
            logger.info("")
            logger.info("-" * 80)
            logger.info(f"PER-CLASS METRICS (Epoch {epoch})")
            logger.info("-" * 80)
            
            # Compute per-class metrics for validation set
            val_metrics_detailed = compute_metrics(
                val_results["labels"],
                val_results["preds"],
                val_results["probas"] if len(val_results["probas"]) > 0 else None,
                return_per_class=True
            )
            
            # Print per-class metrics
            logger.info(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
            logger.info("-" * 80)
            
            if "per_class" in val_metrics_detailed:
                for class_key in sorted(val_metrics_detailed["per_class"].keys()):
                    class_num = class_key.split("_")[1]
                    metrics = val_metrics_detailed["per_class"][class_key]
                    logger.info(
                        f"{class_num:<10} "
                        f"{metrics['precision']:<12.4f} "
                        f"{metrics['recall']:<12.4f} "
                        f"{metrics['f1']:<12.4f} "
                        f"{metrics['support']:<10}"
                    )
            
            # Print summary metrics
            logger.info("-" * 80)
            logger.info("SUMMARY METRICS:")
            logger.info(f"  Accuracy:        {val_metrics_detailed['accuracy']:.4f}")
            logger.info(f"  Balanced Acc:    {val_metrics_detailed['balanced_accuracy']:.4f}")
            logger.info(f"  Macro F1:        {val_metrics_detailed['macro_f1']:.4f}")
            logger.info(f"  Weighted F1:     {val_metrics_detailed['weighted_f1']:.4f}")
            if 'auc' in val_metrics_detailed:
                logger.info(f"  AUC:             {val_metrics_detailed['auc']:.4f}")
            logger.info(f"  QWK:             {val_metrics_detailed['qwk']:.4f}")
            logger.info("-" * 80)
            logger.info("")
        
        # Record history
        history_row = {
            "epoch": epoch,
            "stage": stage,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_ce_loss": train_loss_components['ce_loss'],
            "train_loss_k": train_loss_components['loss_k'],
            "train_loss_diversity": train_loss_components.get('loss_diversity', 0.0),
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_loss,
            "val_ce_loss": val_loss_components['ce_loss'],
            "val_loss_k": val_loss_components['loss_k'],
            "val_acc": val_metrics["accuracy"],
            "val_balanced_acc": val_metrics.get("balanced_accuracy", None),
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics.get("weighted_f1", None),
            "val_qwk": val_metrics["qwk"],
            "mean_p_sum": train_loss_components['mean_p_sum'],
            "time_sec": epoch_time
        }
        train_history.append(history_row)
        
        # Save CSV log
        history_df = pd.DataFrame(train_history)
        history_df.to_csv(outdir / "logs" / "metrics.csv", index=False)
        
        # Save plots after each epoch
        if len(history_df) > 0:
            plot_loss_curve(history_df, outdir / "plots" / "loss_curve.png")
            plot_loss_components(history_df, outdir / "plots" / "loss_components.png")
            plot_metrics_curve(history_df, outdir / "plots" / "metrics_curve.png")
            plot_gate_stats(history_df, outdir / "plots" / "gate_stats.png", target_k=args.k)
        
        # Save checkpoints
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": val_metrics["macro_f1"],
            "val_acc": val_metrics["accuracy"],
            "args": vars(args)
        }
        torch.save(checkpoint, outdir / "checkpoints" / "last.pth")
        
        if is_best:
            patience_counter = 0
            torch.save(checkpoint, outdir / "checkpoints" / "best.pth")
            logger.info(f"  → Best checkpoint saved (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Log stage switch
        if epoch == args.warmup_epochs:
            logger.info("=" * 80)
            logger.info("SWITCHING FROM WARMUP TO HARD TOP-K")
            logger.info("=" * 80)
    
    # Save final confusion matrix for best epoch
    if best_confusion_matrix_data is not None:
        y_true_best, y_pred_best = best_confusion_matrix_data
        plot_confusion_matrix(y_true_best, y_pred_best, 
                            outdir / "plots" / "confusion_matrix_best.png", 
                            normalize=False)
        plot_confusion_matrix(y_true_best, y_pred_best, 
                            outdir / "plots" / "confusion_matrix_best_normalized.png", 
                            normalize=True)
        logger.info(f"Saved confusion matrices for best epoch {best_epoch}")
    
    # Load best model for inference
    logger.info("Loading best model for inference...")
    checkpoint = torch.load(outdir / "checkpoints" / "best.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Inference on train and test
    logger.info("=" * 80)
    logger.info("Running inference...")
    logger.info("=" * 80)
    
    model.set_use_hard_topk(True)  # Use hard top-k for inference
    
    for split_name, dataset, loader in [("train", train_dataset, train_loader),
                                         ("test", test_dataset, test_loader)]:
        logger.info(f"Processing {split_name} set...")
        
        model.eval()
        all_case_ids = []
        all_preds = []
        all_probas = []
        all_selected_features = []
        
        with torch.no_grad():
            all_gate_variance = []
            for batch in loader:
                images = batch["image"].to(device)
                radiomics = batch["radiomics"].to(device)
                case_ids = batch["case_id"]
                
                logits, p = model(images, radiomics, return_gates=True)
                probas = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                # Diagnostic: check gate variance across batch
                gate_variance = p.var(dim=0).mean().item()  # Average variance across features
                all_gate_variance.append(gate_variance)
                
                # Get top-k features for each sample
                _, topk_indices = torch.topk(p, args.k, dim=1)
                
                for i in range(len(case_ids)):
                    case_id = case_ids[i]
                    all_case_ids.append(case_id)
                    all_preds.append(preds[i].item())
                    all_probas.append(probas[i].cpu().numpy())
                    
                    # Get selected features
                    topk_idx = topk_indices[i].cpu().numpy()
                    topk_names = [feature_mapping[int(idx)] for idx in topk_idx]
                    topk_gates = p[i][topk_indices[i]].cpu().numpy().tolist()
                    
                    all_selected_features.append({
                        "case_id": case_id,
                        "topk_indices": topk_idx.tolist(),
                        "topk_names": topk_names,
                        "topk_gates": topk_gates
                    })
        
        # Save predictions
        pred_df = pd.DataFrame({
            "case_id": all_case_ids,
            "pred_class": all_preds,
            "prob_0": [p[0] for p in all_probas],
            "prob_1": [p[1] for p in all_probas],
            "prob_2": [p[2] for p in all_probas],
            "prob_3": [p[3] for p in all_probas],
            "prob_4": [p[4] for p in all_probas]
        })
        pred_df.to_csv(outdir / f"predictions_{split_name}.csv", index=False)
        logger.info(f"Saved predictions to {outdir / f'predictions_{split_name}.csv'}")
        
        # Save selected features
        with open(outdir / f"selected_features_{split_name}.json", "w") as f:
            json.dump(all_selected_features, f, indent=2)
        logger.info(f"Saved selected features to {outdir / f'selected_features_{split_name}.json'}")
        
        # Diagnostic: check feature selection diversity
        if len(all_selected_features) > 0:
            # Count unique feature sets
            feature_sets = [tuple(f["topk_indices"]) for f in all_selected_features]
            unique_sets = len(set(feature_sets))
            total_samples = len(feature_sets)
            diversity_ratio = unique_sets / total_samples if total_samples > 0 else 0.0
            avg_gate_variance = np.mean(all_gate_variance) if all_gate_variance else 0.0
            
            logger.info(f"Feature Selection Diversity ({split_name}):")
            logger.info(f"  Unique feature sets: {unique_sets}/{total_samples} ({diversity_ratio:.2%})")
            logger.info(f"  Average gate variance: {avg_gate_variance:.6f}")
            if diversity_ratio < 0.1:
                logger.warning(f"  WARNING: Low diversity! Most patients have the same feature set.")
    
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
