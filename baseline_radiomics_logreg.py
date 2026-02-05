"""
Baseline: Radiomics-only Logistic Regression for KLGrade.

Uses the long-format radiomics CSV (case_id, roi_name, feature_name, value)
and corresponding KLGrade labels to train a multinomial logistic regression.
Outputs metrics, predictions, and saved scaler/model artifacts.

Example (train/val split):
python baseline_radiomics_logreg.py ^
  --radiomics-train-csv output_train/radiomics_results.csv ^
  --labels-train-file subInfo_train.xlsx ^
  --outdir baseline_radiomics_logreg ^
  --val-ratio 0.2 ^
  --class-weight balanced ^
  --cv-folds 5 --cv-seed 42

Example (also score test set if available):
python baseline_radiomics_logreg.py \
  --radiomics-train-csv /home/yaxi/OA_KLG_TopK/output_train/radiomics_results.csv \
  --labels-train-file /home/yaxi/OA_KLG_TopK/subInfo_train.xlsx \
  --radiomics-test-csv /home/yaxi/OA_KLG_TopK/output_test/radiomics_results.csv \
  --labels-test-file /home/yaxi/OA_KLG_TopK/subInfo_test.xlsx \
  --outdir baseline_radiomics \
  --val-ratio 0.2 \
  --class-weight balanced
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn

from data_loader import load_klgrade_labels, load_radiomics_long_format
from training_utils import compute_metrics, ordinal_cross_entropy_loss, ordinal_logits_to_pred, ordinal_logits_to_proba

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_arrays(
    radiomics_dict: Dict[str, np.ndarray],
    labels_dict: Optional[Dict[str, int]],
    case_ids: List[str],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Stack radiomics vectors; optionally return labels."""
    X = np.stack([radiomics_dict[cid] for cid in case_ids])
    y = None
    if labels_dict is not None:
        y = np.array([labels_dict[cid] for cid in case_ids], dtype=int)
    return X, y


def save_predictions(
    out_path: Path,
    case_ids: List[str],
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
):
    """Save predictions with dynamic class probabilities."""
    rows = []
    for i, cid in enumerate(case_ids):
        row = {
            "case_id": cid,
            "pred_class": int(y_pred[i]),
        }
        if y_true is not None:
            row["true_class"] = int(y_true[i])

        # Add probability columns per class (works for binary or multiclass).
        for j, cls in enumerate(classes):
            row[f"prob_{cls}"] = float(y_proba[i, j])
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"Saved predictions to {out_path}")


def map_labels_to_mode(labels: Dict[str, int], mode: str) -> Dict[str, int]:
    """Map KL grades depending on label mode."""
    if mode == "multiclass":
        return labels
    if mode == "binary_oa":
        mapped = {}
        for cid, grade in labels.items():
            if grade in (0, 1):
                mapped[cid] = 0  # no OA
            elif grade in (2, 3, 4):
                mapped[cid] = 1  # OA
            else:
                raise ValueError(f"Unsupported KLGrade {grade} for case {cid}")
        return mapped
    raise ValueError(f"Unknown label mode: {mode}")


class OrdinalLogReg(nn.Module):
    """
    Simple ordinal logistic regression (CORAL-style targets) implemented as a linear layer
    producing K-1 logits.
    """

    def __init__(self, in_dim: int, label_num_classes: int = 5):
        super().__init__()
        if label_num_classes < 3:
            raise ValueError("OrdinalLogReg requires K>=3 classes.")
        self.label_num_classes = int(label_num_classes)
        self.linear = nn.Linear(in_dim, self.label_num_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def fit_ordinal_logreg_torch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    label_num_classes: int = 5,
    class_weights: Optional[np.ndarray] = None,
    max_iter: int = 2000,
    lr: float = 1.0,
    weight_decay: float = 0.0,
    device: str = "cpu",
) -> OrdinalLogReg:
    """
    Fit ordinal logistic regression with LBFGS (good for convex-ish linear models).
    """
    X_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_train.astype(np.int64)).to(device)

    model = OrdinalLogReg(in_dim=X_t.shape[1], label_num_classes=label_num_classes).to(device)

    cw_t = None
    if class_weights is not None:
        cw_t = torch.from_numpy(class_weights.astype(np.float32)).to(device)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        logits = model(X_t)
        loss = ordinal_cross_entropy_loss(
            logits, y_t, label_num_classes=label_num_classes, class_weights=cw_t
        )
        # L2 regularization (weight decay) for linear models
        if weight_decay > 0:
            l2 = 0.0
            for p in model.parameters():
                l2 = l2 + torch.sum(p ** 2)
            loss = loss + weight_decay * l2
        loss.backward()
        return loss

    optimizer.step(closure)
    return model


@torch.no_grad()
def predict_ordinal(
    model: OrdinalLogReg,
    X: np.ndarray,
    label_num_classes: int = 5,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        preds: [N] int
        probas: [N,K] float
    """
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    logits = model(X_t)
    probas = ordinal_logits_to_proba(logits, label_num_classes=label_num_classes).cpu().numpy()
    preds = ordinal_logits_to_pred(logits).cpu().numpy()
    return preds, probas


def main():
    parser = argparse.ArgumentParser(
        description="Baseline radiomics-only multinomial logistic regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--radiomics-train-csv", type=str, required=True)
    parser.add_argument("--labels-train-file", type=str, required=True)
    parser.add_argument("--radiomics-test-csv", type=str, default=None)
    parser.add_argument("--labels-test-file", type=str, default=None)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-weight",
        type=str,
        default=None,
        choices=[None, "balanced", "balanced_subsample"],
        help="Class weight strategy for LogisticRegression"
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--ordinal-weight-decay", type=float, default=0.0,
                        help="L2 regularization for torch ordinal logreg (multiclass only).")
    parser.add_argument("--ordinal-lbfgs-lr", type=float, default=1.0,
                        help="LBFGS learning rate for torch ordinal logreg (multiclass only).")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="multiclass",
        choices=["multiclass", "binary_oa"],
        help="Use 'binary_oa' to merge KL 0/1 -> 0 (no OA) and 2/3/4 -> 1 (OA); default keeps 5-class labels.",
    )
    parser.add_argument("--cv-folds", type=int, default=1,
                        help="Number of CV folds for mean±std report (requires labels).")
    parser.add_argument("--cv-seed", type=int, default=42,
                        help="Random seed for CV split.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load radiomics and labels
    rad_train, roi_names, feature_names, missing_stats = load_radiomics_long_format(
        Path(args.radiomics_train_csv)
    )
    labels_train_raw = load_klgrade_labels(Path(args.labels_train_file))
    labels_train = map_labels_to_mode(labels_train_raw, args.label_mode)
    logger.info(f"Radiomics missing stats (train): {missing_stats}")
    logger.info(f"Label mode: {args.label_mode}")

    # Align cases with labels
    train_case_ids = sorted(set(rad_train.keys()) & set(labels_train.keys()))
    if len(train_case_ids) == 0:
        raise ValueError("No overlapping cases between radiomics and labels (train).")
    logger.info(f"Train cases used: {len(train_case_ids)}")

    X_all, y_all = build_arrays(rad_train, labels_train, train_case_ids)
    unique_classes = np.unique(y_all)
    n_classes = len(unique_classes)
    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes for training, got {unique_classes.tolist()}")

    # Train/val split
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_all,
        y_all,
        train_case_ids,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=y_all,
    )

    # Scale
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Model
    is_multiclass_ordinal = (args.label_mode == "multiclass") and (n_classes == 5)
    if is_multiclass_ordinal:
        classes_full = np.array([0, 1, 2, 3, 4])
        cw = None
        if args.class_weight is not None:
            cw = compute_class_weight(
                class_weight=args.class_weight,
                classes=classes_full,
                y=y_train,
            ).astype(np.float32)
            logger.info(f"Using class weights (ordinal): {cw}")
        clf = fit_ordinal_logreg_torch(
            X_train_s,
            y_train,
            label_num_classes=5,
            class_weights=cw,
            max_iter=args.max_iter,
            lr=args.ordinal_lbfgs_lr,
            weight_decay=args.ordinal_weight_decay,
        )
    else:
        # Binary OA: keep standard sklearn logistic regression.
        clf = LogisticRegression(
            multi_class="multinomial" if n_classes > 2 else "auto",
            solver="lbfgs",
            max_iter=args.max_iter,
            class_weight=args.class_weight,
            n_jobs=-1,
        )
        clf.fit(X_train_s, y_train)

    # Train metrics
    if is_multiclass_ordinal:
        train_pred, train_proba = predict_ordinal(clf, X_train_s, label_num_classes=5)
        classes_for_save = np.array([0, 1, 2, 3, 4])
    else:
        train_pred = clf.predict(X_train_s)
        train_proba = clf.predict_proba(X_train_s)
        classes_for_save = clf.classes_
    train_metrics = compute_metrics(y_train, train_pred, train_proba, return_per_class=True)
    with open(outdir / "metrics_train.json", "w") as f:
        json.dump(train_metrics, f, indent=2)
    logger.info(f"Train metrics saved: {outdir / 'metrics_train.json'}")

    # Val metrics
    if is_multiclass_ordinal:
        val_pred, val_proba = predict_ordinal(clf, X_val_s, label_num_classes=5)
    else:
        val_pred = clf.predict(X_val_s)
        val_proba = clf.predict_proba(X_val_s)
    val_metrics = compute_metrics(y_val, val_pred, val_proba, return_per_class=True)
    with open(outdir / "metrics_val.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    logger.info(f"Val metrics saved: {outdir / 'metrics_val.json'}")

    # Save predictions
    save_predictions(outdir / "predictions_train.csv", ids_train, y_train, train_pred, train_proba, classes_for_save)
    save_predictions(outdir / "predictions_val.csv", ids_val, y_val, val_pred, val_proba, classes_for_save)

    # Optional test set
    if args.radiomics_test_csv:
        rad_test, roi_names_t, feature_names_t, missing_stats_t = load_radiomics_long_format(
            Path(args.radiomics_test_csv),
            expected_rois=roi_names,
            expected_features=feature_names,
        )
        logger.info(f"Radiomics missing stats (test): {missing_stats_t}")
        test_case_ids = sorted(rad_test.keys())

        labels_test = None
        if args.labels_test_file:
            labels_test_raw = load_klgrade_labels(Path(args.labels_test_file))
            labels_test = map_labels_to_mode(labels_test_raw, args.label_mode)
            test_case_ids = [cid for cid in test_case_ids if cid in labels_test]
            logger.info(f"Test cases with labels: {len(test_case_ids)}")

        if len(test_case_ids) > 0:
            X_test, y_test = build_arrays(rad_test, labels_test, test_case_ids)
            X_test_s = scaler.transform(X_test)
            if is_multiclass_ordinal:
                test_pred, test_proba = predict_ordinal(clf, X_test_s, label_num_classes=5)
            else:
                test_pred = clf.predict(X_test_s)
                test_proba = clf.predict_proba(X_test_s)
            save_predictions(outdir / "predictions_test.csv", test_case_ids, y_test, test_pred, test_proba, classes_for_save)

            if labels_test is not None:
                test_metrics = compute_metrics(y_test, test_pred, test_proba, return_per_class=True)
                with open(outdir / "metrics_test.json", "w") as f:
                    json.dump(test_metrics, f, indent=2)
                logger.info(f"Test metrics saved: {outdir / 'metrics_test.json'}")
        else:
            logger.warning("No test cases to evaluate.")

    # CV summary (mean ± std) aligned with evaluate_image_baseline.py
    def cv_summary(X: np.ndarray, y: np.ndarray, tag: str):
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.cv_seed)
        fold_metrics = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_test_fold = X[test_idx]
            y_test_fold = y[test_idx]

            # Fit scaler and model per fold to avoid leakage
            scaler_fold = StandardScaler()
            scaler_fold.fit(X_train_fold)
            X_train_fold_s = scaler_fold.transform(X_train_fold)
            X_test_fold_s = scaler_fold.transform(X_test_fold)

            fold_is_ordinal = (args.label_mode == "multiclass") and (len(np.unique(y_train_fold)) == 5)
            if fold_is_ordinal:
                classes_full = np.array([0, 1, 2, 3, 4])
                cw = None
                if args.class_weight is not None:
                    cw = compute_class_weight(
                        class_weight=args.class_weight,
                        classes=classes_full,
                        y=y_train_fold,
                    ).astype(np.float32)
                clf_fold = fit_ordinal_logreg_torch(
                    X_train_fold_s,
                    y_train_fold,
                    label_num_classes=5,
                    class_weights=cw,
                    max_iter=args.max_iter,
                    lr=args.ordinal_lbfgs_lr,
                    weight_decay=args.ordinal_weight_decay,
                )
                fold_pred, fold_proba = predict_ordinal(clf_fold, X_test_fold_s, label_num_classes=5)
            else:
                clf_fold = LogisticRegression(
                    multi_class="multinomial" if len(np.unique(y_train_fold)) > 2 else "auto",
                    solver="lbfgs",
                    max_iter=args.max_iter,
                    class_weight=args.class_weight,
                    n_jobs=-1,
                )
                clf_fold.fit(X_train_fold_s, y_train_fold)
                fold_pred = clf_fold.predict(X_test_fold_s)
                fold_proba = clf_fold.predict_proba(X_test_fold_s)
            metrics = compute_metrics(y_test_fold, fold_pred, fold_proba, return_per_class=True)
            fold_metrics.append(metrics)
            logger.info(
                f"CV({tag}) Fold {fold_idx}/{args.cv_folds} | "
                f"Acc={metrics['accuracy']:.4f} Macro-F1={metrics['macro_f1']:.4f} QWK={metrics['qwk']:.4f}"
            )

        keys = sorted({k for m in fold_metrics for k in m.keys() if k != "per_class"})
        summary = {"folds": fold_metrics, "mean": {}, "std": {}, "per_class_mean": {}, "per_class_std": {}}
        for key in keys:
            values = [m[key] for m in fold_metrics if key in m]
            summary["mean"][key] = float(np.mean(values))
            summary["std"][key] = float(np.std(values))
            logger.info(f"CV({tag}) {key}: {summary['mean'][key]:.4f} ± {summary['std'][key]:.4f}")

        class_keys = sorted({
            cls_key
            for m in fold_metrics
            for cls_key in (m.get("per_class") or {}).keys()
        })
        for cls_key in class_keys:
            metric_names = sorted({
                metric_name
                for m in fold_metrics
                for metric_name in (m.get("per_class", {}).get(cls_key, {}) or {}).keys()
            })
            summary["per_class_mean"][cls_key] = {}
            summary["per_class_std"][cls_key] = {}
            for metric_name in metric_names:
                values = [
                    m["per_class"][cls_key][metric_name]
                    for m in fold_metrics
                    if cls_key in m.get("per_class", {}) and metric_name in m["per_class"][cls_key]
                ]
                if not values:
                    continue
                summary["per_class_mean"][cls_key][metric_name] = float(np.mean(values))
                summary["per_class_std"][cls_key][metric_name] = float(np.std(values))
                logger.info(
                    f"CV({tag}) {cls_key}.{metric_name}: "
                    f"{summary['per_class_mean'][cls_key][metric_name]:.4f} ± "
                    f"{summary['per_class_std'][cls_key][metric_name]:.4f}"
                )

        cv_path = outdir / f"metrics_eval_cv_{tag}.json"
        with open(cv_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"CV metrics saved: {cv_path}")

    if args.cv_folds and args.cv_folds > 1:
        if args.radiomics_test_csv and "X_test" in locals() and "y_test" in locals() and y_test is not None:
            cv_summary(X_test, y_test, "test")
        else:
            cv_summary(X_val, y_val, "val")

    # Save artifacts
    joblib.dump(scaler, outdir / "scaler.joblib")
    if is_multiclass_ordinal:
        torch.save(
            {
                "state_dict": clf.state_dict(),
                "label_num_classes": 5,
                "ordinal": True,
                "in_dim": int(X_train_s.shape[1]),
            },
            outdir / "ordinal_logreg_model.pt",
        )
        logger.info(f"Saved scaler + ordinal model to {outdir}")
    else:
        joblib.dump(clf, outdir / "logreg_model.joblib")
        logger.info(f"Saved scaler and model to {outdir}")


if __name__ == "__main__":
    main()

