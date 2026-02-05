"""
Evaluate an image-only baseline model (train_image_baseline.py).

Saves:
- predictions_eval.csv (case_id, pred_class, prob_0..prob_4)
- metrics_eval.json (if labels provided)

python evaluate_image_baseline.py \
  --model-dir "training_logs_image_baseline_b"   \
  --images-dir "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTs" \
  --radiomics-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/output_test/radiomics_results_wide.csv" \
  --klgrade-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/subInfo_test.xlsx" \
  --arch resnet3d \
  --cv-folds 5 --cv-seed 42 \
  --label-mode binary_oa \
  --device cuda:0

python evaluate_image_baseline.py \
  --model-dir "training_logs_image_baseline_ordinal_eff_b" \
  --images-dir "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTs" \
  --radiomics-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/output_test/radiomics_results_wide.csv" \
  --klgrade-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/subInfo_test.xlsx" \
  --arch efficientnet_b0 \
  --cv-folds 5 --cv-seed 42 \
  --device cuda:0 \
  --label-mode multiclass \

"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from data_loader import KLGradeDataset, load_klgrade_labels, load_radiomics_wide_format
from training_utils import compute_metrics, ordinal_logits_to_proba, ordinal_logits_to_pred
from train_image_baseline import ImageOnlyModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def infer_out_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    for key in state_dict:
        if key.endswith("classifier.weight"):
            return state_dict[key].shape[0]
        if key.endswith("backbone.classifier.weight"):
            return state_dict[key].shape[0]
        if key.endswith("backbone.classifier.1.weight"):
            return state_dict[key].shape[0]
        if key.endswith("backbone.fc.weight"):
            return state_dict[key].shape[0]
    raise KeyError("Could not infer num_classes from checkpoint state_dict.")


def _safe_torch_load(path: Path, device: torch.device):
    """
    Backward compatible torch.load wrapper.
    `weights_only` is only supported in newer PyTorch versions.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _has_image(images_dir: Path, case_id: str) -> bool:
    for name in (
        f"{case_id}_0000.nii.gz",
        f"{case_id}_0000.nii",
        f"{case_id}.nii.gz",
        f"{case_id}.nii",
    ):
        if (images_dir / name).exists():
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate image-only baseline model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-dir", type=str, required=True, help="Training output dir")
    parser.add_argument("--images-dir", type=str, required=True, help="Images directory")
    parser.add_argument("--radiomics-csv", type=str, required=True, help="Radiomics CSV (wide format)")
    parser.add_argument("--klgrade-csv", type=str, default=None, help="Optional KLGrade labels")
    parser.add_argument("--outdir", type=str, default=None, help="Output dir (defaults to model-dir)")
    parser.add_argument("--checkpoint-name", type=str, default="best.pth", help="Checkpoint name")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--target-shape", type=int, nargs=3, default=[32, 128, 128], help="D H W")
    parser.add_argument("--arch", type=str, default=None,
                        choices=["resnet3d", "efficientnet_b0", "efficientnet3d_b0"],
                        help="Override model arch (default: from checkpoint)")
    parser.add_argument("--label-mode", type=str, default=None,
                        choices=["multiclass", "binary_oa"],
                        help="Override label mode (default: from checkpoint)")
    parser.add_argument("--cv-folds", type=int, default=1,
                        help="Number of CV folds for mean±std report (requires labels)")
    parser.add_argument("--cv-seed", type=int, default=42,
                        help="Random seed for CV split")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    outdir = Path(args.outdir) if args.outdir else model_dir
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = model_dir / "checkpoints"
    ckpt_path = ckpt_dir / args.checkpoint_name
    if not ckpt_path.exists():
        fallback = ckpt_dir / "last.pth"
        if fallback.exists():
            logger.warning(f"{args.checkpoint_name} not found; using {fallback.name}")
            ckpt_path = fallback
        else:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    checkpoint = _safe_torch_load(ckpt_path, device)
    ckpt_args = checkpoint.get("args", {})

    out_dim = checkpoint.get("logit_dim")
    if out_dim is None:
        out_dim = infer_out_dim(checkpoint["model_state_dict"])

    ordinal = checkpoint.get("ordinal")
    if ordinal is None:
        ordinal = (out_dim == 4)

    label_num_classes = checkpoint.get("label_num_classes")
    if label_num_classes is None:
        # Backward compatible
        if ordinal:
            label_num_classes = 5
        else:
            label_num_classes = checkpoint.get("num_classes", out_dim)

    arch = args.arch or checkpoint.get("arch") or ckpt_args.get("arch") or "resnet3d"
    label_mode = args.label_mode or ckpt_args.get("label_mode")
    if label_mode is None:
        label_mode = "binary_oa" if int(label_num_classes) == 2 else "multiclass"
    logger.info(
        f"Arch: {arch} | label_mode: {label_mode} | label_num_classes={label_num_classes} | "
        f"ordinal={bool(ordinal)} | out_dim={out_dim}"
    )

    # Load radiomics for case_ids only
    radiomics, _, _, _ = load_radiomics_wide_format(Path(args.radiomics_csv))

    labels_dict = None
    if args.klgrade_csv:
        labels_raw = load_klgrade_labels(Path(args.klgrade_csv))
        if label_mode == "binary_oa":
            labels_dict = {cid: (0 if lab <= 1 else 1) for cid, lab in labels_raw.items()}
        else:
            labels_dict = labels_raw
        logger.info(f"Loaded {len(labels_dict)} labels")

    case_ids = list(radiomics.keys())
    if labels_dict:
        case_ids = [cid for cid in case_ids if cid in labels_dict]
        logger.info(f"Cases with labels: {len(case_ids)}")
    else:
        logger.info(f"Cases for inference: {len(case_ids)}")

    # Drop cases without corresponding image files (radiomics may contain extra rows).
    images_dir = Path(args.images_dir)
    before = len(case_ids)
    case_ids = [cid for cid in case_ids if _has_image(images_dir, cid)]
    dropped = before - len(case_ids)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} cases without matching image files in {images_dir}.")

    model = ImageOnlyModel(arch=arch, out_dim=int(out_dim), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def run_inference(eval_case_ids):
        dataset = KLGradeDataset(
            eval_case_ids,
            Path(args.images_dir),
            radiomics,
            labels_dict=labels_dict,
            target_shape=tuple(args.target_shape)
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == "cuda" else False
        )

        case_list = []
        preds_list = []
        probas_list = []
        labels_list = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                logits = model(images)

                # Probabilities + predictions
                if bool(ordinal) and int(label_num_classes) == 5 and logits.shape[1] == 4:
                    probs_t = ordinal_logits_to_proba(logits, label_num_classes=5)
                    preds_t = ordinal_logits_to_pred(logits)
                    probas = probs_t.cpu().numpy()
                    preds = preds_t.cpu().numpy()
                elif logits.shape[1] == 1:
                    # Binary with single logit: sigmoid -> [p0, p1]
                    prob_pos = torch.sigmoid(logits).cpu().numpy()
                    probas = np.concatenate([1.0 - prob_pos, prob_pos], axis=1)
                    preds = (prob_pos[:, 0] > 0.5).astype(int)
                else:
                    probas = F.softmax(logits, dim=1).cpu().numpy()
                    preds = logits.argmax(dim=1).cpu().numpy()

                for i, cid in enumerate(batch["case_id"]):
                    case_list.append(cid)
                    preds_list.append(int(preds[i]))
                    probas_list.append(probas[i])
                    if labels_dict:
                        labels_list.append(int(batch["label"][i].item()))

        return case_list, preds_list, probas_list, labels_list

    all_case_ids, all_preds, all_probas, all_labels = run_inference(case_ids)

    # Save predictions
    pred_rows = []
    for cid, pred, proba in zip(all_case_ids, all_preds, all_probas):
        row = {"case_id": cid, "pred_class": pred}
        for cls_idx in range(proba.shape[0]):
            row[f"prob_{cls_idx}"] = float(proba[cls_idx])
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)
    pred_path = outdir / "predictions_eval.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions to {pred_path}")

    # Metrics
    if labels_dict and all_labels:
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_proba = np.vstack(all_probas)
        metrics = compute_metrics(y_true, y_pred, y_proba, return_per_class=True)
        metrics_path = outdir / "metrics_eval.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(
            f"Metrics saved to {metrics_path} | "
            f"Acc={metrics['accuracy']:.4f} Macro-F1={metrics['macro_f1']:.4f} QWK={metrics['qwk']:.4f}"
        )
    else:
        logger.info("Labels not provided; skipped metrics.")

    # Cross-validation summary (mean ± std)
    if labels_dict and args.cv_folds and args.cv_folds > 1:
        labels_array = np.array([labels_dict[cid] for cid in case_ids])
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.cv_seed)
        fold_metrics = []
        for fold_idx, (_, test_idx) in enumerate(skf.split(case_ids, labels_array), start=1):
            fold_case_ids = [case_ids[i] for i in test_idx]
            _, fold_preds, fold_probas, fold_labels = run_inference(fold_case_ids)
            metrics = compute_metrics(
                np.array(fold_labels),
                np.array(fold_preds),
                np.vstack(fold_probas),
                return_per_class=True
            )
            fold_metrics.append(metrics)
            logger.info(
                f"CV Fold {fold_idx}/{args.cv_folds} | "
                f"Acc={metrics['accuracy']:.4f} Macro-F1={metrics['macro_f1']:.4f} QWK={metrics['qwk']:.4f}"
            )

        keys = sorted({k for m in fold_metrics for k in m.keys() if k != "per_class"})
        summary = {"folds": fold_metrics, "mean": {}, "std": {}, "per_class_mean": {}, "per_class_std": {}}
        for key in keys:
            values = [m[key] for m in fold_metrics if key in m]
            summary["mean"][key] = float(np.mean(values))
            summary["std"][key] = float(np.std(values))
            logger.info(f"CV {key}: {summary['mean'][key]:.4f} ± {summary['std'][key]:.4f}")

        # Per-class mean ± std
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
                    f"CV {cls_key}.{metric_name}: "
                    f"{summary['per_class_mean'][cls_key][metric_name]:.4f} ± "
                    f"{summary['per_class_std'][cls_key][metric_name]:.4f}"
                )

        cv_path = outdir / "metrics_eval_cv.json"
        with open(cv_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"CV metrics saved to {cv_path}")


if __name__ == "__main__":
    main()

