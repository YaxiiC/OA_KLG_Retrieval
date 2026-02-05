r"""
Evaluate a trained JointScoringModel on images + radiomics (wide format).

Saves:
- predictions_eval.csv (case_id, pred_class, prob_0..prob_4)
- selected_subsets_eval.json (TopM subset indices + names per case)
- metrics_eval.json (if labels provided)


python evaluate_joint_scoring_kl.py ^
  --model-dir training_logs_joint_scoring_k30 ^
  --images-dir "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTs" ^
  --radiomics-csv "C:\Users\chris\MICCAI2026\YaxiiC-OA_KLG_Retrieval\output_test\radiomics_results_wide.csv" ^
  --klgrade-csv "C:\Users\chris\MICCAI2026\YaxiiC-OA_KLG_Retrieval\subInfo_test.xlsx" ^
  --device cuda:0

python evaluate_joint_scoring_kl.py \
    --model-dir training_logs_joint_scoring_k15 \
    --images-dir "home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTs" \
    --radiomics-csv "home/yaxi/YaxiiC-OA_KLG_Retrieval/output_test/radiomics_results_wide.csv" \
    --klgrade-csv "home/yaxi/YaxiiC-OA_KLG_Retrieval/subInfo_test.xlsx" \
    --device cuda:0

python evaluate_joint_scoring_kl.py \
  --model-dir training_logs_joint_scoring_k30 \
  --images-dir "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTs" \
  --radiomics-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/output_test/radiomics_results_wide.csv" \
  --klgrade-csv "/home/yaxi/YaxiiC-OA_KLG_Retrieval/subInfo_test.xlsx" \
  --retrieval budgeted \
  --recall 4000 --keep 100 --seeds 5 --mut-per-seed 200 --final-top 10 \
  --cv-folds 5 --cv-seed 42 \
  --device cuda:0 \
  --label-mode binary_oa
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from data_loader import KLGradeDataset, load_klgrade_labels, load_radiomics_wide_format
from models import JointScoringModel
from training_utils import (
    build_subset_tensors,
    compute_metrics,
    sample_random_subsets,
    ordinal_logits_to_proba,
    ordinal_logits_to_pred,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_feature_names(feature_names_path: Path) -> Optional[List[str]]:
    if not feature_names_path.exists():
        return None
    with open(feature_names_path, "r") as f:
        mapping = json.load(f)
    return [mapping[str(i)] for i in sorted(map(int, mapping.keys()))]


def mutate_subset(subset: np.ndarray, num_features: int, rng: random.Random) -> np.ndarray:
    """
    1-swap or 2-swap mutation on subset indices (used in budgeted retrieval).
    """
    subset = subset.copy()
    k = len(subset)
    swap_count = 1 if rng.random() < 0.7 else 2
    for _ in range(swap_count):
        pos = rng.randrange(k)
        new_idx = rng.randrange(num_features)
        subset[pos] = new_idx
    return subset


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate JointScoringModel on images + radiomics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-dir", type=str, required=True, help="Training output dir")
    parser.add_argument("--images-dir", type=str, required=True, help="Images directory")
    parser.add_argument("--radiomics-csv", type=str, required=True, help="Radiomics CSV (wide format)")
    parser.add_argument("--klgrade-csv", type=str, default=None, help="Optional KLGrade labels")
    parser.add_argument("--outdir", type=str, default=None, help="Output dir (defaults to model-dir)")
    parser.add_argument("--checkpoint-name", type=str, default="best.pth", help="Checkpoint name")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--target-shape", type=int, nargs=3, default=[32, 128, 128], help="D H W")
    parser.add_argument("--pool-size", type=int, default=None, help="Candidate pool size")
    parser.add_argument("--top-m", type=int, default=None, help="TopM subsets for ensemble")
    parser.add_argument("--k", type=int, default=None, help="Subset size K")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--retrieval", type=str, default="random",
                        choices=["random", "budgeted"],
                        help="Subset retrieval strategy for evaluation")
    parser.add_argument("--recall", type=int, default=4000, help="Budgeted recall size")
    parser.add_argument("--keep", type=int, default=50, help="Budgeted keep after recall")
    parser.add_argument("--seeds", type=int, default=5, help="Budgeted local search seeds")
    parser.add_argument("--mut-per-seed", type=int, default=200, help="Budgeted mutations per seed")
    parser.add_argument("--final-top", type=int, default=10, help="Final top subsets for ensemble")
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

    feature_names_path = ckpt_dir / "feature_names.json"
    scaler_path = ckpt_dir / "scaler.joblib"

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})

    k = args.k if args.k is not None else ckpt_args.get("k", 15)
    pool_size = args.pool_size if args.pool_size is not None else ckpt_args.get("pool_size", 320)
    top_m = args.top_m if args.top_m is not None else ckpt_args.get("top_m", 4)

    def infer_out_dim(state_dict: Dict[str, torch.Tensor]) -> int:
        for key in state_dict:
            if key.endswith("classifier.classifier.weight"):
                return state_dict[key].shape[0]
        for key in state_dict:
            if key.endswith("classifier.mlp.3.weight"):
                return state_dict[key].shape[0]
        for key in state_dict:
            if key.endswith("classifier.mlp.2.weight") or key.endswith("classifier.mlp.1.weight"):
                return state_dict[key].shape[0]
        raise KeyError("Could not infer num_classes from checkpoint state_dict.")

    out_dim = checkpoint.get("logit_dim")
    if out_dim is None:
        out_dim = infer_out_dim(checkpoint["model_state_dict"])

    ordinal = checkpoint.get("ordinal")
    if ordinal is None:
        ordinal = (out_dim == 4)

    label_num_classes = checkpoint.get("label_num_classes")
    if label_num_classes is None:
        if ordinal:
            label_num_classes = 5
        else:
            label_num_classes = checkpoint.get("num_classes", out_dim)

    label_mode = args.label_mode or ckpt_args.get("label_mode")
    if label_mode is None:
        label_mode = "binary_oa" if int(label_num_classes) == 2 else "multiclass"
    def infer_emb_dim(state_dict: Dict[str, torch.Tensor]) -> int:
        for key in state_dict:
            if key.endswith("token_set_encoder.token_mlp.0.weight"):
                return state_dict[key].shape[0]
        for key in state_dict:
            if key.endswith("scorer.mlp.0.weight"):
                return state_dict[key].shape[0]
        raise KeyError("Could not infer emb_dim from checkpoint state_dict.")

    emb_dim = ckpt_args.get("emb_dim")
    if emb_dim is None:
        emb_dim = infer_emb_dim(checkpoint["model_state_dict"])

    logger.info(
        f"Label mode: {label_mode} | label_num_classes={label_num_classes} | "
        f"ordinal={bool(ordinal)} | out_dim={out_dim} | emb_dim={emb_dim}"
    )

    # Load radiomics (wide format)
    radiomics, feature_names, _, feature_meta = load_radiomics_wide_format(
        Path(args.radiomics_csv)
    )

    # Validate feature order with training mapping if available
    trained_feature_names = load_feature_names(feature_names_path)
    if trained_feature_names and len(trained_feature_names) != len(feature_names):
        logger.warning(
            "Feature count mismatch (train vs eval): "
            f"{len(trained_feature_names)} vs {len(feature_names)}"
        )

    # Apply scaler if available
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        for cid in list(radiomics.keys()):
            radiomics[cid] = scaler.transform(radiomics[cid].reshape(1, -1))[0]
    else:
        logger.warning("Scaler not found; proceeding without feature scaling.")

    labels_dict = None
    if args.klgrade_csv:
        labels_dict_raw = load_klgrade_labels(Path(args.klgrade_csv))
        if label_mode == "binary_oa":
            labels_dict = {cid: (0 if lab <= 1 else 1) for cid, lab in labels_dict_raw.items()}
        else:
            labels_dict = labels_dict_raw
        logger.info(f"Loaded {len(labels_dict)} labels")

    case_ids = list(radiomics.keys())
    if labels_dict:
        case_ids = [cid for cid in case_ids if cid in labels_dict]
        logger.info(f"Cases with labels: {len(case_ids)}")
    else:
        logger.info(f"Cases for inference: {len(case_ids)}")

    dataset = KLGradeDataset(
        case_ids,
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

    num_features = checkpoint.get("num_features", len(feature_names))
    feature_to_roi = np.array(feature_meta["feature_to_roi"], dtype=np.int64)
    feature_to_type = np.array(feature_meta["feature_to_type"], dtype=np.int64)
    num_rois = len(feature_meta["roi_names"])
    num_types = len(feature_meta["type_names"])

    model = JointScoringModel(
        num_features=num_features,
        num_rois=num_rois,
        num_types=num_types,
        emb_dim=emb_dim,
        num_classes=int(out_dim)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rng = random.Random(args.seed)
    predictions = []
    selected_subsets = []
    all_labels = []
    all_preds = []
    all_probas = []

    def run_metrics_on_case_ids(eval_case_ids: List[str]) -> Dict[str, float]:
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
        fold_preds = []
        fold_labels = []
        fold_probas = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                case_ids_batch = batch["case_id"]

                for i, case_id in enumerate(case_ids_batch):
                    image = images[i:i + 1]
                    radiomics_vec = radiomics[case_id]

                    if args.retrieval == "budgeted":
                        recall_subsets = sample_random_subsets(num_features, k, args.recall, rng)
                        recall_scores = []
                        for subset in recall_subsets:
                            f_ids, r_ids, t_ids, vals = build_subset_tensors(
                                radiomics_vec, subset, feature_to_roi, feature_to_type, device
                            )
                            score = model.score_subset(
                                image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                            )
                            recall_scores.append(score.item())
                        recall_scores = np.array(recall_scores)
                        keep_indices = np.argsort(recall_scores)[-args.keep:]
                        keep_subsets = recall_subsets[keep_indices]

                        seed_indices = np.argsort(recall_scores[keep_indices])[-args.seeds:]
                        seed_subsets = keep_subsets[seed_indices]
                        mutated_subsets = []
                        for seed in seed_subsets:
                            for _ in range(args.mut_per_seed):
                                mutated_subsets.append(mutate_subset(seed, num_features, rng))
                        if mutated_subsets:
                            mutated_subsets = np.array(mutated_subsets, dtype=np.int64)
                            keep_subsets = np.concatenate([keep_subsets, mutated_subsets], axis=0)

                        final_scores = []
                        for subset in keep_subsets:
                            f_ids, r_ids, t_ids, vals = build_subset_tensors(
                                radiomics_vec, subset, feature_to_roi, feature_to_type, device
                            )
                            score = model.score_subset(
                                image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                            )
                            final_scores.append(score.item())
                        final_scores = np.array(final_scores)
                        final_indices = np.argsort(final_scores)[-args.final_top:]
                        topm_subsets = keep_subsets[final_indices]
                    else:
                        candidate_subsets = sample_random_subsets(num_features, k, pool_size, rng)
                        subset_scores = []
                        for subset in candidate_subsets:
                            f_ids, r_ids, t_ids, vals = build_subset_tensors(
                                radiomics_vec, subset, feature_to_roi, feature_to_type, device
                            )
                            score = model.score_subset(
                                image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                            )
                            subset_scores.append(score.item())
                        subset_scores = np.array(subset_scores)
                        topm_indices = np.argsort(subset_scores)[-top_m:]
                        topm_subsets = candidate_subsets[topm_indices]

                    logits_list = []
                    for subset in topm_subsets:
                        f_ids, r_ids, t_ids, vals = build_subset_tensors(
                            radiomics_vec, subset, feature_to_roi, feature_to_type, device
                        )
                        logits = model.classify_subset(
                            image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                        )
                        logits_list.append(logits)

                    logits_ensemble = torch.mean(torch.cat(logits_list, dim=0), dim=0, keepdim=True)
                    if bool(ordinal) and int(label_num_classes) == 5 and logits_ensemble.shape[1] == 4:
                        probs_t = ordinal_logits_to_proba(logits_ensemble, label_num_classes=5)
                        pred_t = ordinal_logits_to_pred(logits_ensemble)
                        probas = probs_t.cpu().numpy()[0]
                        pred_class = int(pred_t.cpu().numpy()[0])
                    elif logits_ensemble.shape[1] == 1:
                        # Binary with single logit: sigmoid -> [p0, p1]
                        prob_pos = torch.sigmoid(logits_ensemble).cpu().numpy()[0:1]
                        probas = np.concatenate([1.0 - prob_pos, prob_pos], axis=1)[0]
                        pred_class = int(probas[1] > 0.5)
                    else:
                        probas = F.softmax(logits_ensemble, dim=1).cpu().numpy()[0]
                        pred_class = int(np.argmax(probas))

                    fold_preds.append(pred_class)
                    fold_probas.append(probas)
                    fold_labels.append(int(batch["label"][i].item()))

        metrics = compute_metrics(
            np.array(fold_labels),
            np.array(fold_preds),
            np.vstack(fold_probas),
            return_per_class=True
        )
        return metrics

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            case_ids_batch = batch["case_id"]

            for i, case_id in enumerate(case_ids_batch):
                image = images[i:i + 1]
                radiomics_vec = radiomics[case_id]

                if args.retrieval == "budgeted":
                    # Recall
                    recall_subsets = sample_random_subsets(num_features, k, args.recall, rng)
                    recall_scores = []
                    for subset in recall_subsets:
                        f_ids, r_ids, t_ids, vals = build_subset_tensors(
                            radiomics_vec, subset, feature_to_roi, feature_to_type, device
                        )
                        score = model.score_subset(
                            image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                        )
                        recall_scores.append(score.item())
                    recall_scores = np.array(recall_scores)
                    keep_indices = np.argsort(recall_scores)[-args.keep:]
                    keep_subsets = recall_subsets[keep_indices]

                    # Local search (mutations)
                    seed_indices = np.argsort(recall_scores[keep_indices])[-args.seeds:]
                    seed_subsets = keep_subsets[seed_indices]
                    mutated_subsets = []
                    for seed in seed_subsets:
                        for _ in range(args.mut_per_seed):
                            mutated_subsets.append(mutate_subset(seed, num_features, rng))
                    if mutated_subsets:
                        mutated_subsets = np.array(mutated_subsets, dtype=np.int64)
                        keep_subsets = np.concatenate([keep_subsets, mutated_subsets], axis=0)

                    # Final ranking
                    final_scores = []
                    for subset in keep_subsets:
                        f_ids, r_ids, t_ids, vals = build_subset_tensors(
                            radiomics_vec, subset, feature_to_roi, feature_to_type, device
                        )
                        score = model.score_subset(
                            image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                        )
                        final_scores.append(score.item())
                    final_scores = np.array(final_scores)
                    final_indices = np.argsort(final_scores)[-args.final_top:]
                    topm_subsets = keep_subsets[final_indices]
                    topm_scores = final_scores[final_indices]
                else:
                    candidate_subsets = sample_random_subsets(num_features, k, pool_size, rng)
                    subset_scores = []
                    for subset in candidate_subsets:
                        f_ids, r_ids, t_ids, vals = build_subset_tensors(
                            radiomics_vec, subset, feature_to_roi, feature_to_type, device
                        )
                        score = model.score_subset(
                            image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                        )
                        subset_scores.append(score.item())

                    subset_scores = np.array(subset_scores)
                    topm_indices = np.argsort(subset_scores)[-top_m:]
                    topm_subsets = candidate_subsets[topm_indices]
                    topm_scores = subset_scores[topm_indices]

                logits_list = []
                for subset in topm_subsets:
                    f_ids, r_ids, t_ids, vals = build_subset_tensors(
                        radiomics_vec, subset, feature_to_roi, feature_to_type, device
                    )
                    logits = model.classify_subset(
                        image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                    )
                    logits_list.append(logits)

                logits_ensemble = torch.mean(torch.cat(logits_list, dim=0), dim=0, keepdim=True)
                if bool(ordinal) and int(label_num_classes) == 5 and logits_ensemble.shape[1] == 4:
                    probs_t = ordinal_logits_to_proba(logits_ensemble, label_num_classes=5)
                    pred_t = ordinal_logits_to_pred(logits_ensemble)
                    probas = probs_t.cpu().numpy()[0]
                    pred_class = int(pred_t.cpu().numpy()[0])
                elif logits_ensemble.shape[1] == 1:
                    # Binary with single logit: sigmoid -> [p0, p1]
                    prob_pos = torch.sigmoid(logits_ensemble).cpu().numpy()[0:1]
                    probas = np.concatenate([1.0 - prob_pos, prob_pos], axis=1)[0]
                    pred_class = int(probas[1] > 0.5)
                else:
                    probas = F.softmax(logits_ensemble, dim=1).cpu().numpy()[0]
                    pred_class = int(np.argmax(probas))

                predictions.append({
                    "case_id": case_id,
                    "pred_class": pred_class,
                    **{f"prob_{i}": float(probas[i]) for i in range(probas.shape[0])}
                })

                subset_names = None
                if trained_feature_names:
                    subset_names = [
                        [trained_feature_names[idx] for idx in subset.tolist()]
                        for subset in topm_subsets
                    ]

                if args.retrieval == "budgeted":
                    selected_subsets.append({
                        "case_id": case_id,
                        "final_top_indices": topm_subsets.tolist(),
                        "final_top_scores": topm_scores.tolist(),
                        "final_top_names": subset_names
                    })
                else:
                    selected_subsets.append({
                        "case_id": case_id,
                        "topm_indices": topm_subsets.tolist(),
                        "topm_scores": topm_scores.tolist(),
                        "topm_names": subset_names
                    })

                if labels_dict:
                    all_labels.append(int(batch["label"][i].item()))
                    all_preds.append(pred_class)
                    all_probas.append(probas)

    pred_df = pd.DataFrame(predictions)
    pred_path = outdir / "predictions_eval.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions to {pred_path}")

    selected_path = outdir / "selected_subsets_eval.json"
    with open(selected_path, "w") as f:
        json.dump(selected_subsets, f, indent=2)
    logger.info(f"Saved selected subsets to {selected_path}")

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
            fold_result = run_metrics_on_case_ids(fold_case_ids)
            fold_metrics.append(fold_result)
            logger.info(
                f"CV Fold {fold_idx}/{args.cv_folds} | "
                f"Acc={fold_result['accuracy']:.4f} Macro-F1={fold_result['macro_f1']:.4f} QWK={fold_result['qwk']:.4f}"
            )

        keys = sorted({k for m in fold_metrics for k in m.keys() if k != "per_class"})
        summary = {"folds": fold_metrics, "mean": {}, "std": {}, "per_class_mean": {}, "per_class_std": {}}
        for key in keys:
            values = [m[key] for m in fold_metrics if key in m]
            summary["mean"][key] = float(np.mean(values))
            summary["std"][key] = float(np.std(values))
            logger.info(f"CV {key}: {summary['mean'][key]:.4f} ± {summary['std'][key]:.4f}")

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

