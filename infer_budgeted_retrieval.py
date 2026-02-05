"""
Budgeted inference: recall + rank + local search + ensemble classification.
"""

import json
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import joblib

from models import JointScoringModel
from data_loader import load_radiomics_wide_format, KLGradeDataset
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def sample_random_subsets(num_features: int, k: int, n_subsets: int, rng: random.Random) -> np.ndarray:
    subsets = []
    for _ in range(n_subsets):
        subsets.append(rng.sample(range(num_features), k))
    return np.array(subsets, dtype=np.int64)


def mutate_subset(subset: np.ndarray, num_features: int, rng: random.Random) -> np.ndarray:
    """
    1-swap or 2-swap mutation on subset indices.
    """
    subset = subset.copy()
    k = len(subset)
    swap_count = 1 if rng.random() < 0.7 else 2
    for _ in range(swap_count):
        pos = rng.randrange(k)
        new_idx = rng.randrange(num_features)
        subset[pos] = new_idx
    return subset


def build_subset_tensors(
    radiomics: np.ndarray,
    subset_indices: np.ndarray,
    feature_to_roi: np.ndarray,
    feature_to_type: np.ndarray,
    device: torch.device
):
    f_ids = torch.from_numpy(subset_indices).long().to(device)
    r_ids = torch.from_numpy(feature_to_roi[subset_indices]).long().to(device)
    t_ids = torch.from_numpy(feature_to_type[subset_indices]).long().to(device)
    vals = torch.from_numpy(radiomics[subset_indices]).float().to(device)
    return f_ids, r_ids, t_ids, vals


def main():
    parser = argparse.ArgumentParser(
        description="Inference with budgeted subset retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--images-ts", type=str, required=True, help="Test images directory")
    parser.add_argument("--radiomics-test-csv", type=str, required=True, help="Test radiomics CSV (wide format)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pth")
    parser.add_argument("--scaler", type=str, required=False, default=None, help="Path to scaler.joblib")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--recall", type=int, default=4000)
    parser.add_argument("--keep", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--mut-per-seed", type=int, default=200)
    parser.add_argument("--final-top", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    radiomics_test, feature_names, _, feature_meta = load_radiomics_wide_format(
        Path(args.radiomics_test_csv)
    )

    if args.scaler:
        scaler = joblib.load(Path(args.scaler))
        for cid in radiomics_test.keys():
            radiomics_test[cid] = scaler.transform(radiomics_test[cid].reshape(1, -1))[0]

    test_dataset = KLGradeDataset(
        list(radiomics_test.keys()),
        Path(args.images_ts),
        radiomics_test,
        labels_dict=None,
        target_shape=(32, 128, 128)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )

    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    num_features = checkpoint.get("num_features", len(feature_names))
    feature_to_roi = np.array(feature_meta["feature_to_roi"], dtype=np.int64)
    feature_to_type = np.array(feature_meta["feature_to_type"], dtype=np.int64)
    num_rois = len(feature_meta["roi_names"])
    num_types = len(feature_meta["type_names"])
    model = JointScoringModel(
        num_features=num_features,
        num_rois=num_rois,
        num_types=num_types
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rng = random.Random(12345)
    predictions = []
    selected_sets = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            case_ids = batch["case_id"]

            for i in range(len(case_ids)):
                case_id = case_ids[i]
                image = images[i:i + 1]
                radiomics = radiomics_test[case_id]

                # Recall
                recall_subsets = sample_random_subsets(num_features, args.k, args.recall, rng)
                recall_scores = []
                for subset in recall_subsets:
                    f_ids, r_ids, t_ids, vals = build_subset_tensors(
                        radiomics, subset, feature_to_roi, feature_to_type, device
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
                        radiomics, subset, feature_to_roi, feature_to_type, device
                    )
                    score = model.score_subset(
                        image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                    )
                    final_scores.append(score.item())
                final_scores = np.array(final_scores)
                final_indices = np.argsort(final_scores)[-args.final_top:]
                final_subsets = keep_subsets[final_indices]

                # Classifier ensemble
                logits_list = []
                for subset in final_subsets:
                    f_ids, r_ids, t_ids, vals = build_subset_tensors(
                        radiomics, subset, feature_to_roi, feature_to_type, device
                    )
                    logits = model.classify_subset(
                        image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                    )
                    logits_list.append(logits)
                logits_ensemble = torch.mean(torch.cat(logits_list, dim=0), dim=0, keepdim=True)
                probas = F.softmax(logits_ensemble, dim=1).cpu().numpy()[0]
                pred_class = int(np.argmax(probas))

                predictions.append({
                    "case_id": case_id,
                    "pred_class": pred_class,
                    "prob_0": float(probas[0]),
                    "prob_1": float(probas[1]),
                    "prob_2": float(probas[2]),
                    "prob_3": float(probas[3]),
                    "prob_4": float(probas[4])
                })

                selected_sets.append({
                    "case_id": case_id,
                    "final_top_indices": final_subsets.tolist()
                })

    pred_path = outdir / "predictions_test.csv"
    with open(pred_path, "w") as f:
        header = "case_id,pred_class,prob_0,prob_1,prob_2,prob_3,prob_4\n"
        f.write(header)
        for row in predictions:
            f.write(
                f"{row['case_id']},{row['pred_class']},{row['prob_0']},"
                f"{row['prob_1']},{row['prob_2']},{row['prob_3']},{row['prob_4']}\n"
            )
    with open(outdir / "selected_subsets_test.json", "w") as f:
        json.dump(selected_sets, f, indent=2)

    logger.info(f"Saved predictions to {pred_path}")


if __name__ == "__main__":
    main()

