r"""
PyRadiomics Extraction from Predicted Masks (3D)

This script:
1. Loads pre-saved predicted segmentation masks (NIfTI)
2. Extracts ROIs (labels > 0) from the predicted segmentation mask
3. Extracts configurable radiomics features for each ROI using PyRadiomics
4. Saves results in CSV/Parquet format (long + wide)

Key fixes vs your current version:
- Robustly converts PyRadiomics outputs to scalar float (avoids “101 non-numeric” being dropped)
- Explicitly enforces 3D extraction (force2D=False, voxelBased=False)
- Preserves image metadata correctly when voxelArrayShift is applied
- Optionally resamples ROI mask to image geometry if mismatch (nearest-neighbor)

Example:
python torchradiomics_from_ROIs.py `
  --images-dir "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTr" `
  --predicted-masks-dir "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_results\predicted_masks" `
  --output-dir "output" `
  --split train `
  --output-format csv `
  --max-cases 5

python torchradiomics_from_ROIs.py \
  --images-dir "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib/imagesTr" \
  --predicted-masks-dir "/home/yaxi/nnUNet/nnUNet_results/predicted_masks/train" \
  --output-dir "output" \
  --split train \
  --output-format csv \
  --max-cases 5

"""

import os
import sys
import json
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk

from radiomics import featureextractor

# Reduce PyRadiomics internal verbosity
logging.getLogger("radiomics").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _to_scalar_float(x) -> Optional[float]:
    """
    Convert PyRadiomics output value to a scalar float if possible.
    Handles python scalars, numpy scalars, 0-d / single-element numpy arrays.

    Returns None if cannot convert to a scalar float.
    """
    try:
        if isinstance(x, np.ndarray):
            # allow 0-D or single-element arrays
            if x.size != 1:
                return None
            x = x.reshape(-1)[0]
        val = float(x)
        return val
    except Exception:
        return None


def _ensure_same_geometry(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """
    Ensure mask has the same geometry as reference image.
    If size/spacing/origin/direction mismatch, resample mask to reference using nearest neighbor.
    """
    same_size = (mask.GetSize() == reference.GetSize())
    same_spacing = (mask.GetSpacing() == reference.GetSpacing())
    same_origin = (mask.GetOrigin() == reference.GetOrigin())
    same_direction = (mask.GetDirection() == reference.GetDirection())

    if same_size and same_spacing and same_origin and same_direction:
        return mask

    logger.warning(
        "Mask geometry != image geometry. Resampling mask to image space (nearest-neighbor)."
    )

    resampled = sitk.Resample(
        mask,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,                # default value
        sitk.sitkUInt8
    )
    return resampled


# =============================================================================
# Radiomics Feature Extraction (PyRadiomics)
# =============================================================================

def extract_radiomics_with_groups(
    image: Union[sitk.Image, np.ndarray, torch.Tensor],
    mask: Union[sitk.Image, np.ndarray, torch.Tensor],
    voxelArrayShift: int = 0,
    binWidth: Optional[float] = None,
    groups: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Extract radiomics features with configurable groups using PyRadiomics.

    Returns:
        (filtered_features, feature_names)
    """
    if groups is None:
        groups = ["firstorder", "shape", "glcm", "gldm", "glrlm", "glszm", "ngtdm"]

    # ---- Convert image to SimpleITK if needed
    if isinstance(image, sitk.Image):
        sitk_img = image
    elif isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
        sitk_img = sitk.GetImageFromArray(img_np)
    elif isinstance(image, np.ndarray):
        sitk_img = sitk.GetImageFromArray(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # ---- Convert mask to SimpleITK if needed
    if isinstance(mask, sitk.Image):
        sitk_mask = mask
    elif isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
        sitk_mask = sitk.GetImageFromArray(mask_np)
    elif isinstance(mask, np.ndarray):
        sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)}")

    # ---- Empty mask check
    mask_array = sitk.GetArrayFromImage(sitk_mask)
    if mask_array.sum() == 0:
        logger.warning("Empty ROI mask - returning empty features")
        return {}, []

    # ---- Apply voxel array shift (preserve metadata!)
    if voxelArrayShift != 0:
        old_spacing = sitk_img.GetSpacing()
        old_origin = sitk_img.GetOrigin()
        old_direction = sitk_img.GetDirection()

        img_array = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        img_array = img_array + float(voxelArrayShift)

        sitk_img = sitk.GetImageFromArray(img_array)
        sitk_img.SetSpacing(old_spacing)
        sitk_img.SetOrigin(old_origin)
        sitk_img.SetDirection(old_direction)

    # ---- Ensure mask matches image geometry (important for PyRadiomics)
    sitk_mask = _ensure_same_geometry(sitk_mask, sitk_img)

    # ---- Configure PyRadiomics extractor
    extractor_params = {}
    if binWidth is not None:
        extractor_params["binWidth"] = float(binWidth)

    extractor = featureextractor.RadiomicsFeatureExtractor(**extractor_params)

    # Explicitly enforce 3D behavior (avoid any accidental 2D forcing)
    extractor.settings["force2D"] = False
    extractor.settings["voxelBased"] = False  # ensure we get scalar features, not feature maps

    # Disable everything, then enable requested feature classes
    extractor.disableAllFeatures()

    # Ensure Original image type is enabled (safe default)
    try:
        extractor.enableImageTypeByName("Original")
    except Exception:
        # Some versions don’t require this; ignore if not available
        pass

    enabled_classes = []
    for cls in ["firstorder", "shape", "glcm", "gldm", "glrlm", "glszm", "ngtdm"]:
        if cls in groups:
            extractor.enableFeatureClassByName(cls)
            enabled_classes.append(cls)

    logger.info(f"Enabled feature classes: {enabled_classes}")
    logger.info(
        f"Extractor settings: force2D={extractor.settings.get('force2D', None)}, "
        f"voxelBased={extractor.settings.get('voxelBased', None)}"
    )

    # ---- Execute extraction (label=1 for binary ROI mask)
    try:
        features_dict = extractor.execute(sitk_img, sitk_mask, label=1)
        logger.debug(f"PyRadiomics returned {len(features_dict)} total entries (incl diagnostics)")
    except Exception as e:
        logger.warning(f"Failed to extract radiomics features: {e}", exc_info=True)
        return {}, []

    # ---- Filter + robust float conversion
    filtered_features: Dict[str, float] = {}
    feature_names: List[str] = []
    skipped_diagnostics = 0
    skipped_non_numeric = 0
    skipped_invalid = 0

    for feat_name, feat_value in features_dict.items():
        if feat_name.startswith("diagnostics"):
            skipped_diagnostics += 1
            continue

        val = _to_scalar_float(feat_value)
        if val is None:
            skipped_non_numeric += 1
            continue

        if math.isnan(val) or math.isinf(val):
            skipped_invalid += 1
            continue

        filtered_features[feat_name] = val
        feature_names.append(feat_name)

    logger.info(
        f"Extracted {len(filtered_features)} features "
        f"(skipped {skipped_diagnostics} diagnostics, {skipped_non_numeric} non-numeric, {skipped_invalid} invalid)"
    )

    # Log feature class distribution for debugging
    feature_classes: Dict[str, int] = {}
    for feat_name in feature_names:
        parts = feat_name.split("_")
        if len(parts) >= 2:
            class_name = parts[1]  # e.g., "firstorder", "shape", ...
            feature_classes[class_name] = feature_classes.get(class_name, 0) + 1
    if feature_classes:
        logger.info(f"Feature distribution by class: {feature_classes}")

    return filtered_features, feature_names


# =============================================================================
# ROI Extraction
# =============================================================================

def extract_rois(
    segmentation_mask: np.ndarray,
    roi_labels: Optional[Dict[int, str]] = None,
    min_voxels: int = 1
) -> Dict[str, np.ndarray]:
    """
    Extract binary masks for each ROI from segmentation (3D).
    """
    unique_labels = np.unique(segmentation_mask)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background

    rois: Dict[str, np.ndarray] = {}

    for label_val in unique_labels:
        label_int = int(label_val)
        roi_name = roi_labels[label_int] if (roi_labels and label_int in roi_labels) else f"roi_{label_int}"

        binary_mask = (segmentation_mask == label_val).astype(np.uint8)

        vox = int(binary_mask.sum())
        if vox >= min_voxels:
            rois[roi_name] = binary_mask
        else:
            logger.warning(f"ROI {roi_name} (label {label_int}) has only {vox} voxels, skipping")

    return rois


# =============================================================================
# Main Workflow
# =============================================================================

def process_case_from_predicted_mask(
    case_image_path: Union[str, Path],
    predicted_mask_path: Union[str, Path],
    roi_labels: Optional[Dict[int, str]] = None,
    radiomics_groups: Optional[List[str]] = None,
    voxelArrayShift: int = 0,
    binWidth: Optional[float] = None
) -> pd.DataFrame:
    """
    Extract radiomics from one case using its pre-saved predicted mask.
    Returns a long DataFrame: case_id, roi_name, feature_name, value
    """
    case_image_path = Path(case_image_path)
    predicted_mask_path = Path(predicted_mask_path)

    case_id = case_image_path.stem.replace(".nii", "").replace(".gz", "").replace("_0000", "")
    logger.info(f"Processing case: {case_id}")

    # Load predicted mask (SITK, typically 3D)
    sitk_mask = sitk.ReadImage(str(predicted_mask_path))
    segmentation = sitk.GetArrayFromImage(sitk_mask)  # numpy [z,y,x]

    logger.info(f"Segmentation array shape (z,y,x): {segmentation.shape}")
    logger.info(f"Mask SITK size: {sitk_mask.GetSize()}, spacing: {sitk_mask.GetSpacing()}")

    # Extract ROI masks (3D)
    rois = extract_rois(segmentation, roi_labels=roi_labels)

    if len(rois) == 0:
        logger.warning(f"No valid ROIs found for case {case_id}")
        return pd.DataFrame(columns=["case_id", "roi_name", "feature_name", "value"])

    # Load original image (SITK, typically 3D)
    sitk_image = sitk.ReadImage(str(case_image_path))
    logger.info(f"Image SITK size: {sitk_image.GetSize()}, spacing: {sitk_image.GetSpacing()}")

    results = []

    for roi_name, roi_mask in rois.items():
        logger.info(f"  Extracting radiomics for {roi_name}...")

        # Create SITK mask for this ROI, copy predicted mask metadata
        roi_mask_sitk = sitk.GetImageFromArray(roi_mask.astype(np.uint8))
        roi_mask_sitk.SetSpacing(sitk_mask.GetSpacing())
        roi_mask_sitk.SetOrigin(sitk_mask.GetOrigin())
        roi_mask_sitk.SetDirection(sitk_mask.GetDirection())

        # Radiomics extraction
        features_dict, _ = extract_radiomics_with_groups(
            sitk_image,
            roi_mask_sitk,
            voxelArrayShift=voxelArrayShift,
            binWidth=binWidth,
            groups=radiomics_groups
        )

        for feat_name, feat_value in features_dict.items():
            results.append({
                "case_id": case_id,
                "roi_name": roi_name,
                "feature_name": feat_name,
                "value": feat_value
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Extract radiomics features from pre-saved predicted masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory containing input images (NIfTI format)")
    parser.add_argument("--predicted-masks-dir", type=str, required=True,
                        help="Directory containing predicted masks (NIfTI). Can be flat or have train/test subdirs.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--dataset-json", type=str, default=None,
                        help="Optional dataset.json for ROI label mapping")

    parser.add_argument("--radiomics-groups", type=str, nargs="+",
                        default=["firstorder", "shape", "glcm", "gldm", "glrlm", "glszm", "ngtdm"],
                        choices=["firstorder", "shape", "glcm", "gldm", "glrlm", "glszm", "ngtdm"],
                        help="Radiomics feature groups to extract")

    parser.add_argument("--voxel-array-shift", type=int, default=0,
                        help="Shift applied to image intensities (avoid negatives if needed)")
    parser.add_argument("--bin-width", type=float, default=None,
                        help="Histogram bin width (None uses PyRadiomics default)")

    parser.add_argument("--output-format", type=str, default="csv",
                        choices=["csv", "parquet"], help="Output format")
    parser.add_argument("--split", type=str, choices=["train", "test", "both"], default="both",
                        help="Which split to process if masks are stored in train/test subdirs")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Max number of cases for testing")

    args = parser.parse_args()

    logger.info(f"Radiomics groups: {args.radiomics_groups}")

    # Load ROI label mapping (optional)
    roi_labels = None
    if args.dataset_json:
        dataset_json_path = Path(args.dataset_json)
        if dataset_json_path.exists():
            with open(dataset_json_path, "r") as f:
                dataset_json = json.load(f)
            if "labels" in dataset_json:
                # Reverse mapping: label value -> name
                roi_labels = {int(v): k.replace(" ", "_") for k, v in dataset_json["labels"].items() if int(v) > 0}
                logger.info(f"ROI labels loaded: {roi_labels}")
        else:
            logger.warning(f"Dataset JSON not found: {dataset_json_path}")

    # Find all images
    images_dir = Path(args.images_dir)
    image_files = sorted(images_dir.glob("*.nii.gz")) + sorted(images_dir.glob("*.nii"))

    if len(image_files) == 0:
        logger.error(f"No images found in {images_dir}")
        return

    total_images = len(image_files)
    if args.max_cases is not None and args.max_cases > 0:
        image_files = image_files[:args.max_cases]
        logger.info(f"Found {total_images} images total, processing first {len(image_files)} (--max-cases={args.max_cases})")
    else:
        logger.info(f"Found {len(image_files)} images to process")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted_masks_dir = Path(args.predicted_masks_dir)
    logger.info(f"Extracting radiomics from predicted masks in {predicted_masks_dir}")

    # Check subdir layout
    train_dir = predicted_masks_dir / "train"
    test_dir = predicted_masks_dir / "test"
    has_subdirs = train_dir.exists() or test_dir.exists()

    if has_subdirs:
        if args.split == "both":
            splits_to_search = ["train", "test"]
        else:
            splits_to_search = [args.split]
    else:
        splits_to_search = [None]

    all_results = []

    for idx, image_path in enumerate(image_files, 1):
        logger.info(f"\n[{idx}/{len(image_files)}] Processing {image_path.name}")
        case_id = image_path.stem.replace(".nii", "").replace(".gz", "").replace("_0000", "")

        # Find corresponding predicted mask
        mask_path = None

        if has_subdirs:
            for split_name in splits_to_search:
                if split_name == "train":
                    candidate = train_dir / f"{case_id}_majority_vote.nii.gz"
                else:
                    candidate = test_dir / f"{case_id}_majority_vote.nii.gz"

                if candidate.exists():
                    mask_path = candidate
                    break
        else:
            # Flat directory: try common naming schemes
            candidate = predicted_masks_dir / image_path.name
            if candidate.exists():
                mask_path = candidate
            else:
                candidate2 = predicted_masks_dir / image_path.name.replace("_0000.nii.gz", ".nii.gz").replace("_0000.nii", ".nii")
                if candidate2.exists():
                    mask_path = candidate2
                else:
                    candidate3 = predicted_masks_dir / (image_path.stem.replace(".nii", "").replace(".gz", "") + "_pred_mask.nii.gz")
                    if candidate3.exists():
                        mask_path = candidate3
                    else:
                        candidate4 = predicted_masks_dir / f"{case_id}_majority_vote.nii.gz"
                        if candidate4.exists():
                            mask_path = candidate4

        if not mask_path:
            logger.warning(f"Predicted mask not found for {image_path.name} (case_id: {case_id}), skipping")
            continue

        case_df = process_case_from_predicted_mask(
            image_path,
            mask_path,
            roi_labels=roi_labels,
            radiomics_groups=args.radiomics_groups,
            voxelArrayShift=args.voxel_array_shift,
            binWidth=args.bin_width
        )

        if len(case_df) > 0:
            all_results.append(case_df)
        else:
            logger.warning(f"No results for {image_path.name}")

    if len(all_results) == 0:
        logger.error("No results to save!")
        return

    df_all = pd.concat(all_results, ignore_index=True)

    # Save long format
    if args.output_format == "csv":
        out_file = output_dir / "radiomics_results.csv"
        df_all.to_csv(out_file, index=False)
    else:
        out_file = output_dir / "radiomics_results.parquet"
        df_all.to_parquet(out_file, index=False)

    logger.info(f"\nSaved results to {out_file}")

    # Save wide format
    df_wide = df_all.pivot_table(
        index=["case_id", "roi_name"],
        columns="feature_name",
        values="value"
    ).reset_index()

    wide_file = output_dir / f"radiomics_results_wide.{args.output_format}"
    if args.output_format == "csv":
        df_wide.to_csv(wide_file, index=False)
    else:
        df_wide.to_parquet(wide_file, index=False)

    logger.info(f"Saved wide format to {wide_file}")
    logger.info(f"\nTotal cases processed: {df_all['case_id'].nunique()}")
    logger.info(f"Total ROIs: {df_all.groupby(['case_id', 'roi_name']).ngroups}")
    logger.info(f"Total unique features: {df_all['feature_name'].nunique()}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
