"""
Data Loading and Preprocessing for KLGrade Prediction

This module contains:
- load_radiomics_long_format: Load radiomics from long format CSV
- load_radiomics_wide_format: Load radiomics from wide format CSV
- load_klgrade_labels: Load KLGrade labels from CSV
- preprocess_image: Load and preprocess NIfTI images
- KLGradeDataset: PyTorch dataset for images + radiomics + labels
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy import ndimage

logger = logging.getLogger(__name__)


def load_radiomics_long_format(
    csv_path: Path,
    expected_rois: Optional[List[str]] = None,
    expected_features: Optional[List[str]] = None
) -> Tuple[Dict[str, np.ndarray], List[str], List[str], Dict[str, int]]:
    """
    Load radiomics from long format CSV and build fixed-length vectors.
    
    Args:
        csv_path: Path to CSV with columns: case_id, roi_name, feature_name, value
        expected_rois: Optional list of expected ROI names (for validation)
        expected_features: Optional list of expected feature names (for validation)
    
    Returns:
        Tuple of:
        - case_radiomics: Dict[case_id, np.ndarray] of shape (n_features_total,)
        - roi_names: Sorted list of ROI names
        - feature_names: Sorted list of feature names
        - missing_stats: Dict with missing ROI/feature counts
    """
    logger.info(f"Loading radiomics from {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Radiomics CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_cols = ["case_id", "roi_name", "feature_name", "value"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")
    
    # Get unique ROIs and features
    all_rois = sorted(df["roi_name"].unique())
    all_features = sorted(df["feature_name"].unique())
    
    logger.info(f"Found {len(all_rois)} unique ROIs: {all_rois}")
    logger.info(f"Found {len(all_features)} unique features per ROI")
    
    # Validate expected ROIs/features if provided
    if expected_rois is not None:
        missing_rois = set(expected_rois) - set(all_rois)
        if missing_rois:
            logger.warning(f"Missing expected ROIs: {missing_rois}")
    
    if expected_features is not None:
        missing_features = set(expected_features) - set(all_features)
        if missing_features:
            logger.warning(f"Missing expected features: {missing_features}")
    
    # Build fixed mapping: index -> (roi_name, feature_name)
    feature_mapping = []
    for roi in all_rois:
        for feat in all_features:
            feature_mapping.append(f"{roi}:{feat}")
    
    n_features_total = len(feature_mapping)
    logger.info(f"Total feature vector length: {n_features_total} ({len(all_rois)} ROIs × {len(all_features)} features)")
    
    # Build vectors for each case
    case_radiomics = {}
    missing_stats = {
        "cases_with_missing_rois": 0,
        "cases_with_missing_features": 0,
        "total_missing_values": 0
    }
    
    for case_id in df["case_id"].unique():
        case_df = df[df["case_id"] == case_id]
        
        # Build vector
        vector = np.zeros(n_features_total, dtype=np.float32)
        missing_count = 0
        
        for idx, (roi, feat) in enumerate([(r, f) for r in all_rois for f in all_features]):
            matching = case_df[(case_df["roi_name"] == roi) & (case_df["feature_name"] == feat)]
            if len(matching) > 0:
                value = matching["value"].iloc[0]
                if pd.notna(value):
                    vector[idx] = float(value)
                else:
                    missing_count += 1
            else:
                missing_count += 1
        
        if missing_count > 0:
            missing_stats["total_missing_values"] += missing_count
            # Check if entire ROI is missing
            case_rois = set(case_df["roi_name"].unique())
            missing_rois = set(all_rois) - case_rois
            if missing_rois:
                missing_stats["cases_with_missing_rois"] += 1
            # Check if features are missing
            for roi in case_rois:
                roi_features = set(case_df[case_df["roi_name"] == roi]["feature_name"].unique())
                expected_roi_features = set(all_features)
                if roi_features != expected_roi_features:
                    missing_stats["cases_with_missing_features"] += 1
                    break
        
        case_radiomics[case_id] = vector
    
    logger.info(f"Loaded {len(case_radiomics)} cases")
    logger.info(f"Missing stats: {missing_stats}")
    
    return case_radiomics, all_rois, all_features, missing_stats


def load_klgrade_labels(file_path: Path) -> Dict[str, int]:
    """
    Load KLGrade labels from CSV or Excel file.
    
    The CMT-ID column in the Excel file corresponds to the image naming pattern:
    oaizib_[CMT-ID]_0000.nii, so case_id is formatted as oaizib_[CMT-ID] where
    CMT-ID is zero-padded to 3 digits.
    
    Args:
        file_path: Path to CSV or Excel file with columns: CMT-ID (or similar), KLGrade
    
    Returns:
        Dict[case_id, KLGrade] where case_id is in format oaizib_XXX and KLGrade is int 0-4
    """
    logger.info(f"Loading KLGrade labels from {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"KLGrade file not found: {file_path}")
    
    # Load file based on extension
    if file_path.suffix.lower() == '.xlsx' or file_path.suffix.lower() == '.xls':
        df = pd.read_excel(file_path)
    elif file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .csv, .xlsx, or .xls")
    
    # Find CMT-ID column (try common variations)
    cmt_id_col = None
    for col in df.columns:
        col_lower = col.lower().strip().replace(' ', '_').replace('-', '_')
        if col_lower in ['cmt_id', 'cmtid', 'cmt-id', 'cmt']:
            cmt_id_col = col
            break
    
    # If CMT-ID not found, try other common ID column names
    if cmt_id_col is None:
        for col in df.columns:
            col_lower = col.lower().strip().replace(' ', '_')
            if col_lower in ['case_id', 'caseid', 'id', 'subject_id', 'subjectid', 'subid', 'sub_id']:
                cmt_id_col = col
                break
    
    if cmt_id_col is None:
        # Try first column if no match found
        logger.warning(f"Could not find CMT-ID or case_id column in {df.columns.tolist()}, using first column: {df.columns[0]}")
        cmt_id_col = df.columns[0]
    
    if "KLGrade" not in df.columns:
        raise ValueError(f"File must have 'KLGrade' column. Found columns: {df.columns.tolist()}")
    
    labels = {}
    for _, row in df.iterrows():
        # Handle NaN values in CMT-ID
        if pd.isna(row[cmt_id_col]):
            logger.warning(f"Missing CMT-ID in row, skipping")
            continue
        
        # Get CMT-ID value
        cmt_id_raw = str(row[cmt_id_col]).strip()
        
        # Convert CMT-ID to case_id format: oaizib_XXX (zero-padded to 3 digits)
        try:
            # Try to convert to int first (handles numeric CMT-IDs)
            cmt_id_int = int(float(cmt_id_raw))  # Use float first to handle "001.0" type strings
            cmt_id_str = f"{cmt_id_int:03d}"  # Zero-pad to 3 digits
        except (ValueError, TypeError):
            # If not numeric, try to extract digits from string
            digits = re.findall(r'\d+', cmt_id_raw)
            if digits:
                cmt_id_int = int(digits[0])
                cmt_id_str = f"{cmt_id_int:03d}"
            else:
                # If no digits found, use as-is (might already be in correct format)
                cmt_id_str = cmt_id_raw
        
        case_id = f"oaizib_{cmt_id_str}"
        
        # Handle NaN values in KLGrade
        if pd.isna(row["KLGrade"]):
            logger.warning(f"Missing KLGrade for case {case_id} (CMT-ID: {cmt_id_raw}), skipping")
            continue
        
        try:
            klgrade = int(row["KLGrade"])
        except (ValueError, TypeError):
            logger.warning(f"Invalid KLGrade value '{row['KLGrade']}' for case {case_id} (CMT-ID: {cmt_id_raw}), skipping")
            continue
        
        if klgrade < 0 or klgrade > 4:
            logger.warning(f"Invalid KLGrade {klgrade} for case {case_id} (CMT-ID: {cmt_id_raw}), skipping")
            continue
        labels[case_id] = klgrade
    
    logger.info(f"Loaded {len(labels)} KLGrade labels")
    logger.info(f"Label distribution: {pd.Series(list(labels.values())).value_counts().sort_index().to_dict()}")
    
    return labels


def load_radiomics_wide_format(
    csv_path: Path,
    case_id_col: str = "case_id"
) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, int], Dict[str, object]]:
    """
    Load radiomics from wide format CSV.

    Args:
        csv_path: Path to CSV with one row per case.
        case_id_col: Column name containing case IDs.

    Returns:
        Tuple of:
        - case_radiomics: Dict[case_id, np.ndarray] of shape (n_features_total,)
        - feature_names: List of feature column names (ordered)
        - missing_stats: Dict with missing value counts
    """
    logger.info(f"Loading radiomics (wide format) from {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Radiomics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if case_id_col not in df.columns:
        # Fall back to first column if case_id not present
        logger.warning(
            f"Missing '{case_id_col}' column in CSV. Using first column as case_id."
        )
        case_id_col = df.columns[0]

    roi_col = "roi_name" if "roi_name" in df.columns else None
    candidate_cols = [c for c in df.columns if c not in {case_id_col, roi_col}]
    if not candidate_cols:
        raise ValueError("No candidate feature columns found in wide-format CSV.")

    # Coerce feature columns to numeric; drop non-numeric columns
    numeric_df = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    non_numeric_cols = [c for c in candidate_cols if numeric_df[c].isna().all()]
    if non_numeric_cols:
        logger.warning(f"Dropping non-numeric feature columns: {non_numeric_cols}")
        numeric_df = numeric_df.drop(columns=non_numeric_cols)

    base_feature_names = list(numeric_df.columns)
    if not base_feature_names:
        raise ValueError("No numeric feature columns found in wide-format CSV.")

    logger.info(f"Found {len(base_feature_names)} base feature columns.")

    case_radiomics = {}
    missing_stats = {
        "cases_with_missing_values": 0,
        "total_missing_values": 0
    }
    feature_meta: Dict[str, object] = {}

    if roi_col:
        roi_names = sorted(df[roi_col].dropna().unique().tolist())
        feature_names = [f"{roi}:{feat}" for roi in roi_names for feat in base_feature_names]
        roi_to_id = {roi: i for i, roi in enumerate(roi_names)}

        def parse_feature_type(name: str) -> str:
            parts = name.split("_")
            if len(parts) >= 2:
                return "_".join(parts[:2])
            return parts[0]

        type_names = sorted({parse_feature_type(f) for f in base_feature_names})
        type_to_id = {t: i for i, t in enumerate(type_names)}
        feature_to_roi = np.array([roi_to_id[roi] for roi in roi_names for _ in base_feature_names], dtype=np.int64)
        feature_to_type = np.array(
            [type_to_id[parse_feature_type(f)] for _ in roi_names for f in base_feature_names],
            dtype=np.int64
        )

        for case_id in df[case_id_col].astype(str).unique():
            case_id = case_id.strip()
            if not case_id:
                continue
            vec = np.zeros(len(feature_names), dtype=np.float32)
            case_rows = df[df[case_id_col].astype(str) == case_id]
            for _, row in case_rows.iterrows():
                roi = row[roi_col]
                if roi not in roi_to_id:
                    continue
                values = numeric_df.loc[row.name, base_feature_names].to_numpy(dtype=np.float32, copy=True)
                missing_count = int(np.isnan(values).sum())
                if missing_count > 0:
                    missing_stats["cases_with_missing_values"] += 1
                    missing_stats["total_missing_values"] += missing_count
                    values = np.nan_to_num(values, nan=0.0).astype(np.float32)
                start = roi_to_id[roi] * len(base_feature_names)
                vec[start:start + len(base_feature_names)] = values
            case_radiomics[case_id] = vec
    else:
        feature_names = base_feature_names
        roi_names = ["unknown"]
        roi_to_id = {"unknown": 0}

        def parse_feature_type(name: str) -> str:
            parts = name.split("_")
            if len(parts) >= 2:
                return "_".join(parts[:2])
            return parts[0]

        type_names = sorted({parse_feature_type(f) for f in feature_names})
        type_to_id = {t: i for i, t in enumerate(type_names)}
        feature_to_roi = np.zeros(len(feature_names), dtype=np.int64)
        feature_to_type = np.array([type_to_id[parse_feature_type(f)] for f in feature_names], dtype=np.int64)

        for _, row in df.iterrows():
            case_id = str(row[case_id_col]).strip()
            if not case_id:
                continue
            values = numeric_df.loc[row.name, feature_names].to_numpy(dtype=np.float32, copy=True)
            missing_count = int(np.isnan(values).sum())
            if missing_count > 0:
                missing_stats["cases_with_missing_values"] += 1
                missing_stats["total_missing_values"] += missing_count
                values = np.nan_to_num(values, nan=0.0).astype(np.float32)
            case_radiomics[case_id] = values

    feature_meta = {
        "roi_names": roi_names,
        "type_names": type_names,
        "feature_to_roi": feature_to_roi.tolist(),
        "feature_to_type": feature_to_type.tolist()
    }

    logger.info(f"Loaded {len(case_radiomics)} cases")
    logger.info(f"Missing stats: {missing_stats}")

    return case_radiomics, feature_names, missing_stats, feature_meta


def preprocess_image(
    image_path: Path,
    target_shape: Tuple[int, int, int] = (32, 128, 128)
) -> torch.Tensor:
    """
    Load and preprocess NIfTI image.
    
    Args:
        image_path: Path to NIfTI file
        target_shape: Target (D, H, W) shape
    
    Returns:
        Tensor of shape [1, D, H, W] (single channel)
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load with SimpleITK
    sitk_img = sitk.ReadImage(str(image_path))
    img_array = sitk.GetArrayFromImage(sitk_img)  # [z, y, x]
    
    # Z-score normalize per volume
    img_array = img_array.astype(np.float32)
    mean = img_array.mean()
    std = img_array.std()
    if std > 0:
        img_array = (img_array - mean) / std
    else:
        logger.warning(f"Zero std for {image_path.name}, skipping normalization")
    
    # Resize to target shape
    current_shape = img_array.shape
    if current_shape != target_shape:
        # Use scipy.ndimage.zoom for resizing
        zoom_factors = [
            target_shape[0] / current_shape[0],
            target_shape[1] / current_shape[1],
            target_shape[2] / current_shape[2]
        ]
        img_array = ndimage.zoom(img_array, zoom_factors, order=1, mode='nearest')
    
    # Convert to tensor and add channel dimension
    tensor = torch.from_numpy(img_array).unsqueeze(0)  # [1, D, H, W]
    
    return tensor


class KLGradeDataset(Dataset):
    """Dataset for KLGrade prediction with images and radiomics."""
    
    def __init__(
        self,
        case_ids: List[str],
        images_dir: Path,
        radiomics_dict: Dict[str, np.ndarray],
        labels_dict: Optional[Dict[str, int]] = None,
        target_shape: Tuple[int, int, int] = (32, 128, 128),
        transform: Optional[callable] = None
    ):
        """
        Args:
            case_ids: List of case IDs
            images_dir: Directory containing NIfTI images
            radiomics_dict: Dict[case_id, radiomics_vector]
            labels_dict: Optional Dict[case_id, KLGrade]
            target_shape: Target image shape (D, H, W)
            transform: Optional transform function
        """
        self.case_ids = case_ids
        self.images_dir = images_dir
        self.radiomics_dict = radiomics_dict
        self.labels_dict = labels_dict
        self.target_shape = target_shape
        self.transform = transform
        
        # Validate all cases have radiomics
        missing_radiomics = [cid for cid in case_ids if cid not in radiomics_dict]
        if missing_radiomics:
            logger.warning(f"Missing radiomics for {len(missing_radiomics)} cases")
    
    def __len__(self):
        return len(self.case_ids)
    
    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        
        # Load image
        # Try different naming conventions
        image_paths = [
            self.images_dir / f"{case_id}_0000.nii.gz",
            self.images_dir / f"{case_id}_0000.nii",
            self.images_dir / f"{case_id}.nii.gz",
            self.images_dir / f"{case_id}.nii"
        ]
        
        image_path = None
        for path in image_paths:
            if path.exists():
                image_path = path
                break
        
        if image_path is None:
            raise FileNotFoundError(f"Image not found for case {case_id} in {self.images_dir}")
        
        image = preprocess_image(image_path, self.target_shape)
        
        if self.transform:
            image = self.transform(image)
        
        # Get radiomics
        default_length = len(next(iter(self.radiomics_dict.values()))) if self.radiomics_dict else 535
        radiomics = self.radiomics_dict.get(case_id, np.zeros(default_length, dtype=np.float32))
        radiomics = torch.from_numpy(radiomics).float()
        
        # Get label
        if self.labels_dict is not None:
            label = self.labels_dict.get(case_id, 0)
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(-1, dtype=torch.long)  # Dummy label for inference
        
        return {
            "case_id": case_id,
            "image": image,
            "radiomics": radiomics,
            "label": label
        }


