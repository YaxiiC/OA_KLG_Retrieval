"""
nnU-Net Segmentation Inference with Majority Voting

This script:
1. Runs inference on each fold separately (0, 1, 2, 3, 4)
2. Performs majority voting across folds to get final segmentation
3. Saves only the majority-voted mask
4. Calculates Dice score for all ROIs
5. Processes both train and test sets

Author: Created for Dataset360_oaizib segmentation

python nnunet_segmentation_inference.py \
    --dataset-dir "/home/yaxi/nnUNet/nnUNet_raw/Dataset360_oaizib" \
    --model-dir "/home/yaxi/nnUNet/nnUNet_results/Dataset360_oaizib/nnUNetTrainer__nnUNetPlans__3d_fullres" \
    --output-dir "/home/yaxi/nnUNet/nnUNet_results" \
    --split both \
    --folds 0 1 2 3 4 
"""

import torch
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from scipy import stats
import SimpleITK as sitk

# nnU-Net imports
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Individual Fold Prediction
# ============================================================================

def predict_single_fold(
    case_image_path: Path,
    model_dir: Path,
    fold: int,
    device: torch.device,
    checkpoint_name: str = 'checkpoint_best.pth'
) -> Tuple[np.ndarray, Dict]:
    """
    Predict segmentation mask using a single fold.
    
    Args:
        case_image_path: Path to input image (NIfTI)
        model_dir: Path to nnU-Net model directory (contains fold_X subdirs)
        fold: Fold index (0, 1, 2, 3, or 4)
        device: PyTorch device
        checkpoint_name: Checkpoint filename
    
    Returns:
        Tuple of (segmentation_array, image_properties_dict)
    """
    logger.info(f"Predicting with fold {fold}...")
    
    # Initialize predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        allow_tqdm=False
    )
    
    # Initialize from trained model folder (single fold)
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=[fold],  # Single fold
        checkpoint_name=checkpoint_name
    )
    
    # Read image
    image_reader = SimpleITKIO()
    image_data, image_properties = image_reader.read_images([str(case_image_path)])
    
    # Preprocess
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    preprocessed_data, _, data_properties = preprocessor.run_case_npy(
        image_data,
        None,
        image_properties,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.dataset_json
    )
    
    # Convert to tensor
    preprocessed_tensor = torch.from_numpy(preprocessed_data).to(
        dtype=torch.float32,
        device=device,
        memory_format=torch.contiguous_format
    )
    
    # Predict logits
    predicted_logits = predictor.predict_logits_from_preprocessed_data(preprocessed_tensor)
    
    # Convert logits to segmentation
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    segmentation = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_logits.cpu(),
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.label_manager,
        data_properties,
        return_probabilities=False
    )
    
    # Ensure segmentation is 3D numpy array
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    
    return segmentation, image_properties


# ============================================================================
# Evaluation Metrics
# ============================================================================

def calculate_dice(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """
    Calculate Dice score for a specific label.
    
    Args:
        pred: Prediction array
        gt: Ground truth array
        label: Label value to evaluate
    
    Returns:
        Dice score (0-1, higher is better)
    """
    pred_mask = (pred == label).astype(np.float32)
    gt_mask = (gt == label).astype(np.float32)
    
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / union
    return float(dice)


def calculate_metrics_all_rois(
    pred: np.ndarray,
    gt: np.ndarray,
    roi_labels: Optional[Dict[int, str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate Dice score for all ROIs.
    
    Args:
        pred: Prediction array
        gt: Ground truth array
        roi_labels: Optional dict mapping label value -> ROI name
    
    Returns:
        Dictionary mapping ROI name -> {'dice': float, 'label': int}
    """
    # Get unique labels (excluding background 0)
    unique_labels = np.unique(gt)
    unique_labels = unique_labels[unique_labels > 0]
    
    if roi_labels is None:
        roi_labels = {int(label): f'ROI_{int(label)}' for label in unique_labels}
    
    metrics = {}
    
    for label in unique_labels:
        label_int = int(label)
        roi_name = roi_labels.get(label_int, f'ROI_{label_int}')
        
        dice = calculate_dice(pred, gt, label_int)
        
        metrics[roi_name] = {
            'dice': dice,
            'label': label_int
        }
    
    return metrics


# ============================================================================
# Majority Voting
# ============================================================================

def majority_vote(segmentations: List[np.ndarray]) -> np.ndarray:
    """
    Perform majority voting across multiple segmentation predictions.
    Uses numpy operations for efficient computation.
    
    Args:
        segmentations: List of segmentation arrays (all same shape)
    
    Returns:
        Majority-voted segmentation array
    """
    if len(segmentations) == 0:
        raise ValueError("No segmentations provided for majority voting")
    
    if len(segmentations) == 1:
        return segmentations[0].copy()
    
    # Stack all segmentations
    stacked = np.stack(segmentations, axis=0)  # Shape: [n_folds, z, y, x]
    
    logger.info("Performing majority voting...")
    
    # Use scipy.stats.mode for efficient majority voting
    result, _ = stats.mode(stacked, axis=0, keepdims=False)
    
    # Remove the extra dimension and convert to same dtype
    result = result.astype(stacked.dtype)
    
    return result


# ============================================================================
# Process Single Case
# ============================================================================

def process_case(
    case_image_path: Path,
    model_dir: Path,
    output_dir: Path,
    gt_label_path: Optional[Path],
    folds: List[int],
    device: torch.device,
    checkpoint_name: str = 'checkpoint_best.pth',
    roi_labels: Optional[Dict[int, str]] = None
) -> Dict:
    """
    Process a single case: predict with each fold, perform majority voting,
    save prediction, and calculate metrics.
    
    Args:
        case_image_path: Path to input image
        model_dir: Path to nnU-Net model directory
        output_dir: Output directory for predictions
        gt_label_path: Path to ground truth label (None if not available)
        folds: List of fold indices
        device: PyTorch device
        checkpoint_name: Checkpoint filename
        roi_labels: Optional dict mapping label value -> ROI name
    
    Returns:
        Dictionary with saved path and metrics
    """
    case_id = case_image_path.stem.replace('.nii', '').replace('.gz', '').replace('_0000', '')
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing case: {case_id}")
    logger.info(f"{'='*80}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Predict with each fold
    segmentations = []
    image_properties = None
    
    for fold in folds:
        try:
            seg, img_props = predict_single_fold(
                case_image_path,
                model_dir,
                fold,
                device,
                checkpoint_name
            )
            segmentations.append(seg)
            if image_properties is None:
                image_properties = img_props
        
        except Exception as e:
            logger.error(f"Error predicting with fold {fold}: {e}", exc_info=True)
            continue
    
    if len(segmentations) == 0:
        logger.error(f"No successful predictions for case {case_id}")
        return {'case_id': case_id, 'success': False}
    
    # Perform majority voting
    if len(segmentations) > 1:
        logger.info(f"Performing majority voting across {len(segmentations)} folds...")
        majority_seg = majority_vote(segmentations)
    else:
        majority_seg = segmentations[0]
        logger.info("Only one fold available, using as prediction")
    
    # Save majority-voted prediction
    majority_output_path = output_dir / f"{case_id}_majority_vote.nii.gz"
    image_reader = SimpleITKIO()
    data_properties = {'sitk_stuff': image_properties.get('sitk_stuff', {})}
    image_reader.write_seg(majority_seg, str(majority_output_path), data_properties)
    logger.info(f"Saved majority-voted prediction to {majority_output_path}")
    
    result = {
        'case_id': case_id,
        'success': True,
        'prediction_path': str(majority_output_path)
    }
    
    # Calculate metrics if ground truth is available
    if gt_label_path is not None and gt_label_path.exists():
        logger.info(f"Loading ground truth from {gt_label_path}")
        gt_sitk = sitk.ReadImage(str(gt_label_path))
        gt_array = sitk.GetArrayFromImage(gt_sitk)
        
        # Calculate metrics for all ROIs
        metrics = calculate_metrics_all_rois(majority_seg, gt_array, roi_labels)
        result['metrics'] = metrics
        
        # Print metrics
        logger.info("\nMetrics:")
        logger.info(f"{'ROI':<30} {'Dice':<10}")
        logger.info("-" * 45)
        for roi_name, roi_metrics in metrics.items():
            dice = roi_metrics['dice']
            logger.info(f"{roi_name:<30} {dice:<10.4f}")
    else:
        logger.warning(f"Ground truth not found at {gt_label_path}, skipping metrics calculation")
        result['metrics'] = None
    
    return result


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='nnU-Net Segmentation Inference with Majority Voting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--images-dir',
        type=str,
        default=None,
        help='Directory containing input images (NIfTI format). If not provided, uses dataset-dir/imagesTr or imagesTs'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default=r'C:\Users\chris\MICCAI2026\nnUNet\nnUNet_results\Dataset360_oaizib\nnUNetTrainer__nnUNetPlans__3d_fullres',
        help='Path to nnU-Net model directory (contains fold_X subdirectories)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=r'C:\Users\chris\MICCAI2026\nnUNet\nnUNet_results',
        help='Output directory for predicted masks'
    )
    
    parser.add_argument(
        '--folds',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4],
        help='Folds to use for inference and majority voting'
    )
    
    parser.add_argument(
        '--checkpoint-name',
        type=str,
        default='checkpoint_best.pth',
        help='Checkpoint filename to load'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='PyTorch device (e.g., cuda:0, cpu). Auto-detected if not specified'
    )
    
    parser.add_argument(
        '--output-subdir',
        type=str,
        default='predicted_masks',
        help='Subdirectory name within output-dir to save predictions'
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=r'C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib',
        help='Path to dataset directory (contains labelsTr and labelsTs)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'test', 'both'],
        default='both',
        help='Which split to process (train, test, or both)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Folds: {args.folds}")
    
    # Validate model directory
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Check that fold directories exist
    for fold in args.folds:
        fold_dir = model_dir / f"fold_{fold}"
        if not fold_dir.exists():
            logger.warning(f"Fold {fold} directory not found: {fold_dir}")
    
    # Load ROI labels from dataset.json if available
    dataset_dir = Path(args.dataset_dir)
    roi_labels = None
    dataset_json_path = dataset_dir / 'dataset.json'
    if dataset_json_path.exists():
        import json
        with open(dataset_json_path, 'r') as f:
            dataset_json = json.load(f)
            if 'labels' in dataset_json:
                # Reverse mapping: label value -> name
                roi_labels = {v: k.replace(' ', '_') for k, v in dataset_json['labels'].items() if v > 0}
                logger.info(f"Loaded ROI labels: {roi_labels}")
    
    # Determine which splits to process
    splits_to_process = []
    if args.split == 'both':
        splits_to_process = ['train', 'test']
    else:
        splits_to_process = [args.split]
    
    # Process each split
    all_results = []
    all_metrics_summary = {}
    
    for split in splits_to_process:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {split.upper()} split")
        logger.info(f"{'='*80}")
        
        # Setup directories
        if split == 'train':
            images_dir = dataset_dir / 'imagesTr'
            labels_dir = dataset_dir / 'labelsTr'
        else:  # test
            images_dir = dataset_dir / 'imagesTs'
            labels_dir = dataset_dir / 'labelsTs'
        
        # If images-dir is provided, use it instead
        if args.images_dir:
            images_dir = Path(args.images_dir)
        
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}, skipping {split} split")
            continue
        
        if not labels_dir.exists():
            logger.warning(f"Labels directory not found: {labels_dir}, skipping metrics for {split} split")
            labels_dir = None
        
        # Find all images
        image_files = sorted(images_dir.glob('*.nii.gz')) + sorted(images_dir.glob('*.nii'))
        
        if len(image_files) == 0:
            logger.warning(f"No images found in {images_dir}")
            continue
        
        logger.info(f"Found {len(image_files)} images in {split} split")
        
        # Setup output directory
        output_dir = Path(args.output_dir) / args.output_subdir / split
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Predictions will be saved to: {output_dir}")
        
        # Process each case
        successful = 0
        failed = 0
        split_metrics = {}
        
        for idx, image_path in enumerate(image_files, 1):
            logger.info(f"\n[{idx}/{len(image_files)}] Processing {image_path.name}")
            
            # Find corresponding ground truth label
            gt_label_path = None
            if labels_dir:
                # Try exact match first
                gt_label_path = labels_dir / image_path.name
                if not gt_label_path.exists():
                    # Try removing _0000 suffix
                    label_name = image_path.name.replace('_0000.nii.gz', '.nii.gz').replace('_0000.nii', '.nii')
                    gt_label_path = labels_dir / label_name
                if not gt_label_path.exists():
                    logger.warning(f"Ground truth not found for {image_path.name}")
            
            try:
                result = process_case(
                    image_path,
                    model_dir,
                    output_dir,
                    gt_label_path,
                    args.folds,
                    device,
                    args.checkpoint_name,
                    roi_labels
                )
                
                if result.get('success', False):
                    successful += 1
                    all_results.append(result)
                    
                    # Aggregate metrics
                    if result.get('metrics'):
                        for roi_name, roi_metrics in result['metrics'].items():
                            if roi_name not in split_metrics:
                                split_metrics[roi_name] = {'dice': []}
                            split_metrics[roi_name]['dice'].append(roi_metrics['dice'])
                else:
                    failed += 1
                    logger.warning(f"Failed to process {image_path.name}")
            
            except Exception as e:
                failed += 1
                logger.error(f"Error processing {image_path.name}: {e}", exc_info=True)
                continue
        
        # Print summary for this split
        logger.info(f"\n{split.upper()} Split Summary:")
        logger.info(f"  Total cases: {len(image_files)}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        
        if split_metrics:
            logger.info(f"\n{split.upper()} Split - Average Metrics:")
            logger.info(f"{'ROI':<30} {'Mean Dice':<12} {'Std Dice':<12}")
            logger.info("-" * 55)
            for roi_name in sorted(split_metrics.keys()):
                dice_values = split_metrics[roi_name]['dice']
                mean_dice = np.mean(dice_values)
                std_dice = np.std(dice_values)
                logger.info(f"{roi_name:<30} {mean_dice:<12.4f} {std_dice:<12.4f}")
            all_metrics_summary[split] = split_metrics
    
    # Overall summary
    logger.info("\n" + "="*80)
    logger.info("OVERALL SUMMARY")
    logger.info("="*80)
    logger.info(f"Total cases processed: {len(all_results)}")
    logger.info(f"Output directory: {Path(args.output_dir) / args.output_subdir}")
    
    if all_metrics_summary:
        logger.info("\nOverall Average Metrics (across all splits):")
        logger.info(f"{'ROI':<30} {'Mean Dice':<12}")
        logger.info("-" * 45)
        
        # Aggregate across all splits
        overall_metrics = {}
        for split, split_metrics in all_metrics_summary.items():
            for roi_name, roi_metrics in split_metrics.items():
                if roi_name not in overall_metrics:
                    overall_metrics[roi_name] = {'dice': []}
                overall_metrics[roi_name]['dice'].extend(roi_metrics['dice'])
        
        for roi_name in sorted(overall_metrics.keys()):
            dice_values = overall_metrics[roi_name]['dice']
            mean_dice = np.mean(dice_values)
            logger.info(f"{roi_name:<30} {mean_dice:<12.4f}")
    
    logger.info("\nDone!")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()

