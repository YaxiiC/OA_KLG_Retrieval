# OA KLGrade Subset Retrieval 

## English

### Project Overview

This project predicts **Kellgren-Lawrence Grade (KLGrade 0-4)** from **3D knee MRI** using a **patient-specific radiomics subset retrieval** framework. Instead of gating each feature independently, we treat a **feature subset** as the unit and learn to **score and retrieve the best subsets** for each patient.

Current implementation:
- **Subset Scorer**: ranks candidate subsets for each patient
- **Token-Set KL Classifier**: predicts KLGrade from selected subsets
- **Few-step linear probe** (training-only): provides Reward B for scorer supervision
- **Budgeted retrieval** at inference (recall + rank + local search)

Legacy gate-based selection remains in `train_selector_klgrade.py`.

### Architecture (Current Implementation)

#### 1) Image Encoder (`models.py`)
- 3D CNN encoder
- Input: MRI `[B,1,D,H,W]`
- Output: `z_img ∈ R^d`

#### 2) Token-Set Encoder (`models.py`)
- Token = `(feature_id, value)` for each feature in a subset
- ID embedding + value MLP + DeepSets pooling
- Output: `z_set ∈ R^d`

#### 3) Subset Scorer (`models.py`)
- Input: `[z_img, z_set]`
- Output: `score ∈ R` (higher is better)

#### 4) Token-Set KL Classifier (`models.py`)
- Input: `z_set` (subset embedding only)
- Logistic regression (single linear layer)
- Output: logits `[B,num_classes]`

### Training Loss

```
Total Loss = L_cls + λ_rank × L_rank
```

- **L_cls**: KL cross-entropy on TopM ensemble logits
- **L_rank**: scorer regression loss vs Reward B (MSE)

### Training Strategy (Two Stages)

#### Stage 1: Warmup (epochs 1..T)
- Sample `N_subsets` randomly per patient
- Run probe on all `N_subsets` → Reward B
- Train scorer with `L_rank`
- Train classifier on TopM subsets by scorer (ensemble logits)

#### Stage 2: Main Training (after T)
- Sample `PoolSize` candidates per patient
- Scorer ranks pool
- Select `N_subsets` for probe (top-ranked + random exploration)
- Train scorer with probe rewards
- Train classifier on TopM subsets by scorer (ensemble logits)

### Data Pipeline

1. Segmentation: `nnunet_segmentation_inference.py`
2. Radiomics extraction: `torchradiomics_from_ROIs.py`
3. Radiomics format: **wide format** recommended (one row per case)
4. Training: `train_joint_scoring_kl.py`
5. Inference: `infer_budgeted_retrieval.py`

### Data Format (Radiomics Wide CSV)

Each row is one case. Required columns:
- `case_id`: unique identifier (must match image filename stem)
- `KLGrade`: integer 0..4 (only required for training)
- feature columns: numeric radiomics values

Notes:
- Wide format is required for subset retrieval.
- Missing values are supported; they are filled during preprocessing.
- `case_id` should align with `imagesTr/imagesTs` filenames (without extension).

### Image Folder Convention

- 3D images should be in `imagesTr/` or `imagesTs/` with nnU-Net style naming.
- Only the filename stem is used for matching to `case_id`.
- Input shape is `[B,1,D,H,W]` after preprocessing.

### Key Scripts (What They Do)

- `train_joint_scoring_kl.py`: trains the scorer + classifier jointly with probe-based rewards.
- `infer_budgeted_retrieval.py`: performs budgeted retrieval and outputs predictions + selected subsets.
- `train_selector_klgrade.py`: legacy gate-based baseline.
- `torchradiomics_from_ROIs.py`: extracts radiomics from ROIs into CSV.
- `nnunet_segmentation_inference.py`: generates ROI masks for radiomics.

### File Structure

```
OA_KLG_Retrieval/
├── train_joint_scoring_kl.py     # Subset scorer + classifier training
├── infer_budgeted_retrieval.py   # Budgeted inference pipeline
├── train_selector_klgrade.py     # Legacy gate-based training
├── models.py                     # Model definitions
├── training_utils.py             # Training utilities (subset training, metrics)
├── data_loader.py                # Data loading and preprocessing
├── nnunet_segmentation_inference.py
├── torchradiomics_from_ROIs.py
├── environment.yml
├── output_train/
│   ├── radiomics_results.csv
│   └── radiomics_results_wide.csv
├── output_test/
│   ├── radiomics_results.csv
│   └── radiomics_results_wide.csv
└── training_logs/
```

### Usage

#### 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate oa_klg_topk
```

#### 2. Train (Subset Retrieval)
```powershell
python train_joint_scoring_kl.py `
    --images-tr "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTr" `
    --radiomics-train-csv "C:\Users\chris\MICCAI2026\OA_KLG_Retrieval\output_train\radiomics_results_wide.csv" `
    --klgrade-train-csv "C:\Users\chris\MICCAI2026\OA_KLG_Retrieval\subInfo_train.xlsx" `
    --outdir "training_logs_subset" `
    --k 15 `
    --n-subsets 32 `
    --top-m 4 `
    --pool-size 320 `
    --warmup-epochs 20 `
    --epochs 200 `
    --lambda-rank 0.1 `
    --exploration-ratio 0.2
```

#### 3. Inference (Budgeted Retrieval)
```powershell
python infer_budgeted_retrieval.py `
    --images-ts "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTs" `
    --radiomics-test-csv "C:\Users\chris\MICCAI2026\OA_KLG_Retrieval\output_test\radiomics_results_wide.csv" `
    --checkpoint "training_logs_subset\checkpoints\best.pth" `
    --scaler "training_logs_subset\checkpoints\scaler.joblib" `
    --outdir "inference_subset"
```

### Key Hyperparameters

- `--k`: subset size K
- `--n-subsets`: number of subsets for probe supervision
- `--top-m`: number of top subsets for classifier training
- `--pool-size`: candidate pool size after warmup
- `--warmup-epochs`: warmup epochs T
- `--lambda-rank`: weight for scorer ranking loss
- `--exploration-ratio`: fraction of random subsets after warmup
- `--probe-support`, `--probe-query`, `--probe-steps`, `--probe-lr`
- `--label-mode`: `multiclass` (KL 0-4) or `binary_oa` (0/1→non-OA, 2/3/4→OA)

### Output Files

#### Training Metrics (`logs/metrics.csv`)
- `epoch`, `train_loss`, `train_cls_loss`, `train_rank_loss`
- `val_loss`, `val_macro_f1`, `val_qwk`, `val_acc`

#### Predictions (`predictions_test.csv`)
- `case_id`, `pred_class`, `prob_0`..`prob_4`

#### Selected Subsets (`selected_subsets_test.json`)
- `case_id`, `final_top_indices` (K feature indices per subset)

### Evaluation Metrics

- Accuracy
- Balanced Accuracy
- Macro F1
- Weighted F1
- QWK (Quadratic Weighted Kappa)




