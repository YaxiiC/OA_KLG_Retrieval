"""
Training Utilities for KLGrade Prediction

This module contains:
- compute_metrics: Calculate accuracy, macro-F1, and QWK
- train_epoch: Train for one epoch
- validate: Validate model
"""

import logging
import random
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, precision_score, 
    recall_score, balanced_accuracy_score, roc_auc_score, 
    classification_report, confusion_matrix
)

from models import JointModel, JointScoringModel

logger = logging.getLogger(__name__)


def is_ordinal_kl_setting(label_num_classes: int) -> bool:
    """
    We treat KLGrade 0-4 as ordinal ONLY in the 5-class setting.
    Binary OA (0/1 vs 2/3/4) stays as standard classification.
    """
    return int(label_num_classes) == 5


def ordinal_targets_from_labels(y: torch.Tensor, label_num_classes: int = 5) -> torch.Tensor:
    """
    Build CORAL-style ordinal targets for labels in {0..K-1}.

    For each threshold t in {0..K-2}, target is 1 if y > t else 0.

    Returns:
        targets: float tensor [B, K-1]
    """
    if label_num_classes < 3:
        raise ValueError("Ordinal targets require K>=3 classes.")
    if y.ndim != 1:
        y = y.view(-1)
    device = y.device
    k_minus_1 = label_num_classes - 1
    thresholds = torch.arange(k_minus_1, device=device, dtype=y.dtype)  # [K-1]
    # y > t
    targets = (y.unsqueeze(1) > thresholds.unsqueeze(0)).to(torch.float32)
    return targets


def ordinal_cross_entropy_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    label_num_classes: int = 5,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Ordinal cross-entropy (CORAL-style):
      - Model outputs K-1 logits for thresholds
      - Loss is BCEWithLogits across thresholds

    Args:
        logits: [B, K-1]
        y: [B] int labels in {0..K-1}
        class_weights: optional tensor [K] with per-class weights; applied per-sample via y
    """
    if y.ndim != 1:
        y = y.view(-1)
    k_minus_1 = label_num_classes - 1
    if logits.ndim != 2 or logits.shape[1] != k_minus_1:
        raise ValueError(f"Expected logits shape [B,{k_minus_1}] for ordinal KL, got {tuple(logits.shape)}")

    targets = ordinal_targets_from_labels(y, label_num_classes=label_num_classes)  # [B,K-1]
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [B,K-1]
    per_sample = bce.mean(dim=1)  # [B]
    if class_weights is not None:
        # class_weights is [K]; apply per sample using its class label
        w = class_weights[y.to(torch.long)].to(per_sample.dtype)  # [B]
        per_sample = per_sample * w
    return per_sample.mean()


def ordinal_logits_to_proba(logits: torch.Tensor, label_num_classes: int = 5) -> torch.Tensor:
    """
    Convert ordinal logits [B,K-1] into class probabilities [B,K].

    Using p(y > t) = sigmoid(logit_t):
      p0 = 1 - p_gt0
      pk = p_gt(k-1) - p_gt(k)  for 1<=k<=K-2
      p_{K-1} = p_gt(K-2)
    """
    k_minus_1 = label_num_classes - 1
    if logits.ndim != 2 or logits.shape[1] != k_minus_1:
        raise ValueError(f"Expected logits shape [B,{k_minus_1}], got {tuple(logits.shape)}")
    p_gt = torch.sigmoid(logits)  # [B,K-1]
    B = logits.shape[0]
    probs = torch.zeros((B, label_num_classes), device=logits.device, dtype=logits.dtype)
    probs[:, 0] = 1.0 - p_gt[:, 0]
    for k in range(1, label_num_classes - 1):
        probs[:, k] = p_gt[:, k - 1] - p_gt[:, k]
    probs[:, label_num_classes - 1] = p_gt[:, label_num_classes - 2]

    # Numerical safety: clamp and renormalize
    probs = torch.clamp(probs, min=0.0, max=1.0)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    return probs


def ordinal_logits_to_pred(logits: torch.Tensor) -> torch.Tensor:
    """
    Predict class by counting how many thresholds are passed (p(y>t) > 0.5).
    logits: [B,K-1] -> preds: [B] in {0..K-1}
    """
    return (torch.sigmoid(logits) > 0.5).to(torch.long).sum(dim=1)


def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_proba: Optional[np.ndarray] = None,
    return_per_class: bool = False
) -> Dict:
    """
    Compute comprehensive metrics for multiclass classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC calculation)
        return_per_class: Whether to return per-class metrics
    
    Returns:
        Dictionary with summary metrics and optionally per-class metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    n_classes = len(classes)
    
    # Summary metrics
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    if len(classes) < 2:
        # QWK is undefined when only one class is present
        qwk = float("nan")
    else:
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # AUC
    auc = None
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            # Binary: accept [N,2], [N,1], or [N] scores.
            if n_classes == 2:
                if len(np.unique(y_true)) < 2:
                    # Undefined AUC (only one class present in ground truth)
                    auc = float("nan")
                else:
                    if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                        y_score = y_proba[:, 1]
                    elif y_proba.ndim == 2 and y_proba.shape[1] == 1:
                        y_score = y_proba[:, 0]
                    elif y_proba.ndim == 1:
                        y_score = y_proba
                    else:
                        raise ValueError(f"Unexpected y_proba shape for binary AUC: {y_proba.shape}")
                    auc = roc_auc_score(y_true, y_score)
            else:
                # Multiclass: macro-averaged OVR
                if y_proba.ndim == 2 and y_proba.shape[0] == y_true.shape[0] and y_proba.shape[1] >= 3:
                    labels = list(range(y_proba.shape[1]))
                    auc = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr', labels=labels)
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")
            auc = float("nan")
    
    metrics = {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "qwk": qwk,
    }
    
    if auc is not None:
        metrics["auc"] = float(auc)
    
    # Per-class metrics
    if return_per_class:
        precision_per_class = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
        
        # Support (number of samples per class in true labels)
        # Count occurrences of each class in y_true
        support_dict = {}
        for cls in classes:
            support_dict[cls] = int(np.sum(y_true == cls))
        
        per_class_metrics = {}
        for i, cls in enumerate(classes):
            per_class_metrics[f"class_{cls}"] = {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
                "support": support_dict[cls]
            }
        
        metrics["per_class"] = per_class_metrics
    
    return metrics


def train_epoch(
    model: JointModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    warmup_epochs: int,
    lambda_k: float,
    lambda_k_start: float,
    warmup_thr_start: float,
    warmup_thr_end: float,
    lambda_diversity: float = 0.1,
    progress_bar=None
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (avg_loss, metrics, loss_components) where loss_components contains:
        - ce_loss: average classification loss
        - loss_k: average sparsity loss
        - mean_p_sum: average sum of gate probabilities
    """
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_loss_k = 0.0
    total_loss_diversity = 0.0
    total_loss_entropy = 0.0
    all_p_sums = []
    all_preds = []
    all_labels = []
    
    # Update warmup parameters
    if epoch < warmup_epochs:
        # Linear interpolation
        alpha = epoch / warmup_epochs
        current_lambda_k = lambda_k_start + alpha * (lambda_k - lambda_k_start)
        current_threshold = warmup_thr_start + alpha * (warmup_thr_end - warmup_thr_start)
        model.set_warmup_threshold(current_threshold)
        model.set_use_hard_topk(False)
        is_warmup = True
        stage = "warmup"
    else:
        current_lambda_k = lambda_k
        model.set_use_hard_topk(True)
        is_warmup = False
        stage = "hard-topk"
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        radiomics = batch["radiomics"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits, gates = model(images, radiomics, return_gates=True)
        
        # Classification loss
        loss_cls = criterion(logits, labels)
        
        # Sparsity loss (target-k regularizer)
        p = torch.sigmoid(model.selector.gate_head(model.selector.backbone(images)))
        loss_k = ((p.sum(dim=1) - model.k) ** 2).mean()
        
        # Diversity loss: encourage different feature selections across batch
        # Penalize high correlation between gate vectors of different samples
        # Compute pairwise cosine similarity between gate vectors
        p_normalized = F.normalize(p, p=2, dim=1)  # [B, n_features]
        # Compute similarity matrix: [B, B]
        similarity_matrix = torch.mm(p_normalized, p_normalized.t())
        # Remove diagonal (self-similarity) and take upper triangle
        mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
        pairwise_similarities = similarity_matrix[mask]  # [B*(B-1)/2]
        # Penalize high similarity (encourage diversity)
        loss_diversity = pairwise_similarities.mean()
        
        # Entropy regularization: encourage exploration (prevent gate collapse)
        # Higher entropy = more uniform distribution = more exploration
        # We want to maximize entropy, so we minimize negative entropy
        # Entropy: -sum(p * log(p + eps) + (1-p) * log(1-p + eps))
        eps = 1e-8
        entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)).sum(dim=1).mean()
        # Normalize by number of features to get per-feature entropy
        n_features = p.shape[1]
        max_entropy = n_features * np.log(2)  # Maximum entropy when p=0.5 for all features
        normalized_entropy = entropy / max_entropy
        # We want to encourage higher entropy, so we penalize low entropy
        # But only during warmup to allow convergence later
        if epoch < warmup_epochs:
            loss_entropy = -normalized_entropy * 0.05  # Small weight, encourage exploration
        else:
            loss_entropy = torch.tensor(0.0, device=device)
        
        # Track p.sum() for gate statistics
        all_p_sums.extend(p.sum(dim=1).detach().cpu().numpy())
        
        # Total loss
        loss = loss_cls + current_lambda_k * loss_k + lambda_diversity * loss_diversity + loss_entropy
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_ce_loss += loss_cls.item()
        total_loss_k += loss_k.item()
        total_loss_diversity += loss_diversity.item()
        total_loss_entropy += loss_entropy.item()
        
        # Update progress bar if provided
        if progress_bar is not None:
            # Compute gate variance across batch as diagnostic
            gate_variance = p.var(dim=0).mean().item()  # Average variance across features
            progress_bar.set_postfix({
                'lr': f'{current_lr:.2e}',
                'loss': f'{loss.item():.4f}',
                'ce': f'{loss_cls.item():.4f}',
                'k': f'{loss_k.item():.4f}',
                'div': f'{loss_diversity.item():.4f}',
                'p_sum': f'{p.sum(dim=1).mean().item():.2f}',
                'p_var': f'{gate_variance:.4f}',
                'stage': stage
            })
            progress_bar.update(1)
        
        # Predictions
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_loss_k = total_loss_k / len(dataloader)
    avg_loss_diversity = total_loss_diversity / len(dataloader)
    avg_loss_entropy = total_loss_entropy / len(dataloader)
    mean_p_sum = np.mean(all_p_sums) if all_p_sums else 0.0
    
    # Note: train_epoch doesn't compute probabilities, so AUC won't be available
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), y_proba=None)
    
    loss_components = {
        'ce_loss': avg_ce_loss,
        'loss_k': avg_loss_k,
        'loss_diversity': avg_loss_diversity,
        'loss_entropy': avg_loss_entropy,
        'mean_p_sum': mean_p_sum
    }
    
    return avg_loss, metrics, loss_components


def validate(
    model: JointModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    lambda_k: float = 0.05
) -> Tuple[float, Dict[str, float], Dict[str, np.ndarray], Dict[str, float]]:
    """
    Validate model.
    
    Returns:
        Tuple of (avg_loss, metrics, results, loss_components) where loss_components contains:
        - ce_loss: average classification loss
        - loss_k: average sparsity loss (if computed)
    """
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_loss_k = 0.0
    all_preds = []
    all_labels = []
    all_probas = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            radiomics = batch["radiomics"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(images, radiomics)
            loss_cls = criterion(logits, labels)
            
            # Compute loss_k for consistency (though not used in validation loss)
            p = torch.sigmoid(model.selector.gate_head(model.selector.backbone(images)))
            loss_k = ((p.sum(dim=1) - model.k) ** 2).mean()
            loss = loss_cls + lambda_k * loss_k
            
            total_loss += loss.item()
            total_ce_loss += loss_cls.item()
            total_loss_k += loss_k.item()
            
            # Probabilities/preds: handle both 2-logit CE and 1-logit BCE-style heads.
            if logits.ndim == 2 and logits.shape[1] == 1:
                prob_pos = torch.sigmoid(logits)
                probas = torch.cat([1.0 - prob_pos, prob_pos], dim=1)
                preds = (prob_pos[:, 0] > 0.5).to(torch.long).cpu().numpy()
            else:
                probas = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probas.append(probas.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_loss_k = total_loss_k / len(dataloader)
    y_true_array = np.array(all_labels)
    y_pred_array = np.array(all_preds)
    y_proba_array = np.vstack(all_probas) if all_probas else None
    
    metrics = compute_metrics(
        y_true_array,
        y_pred_array,
        y_proba_array,
        return_per_class=False
    )
    
    results = {
        "preds": y_pred_array,
        "labels": y_true_array,
        "probas": y_proba_array if y_proba_array is not None else np.array([])
    }
    
    loss_components = {
        'ce_loss': avg_ce_loss,
        'loss_k': avg_loss_k
    }
    
    return avg_loss, metrics, results, loss_components


def sample_random_subsets(
    num_features: int,
    k: int,
    n_subsets: int,
    rng: random.Random
) -> np.ndarray:
    """
    Sample random subsets of feature indices.
    Returns: array of shape [n_subsets, k]
    """
    subsets = []
    for _ in range(n_subsets):
        indices = rng.sample(range(num_features), k)
        subsets.append(indices)
    return np.array(subsets, dtype=np.int64)


def build_subset_tensors(
    radiomics: np.ndarray,
    subset_indices: np.ndarray,
    feature_to_roi: np.ndarray,
    feature_to_type: np.ndarray,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build (feature_ids, values) tensors for subset indices.
    """
    feature_ids = torch.from_numpy(subset_indices).long().to(device)
    roi_ids = torch.from_numpy(feature_to_roi[subset_indices]).long().to(device)
    type_ids = torch.from_numpy(feature_to_type[subset_indices]).long().to(device)
    values = torch.from_numpy(radiomics[subset_indices]).float().to(device)
    return feature_ids, roi_ids, type_ids, values


def compute_probe_reward(
    subset_indices: np.ndarray,
    train_case_ids: List[str],
    radiomics_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, int],
    num_classes: int,
    device: torch.device,
    n_support: int = 16,
    n_query: int = 16,
    steps: int = 3,
    lr: float = 1e-2
) -> float:
    """
    Few-step linear probe reward (negative query CE loss).
    """
    if len(train_case_ids) < (n_support + n_query):
        return 0.0

    selected = random.sample(train_case_ids, n_support + n_query)
    support_ids = selected[:n_support]
    query_ids = selected[n_support:]

    support_x = np.stack([radiomics_dict[cid][subset_indices] for cid in support_ids])
    support_y = np.array([labels_dict[cid] for cid in support_ids])
    query_x = np.stack([radiomics_dict[cid][subset_indices] for cid in query_ids])
    query_y = np.array([labels_dict[cid] for cid in query_ids])

    support_x = torch.from_numpy(support_x).float().to(device)
    support_y = torch.from_numpy(support_y).long().to(device)
    query_x = torch.from_numpy(query_x).float().to(device)
    query_y = torch.from_numpy(query_y).long().to(device)

    use_ordinal = is_ordinal_kl_setting(num_classes)
    out_dim = (num_classes - 1) if use_ordinal else num_classes

    probe = nn.Linear(subset_indices.shape[0], out_dim).to(device)
    opt = torch.optim.SGD(probe.parameters(), lr=lr)

    probe.train()
    for _ in range(steps):
        opt.zero_grad()
        logits = probe(support_x)
        if use_ordinal:
            loss = ordinal_cross_entropy_loss(
                logits, support_y, label_num_classes=num_classes, class_weights=None
            )
        else:
            loss = F.cross_entropy(logits, support_y)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(query_x)
        if use_ordinal:
            query_loss = ordinal_cross_entropy_loss(
                logits, query_y, label_num_classes=num_classes, class_weights=None
            ).item()
        else:
            query_loss = F.cross_entropy(logits, query_y).item()

    return -query_loss


def train_epoch_subset(
    model: JointScoringModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    warmup_epochs: int,
    train_case_ids: List[str],
    radiomics_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, int],
    feature_to_roi: np.ndarray,
    feature_to_type: np.ndarray,
    k: int,
    n_subsets: int,
    top_m: int,
    pool_size: int,
    num_classes: int,
    lambda_rank: float,
    exploration_ratio: float = 0.2,
    probe_support: int = 16,
    probe_query: int = 16,
    probe_steps: int = 3,
    probe_lr: float = 1e-2,
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Train one epoch with subset retrieval.
    """
    model.train()
    rng = random.Random(epoch + 12345)
    use_ordinal = is_ordinal_kl_setting(num_classes)
    total_loss = 0.0
    total_cls_loss = 0.0
    total_rank_loss = 0.0
    all_preds = []
    all_labels = []
    all_rewards = []
    all_scores = []

    num_features = len(next(iter(radiomics_dict.values())))
    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        case_ids = batch["case_id"]

        batch_loss = 0.0
        batch_cls = 0.0
        batch_rank = 0.0

        optimizer.zero_grad()
        for i in range(len(case_ids)):
            case_id = case_ids[i]
            image = images[i:i + 1]
            label = labels[i:i + 1]
            radiomics = radiomics_dict[case_id]

            is_warmup = epoch <= warmup_epochs
            if is_warmup:
                candidate_subsets = sample_random_subsets(num_features, k, n_subsets, rng)
                probe_subsets = candidate_subsets
            else:
                candidate_pool = sample_random_subsets(num_features, k, pool_size, rng)
                pool_scores = []
                with torch.no_grad():
                    for subset in candidate_pool:
                        f_ids, r_ids, t_ids, vals = build_subset_tensors(
                            radiomics, subset, feature_to_roi, feature_to_type, device
                        )
                        score = model.score_subset(
                            image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                        )
                        pool_scores.append(score.item())
                pool_scores = np.array(pool_scores)
                n_exploit = int(n_subsets * (1 - exploration_ratio))
                n_explore = n_subsets - n_exploit
                exploit_indices = np.argsort(pool_scores)[-n_exploit:] if n_exploit > 0 else np.array([], dtype=int)
                explore_indices = rng.sample(range(pool_size), n_explore) if n_explore > 0 else []
                selected_indices = list(exploit_indices) + list(explore_indices)
                probe_subsets = candidate_pool[selected_indices]
                candidate_subsets = candidate_pool

            rewards = []
            scores = []
            for subset in probe_subsets:
                reward = compute_probe_reward(
                    subset,
                    train_case_ids,
                    radiomics_dict,
                    labels_dict,
                    num_classes=num_classes,
                    device=device,
                    n_support=probe_support,
                    n_query=probe_query,
                    steps=probe_steps,
                    lr=probe_lr
                )
                rewards.append(reward)
                f_ids, r_ids, t_ids, vals = build_subset_tensors(
                    radiomics, subset, feature_to_roi, feature_to_type, device
                )
                score = model.score_subset(
                    image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                )
                scores.append(score)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            scores_tensor = torch.cat(scores, dim=0)
            rank_loss = F.mse_loss(scores_tensor, rewards_tensor)
            all_rewards.extend(rewards)
            all_scores.extend(scores_tensor.detach().cpu().numpy().tolist())

            # Classifier training using TopM subsets by scorer
            with torch.no_grad():
                subset_scores = []
                for subset in candidate_subsets:
                    f_ids, r_ids, t_ids, vals = build_subset_tensors(
                        radiomics, subset, feature_to_roi, feature_to_type, device
                    )
                    score = model.score_subset(
                        image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                    )
                    subset_scores.append(score.item())
                subset_scores = np.array(subset_scores)
                topm_indices = np.argsort(subset_scores)[-top_m:]

            logits_list = []
            for idx in topm_indices:
                subset = candidate_subsets[idx]
                f_ids, r_ids, t_ids, vals = build_subset_tensors(
                    radiomics, subset, feature_to_roi, feature_to_type, device
                )
                logits = model.classify_subset(
                    image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                )
                logits_list.append(logits)

            logits_ensemble = torch.mean(torch.cat(logits_list, dim=0), dim=0, keepdim=True)
            if use_ordinal:
                cls_loss = ordinal_cross_entropy_loss(
                    logits_ensemble,
                    label.view(-1),
                    label_num_classes=num_classes,
                    class_weights=class_weights,
                )
            else:
                cls_loss = F.cross_entropy(logits_ensemble, label.view(-1), weight=class_weights)

            loss = cls_loss + lambda_rank * rank_loss
            batch_loss += loss
            batch_cls += cls_loss.item()
            batch_rank += rank_loss.item()

            if use_ordinal:
                preds = ordinal_logits_to_pred(logits_ensemble).cpu().numpy()
            else:
                preds = logits_ensemble.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())

        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        total_cls_loss += batch_cls
        total_rank_loss += batch_rank

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), y_proba=None)
    if all_rewards:
        rewards_arr = np.array(all_rewards, dtype=np.float32)
        scores_arr = np.array(all_scores, dtype=np.float32)
        metrics["probe_reward_mean"] = float(np.mean(rewards_arr))
        metrics["probe_reward_std"] = float(np.std(rewards_arr))
        metrics["probe_reward_min"] = float(np.min(rewards_arr))
        metrics["probe_reward_max"] = float(np.max(rewards_arr))
        metrics["scorer_score_mean"] = float(np.mean(scores_arr))
        metrics["scorer_score_std"] = float(np.std(scores_arr))
    avg_loss = total_loss / len(dataloader)
    avg_cls = total_cls_loss / len(dataloader)
    avg_rank = total_rank_loss / len(dataloader)
    metrics["cls_loss"] = avg_cls
    metrics["rank_loss"] = avg_rank
    return avg_loss, metrics


def validate_subset(
    model: JointScoringModel,
    dataloader: DataLoader,
    device: torch.device,
    radiomics_dict: Dict[str, np.ndarray],
    feature_to_roi: np.ndarray,
    feature_to_type: np.ndarray,
    k: int,
    top_m: int,
    pool_size: int
) -> Tuple[float, Dict[str, float]]:
    """
    Validate using scorer-selected TopM subsets (no probe).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    rng = random.Random(123)
    num_features = len(next(iter(radiomics_dict.values())))

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            case_ids = batch["case_id"]

            for i in range(len(case_ids)):
                case_id = case_ids[i]
                image = images[i:i + 1]
                label = labels[i:i + 1]
                radiomics = radiomics_dict[case_id]

                candidate_subsets = sample_random_subsets(num_features, k, pool_size, rng)
                subset_scores = []
                for subset in candidate_subsets:
                    f_ids, r_ids, t_ids, vals = build_subset_tensors(
                        radiomics, subset, feature_to_roi, feature_to_type, device
                    )
                    score = model.score_subset(
                        image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                    )
                    subset_scores.append(score.item())
                subset_scores = np.array(subset_scores)
                topm_indices = np.argsort(subset_scores)[-top_m:]

                logits_list = []
                for idx in topm_indices:
                    subset = candidate_subsets[idx]
                    f_ids, r_ids, t_ids, vals = build_subset_tensors(
                        radiomics, subset, feature_to_roi, feature_to_type, device
                    )
                    logits = model.classify_subset(
                        image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                    )
                    logits_list.append(logits)
                logits_ensemble = torch.mean(torch.cat(logits_list, dim=0), dim=0, keepdim=True)

                # Infer ordinal vs nominal from logits dim (4 -> ordinal KL 5-class).
                if logits_ensemble.shape[1] == 4:
                    loss = ordinal_cross_entropy_loss(
                        logits_ensemble,
                        label.view(-1),
                        label_num_classes=5,
                        class_weights=None,
                    )
                    preds = ordinal_logits_to_pred(logits_ensemble).cpu().numpy()
                else:
                    loss = F.cross_entropy(logits_ensemble, label.view(-1))
                    preds = logits_ensemble.argmax(dim=1).cpu().numpy()
                total_loss += loss.item()

                all_preds.extend(preds)
                all_labels.extend(label.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), y_proba=None)
    avg_loss = total_loss / len(dataloader)
    metrics["cls_loss"] = avg_loss
    return avg_loss, metrics

