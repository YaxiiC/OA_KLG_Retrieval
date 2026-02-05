"""
Model Definitions for Image-Conditioned Feature Selector + KLGrade Classifier

This module contains:
- FeatureSelector: 3D CNN-based feature selector
- KLGradeClassifier: Simple linear classifier for KLGrade
- JointModel: Combined selector + classifier with two-stage gating
"""

import logging
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import torchvision for pretrained models
try:
    import torchvision.models.video as video_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    logging.warning("torchvision not available, pretrained models disabled")

logger = logging.getLogger(__name__)


def _drop_connect(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """
    Stochastic depth / drop connect for residual blocks.
    """
    if (not training) or drop_prob <= 0.0:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # [B,1,1,1,1] for 3D tensors
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    return x / keep_prob * binary_mask


class ConvBNAct3D(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ):
        padding = kernel_size // 2
        layers = [
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm3d(out_ch),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class SqueezeExcite3D(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        reduced = max(1, int(channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.reduce = nn.Conv3d(channels, reduced, kernel_size=1)
        self.expand = nn.Conv3d(reduced, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = F.silu(self.reduce(s))
        s = torch.sigmoid(self.expand(s))
        return x * s


class MBConv3D(nn.Module):
    """
    EfficientNet-style MBConv block generalized to 3D.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand_ratio: int,
        kernel_size: int,
        stride: int,
        se_ratio: float = 0.25,
        drop_connect: float = 0.0,
    ):
        super().__init__()
        self.use_residual = (stride == 1) and (in_ch == out_ch)
        self.drop_connect = float(drop_connect)

        mid_ch = in_ch * expand_ratio
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct3D(in_ch, mid_ch, kernel_size=1, stride=1, act=True))
        else:
            mid_ch = in_ch

        # depthwise
        layers.append(
            ConvBNAct3D(
                mid_ch,
                mid_ch,
                kernel_size=kernel_size,
                stride=stride,
                groups=mid_ch,
                act=True,
            )
        )
        self.conv = nn.Sequential(*layers)
        self.se = SqueezeExcite3D(mid_ch, se_ratio=se_ratio) if se_ratio and se_ratio > 0 else nn.Identity()
        self.project = ConvBNAct3D(mid_ch, out_ch, kernel_size=1, stride=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = _drop_connect(out, self.drop_connect, self.training)
            out = out + x
        return out


class EfficientNet3DBackbone(nn.Module):
    """
    Lightweight 3D EfficientNet-B0 style network (no pretrained weights).

    Input: [B, C, D, H, W]
    Output: logits [B, out_dim]
    """

    def __init__(self, in_channels: int = 1, out_dim: int = 2, drop_rate: float = 0.2, drop_connect_rate: float = 0.2):
        super().__init__()

        # EfficientNet-B0-like config: (expand, out_ch, repeats, k, stride)
        cfg = [
            (1, 16, 1, 3, 1),
            (6, 24, 2, 3, 2),
            (6, 40, 2, 5, 2),
            (6, 80, 3, 3, 2),
            (6, 112, 3, 5, 1),
            (6, 192, 4, 5, 2),
            (6, 320, 1, 3, 1),
        ]

        stem_out = 32
        self.stem = ConvBNAct3D(in_channels, stem_out, kernel_size=3, stride=2, act=True)

        blocks = []
        total_blocks = sum(r for _, _, r, _, _ in cfg)
        block_idx = 0
        in_ch = stem_out
        for expand, out_ch, repeats, k, s in cfg:
            for i in range(repeats):
                stride = s if i == 0 else 1
                dc = drop_connect_rate * (block_idx / max(1, total_blocks - 1))
                blocks.append(
                    MBConv3D(
                        in_ch=in_ch,
                        out_ch=out_ch,
                        expand_ratio=expand,
                        kernel_size=k,
                        stride=stride,
                        se_ratio=0.25,
                        drop_connect=dc,
                    )
                )
                in_ch = out_ch
                block_idx += 1
        self.blocks = nn.Sequential(*blocks)

        head_ch = 1280
        self.head = ConvBNAct3D(in_ch, head_ch, kernel_size=1, stride=1, act=True)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=float(drop_rate)) if drop_rate and drop_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(head_ch, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


class FeatureSelector(nn.Module):
    """3D CNN-based feature selector."""
    
    def __init__(
        self,
        n_features: int = 535,
        pretrained: bool = False,
        backbone_name: str = "r3d_18"
    ):
        """
        Args:
            n_features: Number of radiomics features to gate
            pretrained: Whether to use pretrained weights
            backbone_name: Backbone model name
        """
        super().__init__()
        self.n_features = n_features
        
        if not HAS_TORCHVISION and pretrained:
            logger.warning("torchvision not available, using random initialization")
            pretrained = False
        
        # Backbone: 3D CNN
        if HAS_TORCHVISION and backbone_name == "r3d_18" and pretrained:
            backbone = video_models.r3d_18(pretrained=True)
            # Replace first conv for 1-channel input
            old_conv = backbone.stem[0]
            new_conv = nn.Conv3d(
                1, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            # Initialize by averaging RGB weights
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias.data = old_conv.bias.data
            backbone.stem[0] = new_conv
            # Get feature dimension
            backbone.fc = nn.Identity()  # Remove classifier
            # Test forward pass to get feature dim
            with torch.no_grad():
                test_input = torch.randn(1, 1, 32, 128, 128)
                test_output = backbone(test_input)
                feature_dim = test_output.shape[1]
        else:
            # Simple 3D CNN backbone
            feature_dim = 512
            backbone = nn.Sequential(
                nn.Conv3d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
                nn.Conv3d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(128, feature_dim),
                nn.ReLU(inplace=True)
            )
        
        self.backbone = backbone
        
        # Gate head: outputs logits for each feature
        self.gate_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_features)
        )
        
        # Initialize gate head with small random weights to encourage diversity
        # This prevents the model from collapsing to a fixed feature set
        for m in self.gate_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Smaller gain for more diversity
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Tensor of shape [B, 1, D, H, W]
        
        Returns:
            Gate logits of shape [B, n_features]
        """
        features = self.backbone(image)  # [B, feature_dim]
        gate_logits = self.gate_head(features)  # [B, n_features]
        return gate_logits


class KLGradeClassifier(nn.Module):
    """
    Logistic Regression Classifier for KLGrade.
    
    This implements multiclass logistic regression (softmax regression) using
    a linear layer. When combined with CrossEntropyLoss, this is equivalent to
    standard logistic regression for multiclass classification.
    
    The model learns: logits = X @ W + b, where:
    - X: input features (masked radiomics) [B, n_features]
    - W: weight matrix [n_features, n_classes]
    - b: bias vector [n_classes]
    
    Probabilities are computed via softmax: P(class_i) = exp(logits_i) / sum(exp(logits))
    """
    
    def __init__(self, n_features: int = 535, n_classes: int = 5):
        """
        Args:
            n_features: Number of input features (selected radiomics)
            n_classes: Number of output classes (KLGrade 0-4)
        """
        super().__init__()
        # Linear layer implements: y = XW^T + b
        # This is the standard logistic regression formulation
        self.classifier = nn.Linear(n_features, n_classes)
    
    def forward(self, masked_radiomics: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: computes logits for multiclass logistic regression.
        
        Args:
            masked_radiomics: Tensor of shape [B, n_features]
        
        Returns:
            Logits of shape [B, n_classes] (before softmax)
            Note: Use F.softmax(logits, dim=1) to get probabilities
        """
        return self.classifier(masked_radiomics)


class JointModel(nn.Module):
    """Joint feature selector + classifier model."""
    
    def __init__(
        self,
        n_features: int = 535,
        n_classes: int = 5,
        pretrained: bool = False,
        k: int = 15,
        warmup_threshold: float = 0.0,
        use_hard_topk: bool = False
    ):
        """
        Args:
            n_features: Number of radiomics features
            n_classes: Number of KLGrade classes
            pretrained: Whether to use pretrained backbone
            k: Top-k for hard gating
            warmup_threshold: Threshold for soft gating during warmup
            use_hard_topk: Whether to use hard top-k (True) or soft gating (False)
        """
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.k = k
        self.warmup_threshold = warmup_threshold
        self.use_hard_topk = use_hard_topk
        
        self.selector = FeatureSelector(n_features, pretrained)
        self.classifier = KLGradeClassifier(n_features, n_classes)
    
    def forward(
        self,
        image: torch.Tensor,
        radiomics: torch.Tensor,
        return_gates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            image: Tensor of shape [B, 1, D, H, W]
            radiomics: Tensor of shape [B, n_features]
            return_gates: Whether to return gate values
        
        Returns:
            If return_gates=False: logits [B, n_classes]
            If return_gates=True: (logits, gates) where gates is [B, n_features]
        """
        # Get gate logits
        gate_logits = self.selector(image)  # [B, n_features]
        p = torch.sigmoid(gate_logits)  # [B, n_features]
        
        # Apply gating
        if self.use_hard_topk:
            # Hard top-k with straight-through
            _, topk_indices = torch.topk(p, self.k, dim=1)  # [B, k]
            hard_mask = torch.zeros_like(p)
            hard_mask.scatter_(1, topk_indices, 1.0)
            # Straight-through estimator
            mask = hard_mask - p.detach() + p
            masked_radiomics = radiomics * mask
        else:
            # Soft thresholding during warmup
            p_thr = F.relu(p - self.warmup_threshold)
            masked_radiomics = radiomics * p_thr
        
        # Classify
        logits = self.classifier(masked_radiomics)
        
        if return_gates:
            return logits, p
        return logits
    
    def set_warmup_threshold(self, threshold: float):
        """Update warmup threshold."""
        self.warmup_threshold = threshold
    
    def set_use_hard_topk(self, use_hard: bool):
        """Switch between soft and hard gating."""
        self.use_hard_topk = use_hard


class ImageEncoder(nn.Module):
    """Lightweight 3D CNN image encoder."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)


class TokenSetEncoder(nn.Module):
    """
    Encode a set of (feature_id, value) tokens using DeepSets-style pooling.
    """

    def __init__(
        self,
        num_features: int,
        num_rois: int,
        num_types: int,
        id_emb_dim: int = 64,
        roi_emb_dim: int = 16,
        type_emb_dim: int = 32,
        value_emb_dim: int = 64,
        out_dim: int = 256
    ):
        super().__init__()
        self.id_embedding = nn.Embedding(num_features, id_emb_dim)
        self.roi_embedding = nn.Embedding(num_rois, roi_emb_dim)
        self.type_embedding = nn.Embedding(num_types, type_emb_dim)
        self.value_mlp = nn.Sequential(
            nn.Linear(1, value_emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(value_emb_dim, value_emb_dim),
            nn.ReLU(inplace=True)
        )
        self.token_mlp = nn.Sequential(
            nn.Linear(id_emb_dim + roi_emb_dim + type_emb_dim + value_emb_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        feature_ids: torch.Tensor,
        roi_ids: torch.Tensor,
        type_ids: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feature_ids: Long tensor [B, K]
            roi_ids: Long tensor [B, K]
            type_ids: Long tensor [B, K]
            values: Float tensor [B, K]

        Returns:
            Set embedding [B, out_dim]
        """
        id_emb = self.id_embedding(feature_ids)  # [B, K, id_emb_dim]
        roi_emb = self.roi_embedding(roi_ids)  # [B, K, roi_emb_dim]
        type_emb = self.type_embedding(type_ids)  # [B, K, type_emb_dim]
        value_emb = self.value_mlp(values.unsqueeze(-1))  # [B, K, value_emb_dim]
        token = torch.cat([id_emb, roi_emb, type_emb, value_emb], dim=-1)
        token = self.token_mlp(token)  # [B, K, out_dim]
        set_emb = token.mean(dim=1)
        return set_emb


class SubsetScorer(nn.Module):
    """Score subset usefulness given image and token-set embedding."""

    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, z_img: torch.Tensor, z_set: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_img, z_set], dim=-1)
        return self.mlp(x).squeeze(-1)


class TokenSetKLClassifier(nn.Module):
    """Predict KLGrade from token-set embedding only (logistic regression)."""

    def __init__(self, emb_dim: int = 256, num_classes: int = 5):
        super().__init__()
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, z_set: torch.Tensor) -> torch.Tensor:
        return self.classifier(z_set)


class JointScoringModel(nn.Module):
    """
    Joint model for subset scoring and KL classification.
    """

    def __init__(
        self,
        num_features: int,
        num_rois: int,
        num_types: int,
        emb_dim: int = 256,
        num_classes: int = 5,
        id_emb_dim: int = 64,
        value_emb_dim: int = 64,
        roi_emb_dim: int = 16,
        type_emb_dim: int = 32
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(out_dim=emb_dim)
        self.token_set_encoder = TokenSetEncoder(
            num_features=num_features,
            num_rois=num_rois,
            num_types=num_types,
            id_emb_dim=id_emb_dim,
            value_emb_dim=value_emb_dim,
            roi_emb_dim=roi_emb_dim,
            type_emb_dim=type_emb_dim,
            out_dim=emb_dim
        )
        self.scorer = SubsetScorer(emb_dim=emb_dim)
        self.classifier = TokenSetKLClassifier(emb_dim=emb_dim, num_classes=num_classes)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def encode_set(
        self,
        feature_ids: torch.Tensor,
        roi_ids: torch.Tensor,
        type_ids: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        return self.token_set_encoder(feature_ids, roi_ids, type_ids, values)

    def score_subset(
        self,
        image: torch.Tensor,
        feature_ids: torch.Tensor,
        roi_ids: torch.Tensor,
        type_ids: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        z_img = self.encode_image(image)
        z_set = self.encode_set(feature_ids, roi_ids, type_ids, values)
        return self.scorer(z_img, z_set)

    def classify_subset(
        self,
        image: torch.Tensor,
        feature_ids: torch.Tensor,
        roi_ids: torch.Tensor,
        type_ids: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        z_set = self.encode_set(feature_ids, roi_ids, type_ids, values)
        return self.classifier(z_set)