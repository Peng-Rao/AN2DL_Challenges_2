from typing import Any, Dict, Optional, Tuple

import lightning as L
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lion_pytorch import Lion
from torchmetrics import AUROC, Accuracy, ConfusionMatrix, F1Score


class Lookahead(torch.optim.Optimizer):
    """
    Lookahead optimizer wrapper (Zhang et al. 2019).

    Wraps any optimizer and maintains slow weights that are updated
    by interpolating towards fast weights every k steps.
    """

    def __init__(self, base_optimizer, k: int = 6, alpha: float = 0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = base_optimizer.param_groups
        self.state = {}
        self._step_count = 0

        # Initialize slow weights
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p] = {"slow_weights": p.data.clone()}

    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step."""
        loss = self.base_optimizer.step(closure)
        self._step_count += 1

        if self._step_count % self.k == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.requires_grad and p in self.state:
                        slow = self.state[p]["slow_weights"]
                        # Interpolate: slow = slow + alpha * (fast - slow)
                        slow.add_(p.data - slow, alpha=self.alpha)
                        # Copy slow weights to fast weights
                        p.data.copy_(slow)

        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def defaults(self):
        return self.base_optimizer.defaults

    def state_dict(self):
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "slow_weights": {
                id(p): self.state[p]["slow_weights"]
                for group in self.param_groups
                for p in group["params"]
                if p in self.state
            },
            "step_count": self._step_count,
        }

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self._step_count = state_dict["step_count"]


class RAdam(torch.optim.Optimizer):
    """
    RAdam optimizer
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Compute the maximum length of the approximated SMA
                rho_inf = 2 / (1 - beta2) - 1
                # Compute the length of the approximated SMA
                rho_t = rho_inf - 2 * step * (beta2**step) / bias_correction2

                # Variance rectification
                if rho_t > 5:
                    # Compute variance rectification term
                    rect = (
                        (rho_t - 4)
                        * (rho_t - 2)
                        * rho_inf
                        / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    ) ** 0.5

                    # Compute adaptive learning rate
                    step_size = group["lr"] * rect / bias_correction1

                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    # Use unadapted learning rate
                    step_size = group["lr"] / bias_correction1
                    p.add_(exp_avg, alpha=-step_size)

        return loss


def Ranger(
    params,
    lr: float = 1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    k=6,
    alpha=0.5,
):
    """
    Ranger optimizer: RAdam + Lookahead.

    Combines the variance rectification of RAdam with the
    stabilizing effect of Lookahead for robust training.

    Args:
        params: Model parameters
        lr: Learning rate
        betas: Coefficients for computing running averages
        eps: Term added to denominator for numerical stability
        weight_decay: Weight decay (L2 penalty)
        k: Lookahead step interval
        alpha: Lookahead interpolation factor

    Reference: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
    """
    base = RAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return Lookahead(base, k=k, alpha=alpha)


L.seed_everything(42)


# =============================================================================
# Attention Modules
# =============================================================================
class SimpleAttention(nn.Module):
    """Simple attention mechanism with proper softmax."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        # features: [B, num_patches, feature_dim]
        attention_scores = self.attention(features)  # [B, num_patches, 1]
        attention_weights = F.softmax(attention_scores, dim=1)
        aggregated = torch.sum(attention_weights * features, dim=1)  # [B, feature_dim]
        return aggregated


class GatedAttention(nn.Module):
    """
    Gated Attention Mechanism
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        # features: [B, num_patches, feature_dim]
        A_V = self.attention_V(features)  # [B, num_patches, hidden_dim]
        A_U = self.attention_U(features)  # [B, num_patches, hidden_dim]
        attention_scores = self.attention_w(A_V * A_U)  # [B, num_patches, 1]
        attention_weights = F.softmax(attention_scores, dim=1)
        aggregated = torch.sum(attention_weights * features, dim=1)  # [B, feature_dim]
        return aggregated


class CLAMAttention(nn.Module):
    """
    CLAM-style Attention (simplified, no unused parameters)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.25,
        num_classes: int = 4,
    ):
        super().__init__()

        # Gated attention network
        self.attention_a = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_c = nn.Linear(hidden_dim, 1)

    def forward(self, features, return_attention=False):
        # features: [B, num_patches, feature_dim]

        # Gated attention
        a = self.attention_a(features)
        b = self.attention_b(features)
        attention_scores = self.attention_c(a * b)  # [B, num_patches, 1]
        attention_weights = F.softmax(attention_scores, dim=1)

        # Weighted aggregation
        aggregated = torch.sum(attention_weights * features, dim=1)  # [B, feature_dim]

        if return_attention:
            return aggregated, attention_weights.squeeze(-1)
        return aggregated


class TransMIL(nn.Module):
    """
    Uses transformer encoder with learnable class token for aggregation.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_patches: int = 512,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Input projection (in case feature_dim is not divisible by num_heads)
        self.input_proj = nn.Linear(feature_dim, feature_dim)

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_patches + 1, feature_dim) * 0.02
        )

        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final layer norm
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        # features: [B, num_patches, feature_dim]
        B, N, D = features.shape

        # Project input
        x = self.input_proj(features)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+N, D]

        # Add positional embedding
        x = x + self.pos_embedding[:, : N + 1, :]

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Return class token as bag representation
        return x[:, 0]  # [B, feature_dim]


class MultiHeadAttentionMIL(nn.Module):
    """
    Multi-Head Self-Attention for MIL with learnable class token.
    Simpler than TransMIL but still effective.
    """

    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(feature_dim, feature_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.proj_drop = nn.Dropout(dropout)

        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        # Layer norm
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, features):
        # features: [B, num_patches, feature_dim]
        B, N, D = features.shape

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, features], dim=1)  # [B, 1+N, D]

        # Self-attention with residual
        x_norm = self.norm1(x)
        qkv = (
            self.qkv(x_norm)
            .reshape(B, N + 1, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N + 1, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        x = x + out

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        # Return class token
        return x[:, 0]  # [B, feature_dim]


class SimpleCNN(nn.Module):
    """
    A lightweight, simple CNN for processing binary masks.
    Structure: 4x (Conv -> BN -> ReLU -> MaxPool) -> GlobalAvgPool
    """

    def __init__(
        self, in_chans: int = 1, base_filters: int = 32, output_dim: int = 128
    ):
        super().__init__()
        self.output_dim = output_dim

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_chans, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(
                base_filters, base_filters * 2, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(
                base_filters * 2, base_filters * 4, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(
                base_filters * 4, output_dim, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        return x.flatten(1)


class PathologyModel(L.LightningModule):
    """
    Lightning Module for pathology image classification with Global-Local Architecture.

    Streams:
    1. Local Stream: High-res patches (Bag of Instances) -> MIL Aggregation
    2. Global Stream: Downsampled whole-slide/ROI -> Standard CNN
    3. (Optional) Mask Stream: Binary mask features fused into Local Stream
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        global_model_name: str = "resnet18",  # Lighter backbone for global view
        num_classes: int = 4,
        pretrained: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        use_patches: bool = True,
        patch_aggregation: str = "clam",
        optimizer_name: str = "adamw",
        dropout_rate: float = 0.3,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        warmup_epochs: int = 5,
        freeze_backbone_epochs: int = 0,
        mixup_alpha: float = 0.0,
        use_dual_stream: bool = False,  # Keeps mask functionality
        drop_path_rate: float = 0.2,
    ):
        """
        Args:
            model_name: Backbone for local stream (patches)
            global_model_name: Backbone for global stream (downsampled image)
            num_classes: Number of output classes
            pretrained: Use ImageNet pre-trained weights
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            use_patches: Whether to use patch-based local stream
            patch_aggregation: MIL aggregation method for local stream ("mean", "max", "attention", "gated_attention", "clam", "transmil", "multihead")
            optimizer_name: Optimizer to use ("adamw", "lion", "ranger")
            dropout_rate: Dropout rate for regularization
            label_smoothing: Label smoothing factor for loss
            class_weights: Class weights for handling imbalance
            warmup_epochs: Number of warmup epochs for LR scheduler
            freeze_backbone_epochs: Epochs to freeze backbone at start
            mixup_alpha: Alpha parameter for Mixup augmentation
            use_dual_stream: Whether to use dual stream with mask input
            drop_path_rate: Stochastic depth rate for regularization
        """
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_patches = use_patches
        self.patch_aggregation = patch_aggregation
        self.optimizer_name = optimizer_name.lower()
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        self.warmup_epochs = warmup_epochs
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.mixup_alpha = mixup_alpha
        self.use_dual_stream = use_dual_stream
        self.drop_path_rate = drop_path_rate

        # Validate optimizer
        valid_optimizers = ["adamw", "lion", "ranger"]
        if self.optimizer_name not in valid_optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        if self.optimizer_name == "lion" and Lion is None:
            raise ImportError("Lion optimizer requires lion-pytorch package.")

        self._build_model(
            model_name, global_model_name, pretrained, dropout_rate, drop_path_rate
        )
        self._setup_loss()
        self._setup_metrics()

        if freeze_backbone_epochs > 0:
            self._freeze_backbone()

    def _build_model(
        self,
        model_name: str,
        global_model_name: str,
        pretrained: bool,
        dropout_rate: float,
        drop_path_rate: float,  # NEW ARGUMENT
    ):
        # --- STREAM 1: LOCAL (High-Res Patches) ---
        # FIX 1: Add Stochastic Depth (drop_path_rate) and global pool
        self.local_backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=0.0,  # Keep 0 here, we use bottleneck dropout instead
            # drop_path_rate=drop_path_rate,  # Critical for regularization
            in_chans=3,
            global_pool="",  # We handle pooling/flattening manually
        )
        self.local_feature_dim_raw = self.local_backbone.num_features

        # FIX 2: Local Bottleneck
        # Compress 2048 (ResNet50) -> 256 to force feature selection
        self.local_bottleneck = nn.Sequential(
            nn.Linear(self.local_feature_dim_raw, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.local_feature_dim = 256  # Updated dim

        # Optional Mask Fusion
        if self.use_dual_stream:
            mask_out_dim = 64  # Reduced from 128
            self.mask_backbone = SimpleCNN(
                in_chans=1, base_filters=16, output_dim=mask_out_dim
            )
            self.local_feature_dim += mask_out_dim

        # --- STREAM 2: GLOBAL (Context) ---
        self.global_backbone = timm.create_model(
            global_model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=0.0,
            # drop_path_rate=drop_path_rate,
            in_chans=3,
        )

        # Compress 512 (ResNet18) -> 128
        self.global_bottleneck = nn.Sequential(
            nn.Linear(self.global_backbone.num_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.global_feature_dim = 128

        # --- AGGREGATION ---
        if self.use_patches:
            self.aggregation = self._build_aggregation_module()
        else:
            self.aggregation = None

        # --- CLASSIFIER ---
        # Input is now significantly smaller: 256 (Local) + 128 (Global) = 384
        self.total_feature_dim = self.local_feature_dim + self.global_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.total_feature_dim, self.total_feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),  # Aggressive dropout
            nn.Linear(self.total_feature_dim // 2, self.num_classes),
        )

    def forward(self, x_local, x_global, mask=None):
        # --- 1. Process Global Stream ---
        global_raw = self.global_backbone(x_global)
        global_feat = self.global_bottleneck(global_raw)  # Apply bottleneck

        # --- 2. Process Local Stream ---
        batch_size, num_patches, c, h, w = x_local.shape
        x_local_flat = x_local.view(batch_size * num_patches, c, h, w)

        # Extract features
        local_raw = self.local_backbone(x_local_flat)

        # Apply Global Average Pooling if the backbone output is spatial (e.g. ResNet)
        if len(local_raw.shape) == 4:
            local_raw = F.adaptive_avg_pool2d(local_raw, (1, 1)).flatten(1)

        # Apply Bottleneck immediately after backbone
        local_feat = self.local_bottleneck(local_raw)

        # Optional: Mask Fusion
        if self.use_dual_stream:
            if mask is None:
                raise ValueError("Mask is None")
            mask_flat = mask.view(batch_size * num_patches, 1, h, w)
            mask_feat = self.mask_backbone(mask_flat)
            local_feat = torch.cat([local_feat, mask_feat], dim=1)

        # Reshape back to bag
        local_feat = local_feat.view(batch_size, num_patches, -1)

        # --- 3. Aggregate Local Stream ---
        if self.patch_aggregation == "mean":
            local_agg = local_feat.mean(dim=1)
        elif self.patch_aggregation == "max":
            local_agg = local_feat.max(dim=1)[0]
        else:
            local_agg = self.aggregation(local_feat)

        # --- 4. Fusion & Classify ---
        fused_feat = torch.cat([local_agg, global_feat], dim=1)
        logits = self.classifier(fused_feat)
        return logits

    def _build_aggregation_module(self) -> Optional[nn.Module]:
        # Uses local_feature_dim because aggregation happens BEFORE fusion with global
        if self.patch_aggregation in ["mean", "max"]:
            return None
        elif self.patch_aggregation == "attention":
            return SimpleAttention(self.local_feature_dim)
        elif self.patch_aggregation == "gated_attention":
            return GatedAttention(self.local_feature_dim)
        elif self.patch_aggregation == "clam":
            return CLAMAttention(self.local_feature_dim, num_classes=self.num_classes)
        elif self.patch_aggregation == "transmil":
            return TransMIL(self.local_feature_dim)
        elif self.patch_aggregation == "multihead":
            return MultiHeadAttentionMIL(self.local_feature_dim)
        else:
            raise ValueError(f"Unknown aggregation: {self.patch_aggregation}")

    def _setup_loss(self):
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights, label_smoothing=self.label_smoothing
        )

    def _setup_metrics(self):
        metric_kwargs = {"task": "multiclass", "num_classes": self.num_classes}
        self.train_acc = Accuracy(**metric_kwargs)
        self.train_f1 = F1Score(**metric_kwargs, average="macro")
        self.val_acc = Accuracy(**metric_kwargs)
        self.val_f1 = F1Score(**metric_kwargs, average="macro")
        self.val_auroc = AUROC(**metric_kwargs)
        self.test_acc = Accuracy(**metric_kwargs)
        self.test_f1 = F1Score(**metric_kwargs, average="macro")
        self.test_auroc = AUROC(**metric_kwargs)
        self.test_confmat = ConfusionMatrix(**metric_kwargs)

    def _freeze_backbone(self):
        for param in self.local_backbone.parameters():
            param.requires_grad = False
        for param in self.global_backbone.parameters():
            param.requires_grad = False
        print(f"Backbones frozen for {self.freeze_backbone_epochs} epochs")

    def _unfreeze_backbone(self):
        for param in self.local_backbone.parameters():
            param.requires_grad = True
        for param in self.global_backbone.parameters():
            param.requires_grad = True
        print("Backbones unfrozen")

    def _apply_mixup(self, x_local, x_global, mask, y):
        """Helper to apply mixup to dual-stream inputs."""
        batch_size = x_local.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        index = torch.randperm(batch_size).to(x_local.device)

        # Mixup Local Patches
        mixed_x_local = lam * x_local + (1 - lam) * x_local[index, :]

        # Mixup Global Image
        mixed_x_global = lam * x_global + (1 - lam) * x_global[index, :]

        # Mixup Mask (Optional)
        mixed_mask = None
        if mask is not None:
            mixed_mask = lam * mask + (1 - lam) * mask[index, :]

        # Targets
        y_a, y_b = y, y[index]
        return mixed_x_local, mixed_x_global, mixed_mask, y_a, y_b, lam

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        if self.use_dual_stream:
            x_local, x_global, mask, y = batch
        else:
            x_local, x_global, y = batch
            mask = None

        if self.mixup_alpha > 0 and self.current_epoch < self.trainer.max_epochs - 5:
            x_local, x_global, mask, y_a, y_b, lam = self._apply_mixup(
                x_local, x_global, mask, y
            )
            logits = self(x_local, x_global, mask)
            loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(
                logits, y_b
            )
        else:
            logits = self(x_local, x_global, mask)
            loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        if self.mixup_alpha == 0:
            self.train_acc(preds, y)
            self.log(
                "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
            )

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple, batch_idx: int):
        if self.use_dual_stream:
            x_local, x_global, mask, y = batch
        else:
            x_local, x_global, y = batch
            mask = None

        logits = self(x_local, x_global, mask)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_auroc(probs, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple, batch_idx: int):
        if self.use_dual_stream:
            x_local, x_global, mask, y = batch
        else:
            x_local, x_global, y = batch
            mask = None

        logits = self(x_local, x_global, mask)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_auroc(probs, y)
        self.test_confmat(preds, y)

        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc)
        self.log("test/f1", self.test_f1)
        self.log("test/auroc", self.test_auroc)

    def predict_step(self, batch: Tuple, batch_idx: int) -> Dict[str, Any]:
        """
        Prediction step with Test-Time Augmentation (TTA).
        Applies TTA (Flips/Rotations) to Local Patches, Global Image, and Mask.
        """
        # Unpack batch based on configuration
        if self.use_dual_stream:
            # batch structure: (patches, global_img, mask, sample_ids)
            x_local, x_global, mask, sample_ids = batch
        else:
            # batch structure: (patches, global_img, sample_ids)
            x_local, x_global, sample_ids = batch
            mask = None

        spatial_dims = [-2, -1]  # H, W dimensions for both patches and global image

        # TTA Augmentations: Tuple of (Local, Global, Mask) -> (Local, Global, Mask)
        augmentations = [
            # 1. Identity (No transform)
            lambda l, g, m: (l, g, m),
            # 2. Horizontal Flip
            lambda l, g, m: (
                torch.flip(l, dims=[-1]),
                torch.flip(g, dims=[-1]),
                torch.flip(m, dims=[-1]) if m is not None else None,
            ),
            # 3. Vertical Flip
            lambda l, g, m: (
                torch.flip(l, dims=[-2]),
                torch.flip(g, dims=[-2]),
                torch.flip(m, dims=[-2]) if m is not None else None,
            ),
            # 4. Rotate 90 degrees
            lambda l, g, m: (
                torch.rot90(l, k=1, dims=spatial_dims),
                torch.rot90(g, k=1, dims=spatial_dims),
                torch.rot90(m, k=1, dims=spatial_dims) if m is not None else None,
            ),
        ]

        logits_sum = 0

        # Apply every augmentation and aggregate logits
        for aug_func in augmentations:
            aug_local, aug_global, aug_mask = aug_func(x_local, x_global, mask)

            # Forward pass with augmented views
            logits = self(aug_local, aug_global, aug_mask)
            logits_sum += logits

        # Average logits across augmentations
        avg_logits = logits_sum / len(augmentations)
        probs = F.softmax(avg_logits, dim=1)
        preds = torch.argmax(avg_logits, dim=1)

        return {"sample_ids": sample_ids, "predictions": preds, "probabilities": probs}

    def configure_optimizers(self) -> Dict[str, Any]:
        max_epochs = self.trainer.max_epochs
        # Separate parameter groups for possibly different LRs
        backbone_params = list(self.local_backbone.parameters()) + list(
            self.global_backbone.parameters()
        )

        classifier_params = list(self.classifier.parameters())
        if self.aggregation is not None:
            classifier_params += list(self.aggregation.parameters())
        if self.use_dual_stream:
            classifier_params += list(self.mask_backbone.parameters())

        if self.optimizer_name == "lion":
            backbone_lr = self.learning_rate * 0.03
            classifier_lr = self.learning_rate * 0.3
            wd = self.weight_decay * 10
        else:
            backbone_lr = self.learning_rate * 0.1
            classifier_lr = self.learning_rate
            wd = self.weight_decay

        param_groups = [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": classifier_params, "lr": classifier_lr, "name": "classifier"},
        ]

        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
        elif self.optimizer_name == "lion":
            optimizer = Lion(param_groups, weight_decay=wd, betas=(0.9, 0.99))
        elif self.optimizer_name == "ranger":
            optimizer = Ranger(param_groups, weight_decay=wd, k=6, alpha=0.5)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch + 1) / float(max_epochs)

            progress = float(epoch - self.warmup_epochs) / float(
                max(1, max_epochs - self.warmup_epochs)
            )
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
