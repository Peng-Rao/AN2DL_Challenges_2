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

    Reference: https://arxiv.org/abs/1907.08610
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
    RAdam optimizer (Liu et al. 2019).

    Rectified Adam - automatically adjusts adaptive learning rate
    based on variance of second moment estimate.

    Reference: https://arxiv.org/abs/1908.03265
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
    Gated Attention Mechanism (Ilse et al. 2018)
    Paper: https://arxiv.org/abs/1802.04712
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
        num_classes: int = 4,  # kept for API compatibility but not used
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
    Transformer-based Multiple Instance Learning (Shao et al. 2021)
    Paper: https://arxiv.org/abs/2106.00908

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
            norm_first=True,  # Pre-norm for better training stability
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


class PathologyModel(L.LightningModule):
    """
    Lightning Module for pathology image classification.

    Supports both single-image and multi-instance learning (patch-based) approaches
    with flexible backbone architectures and aggregation strategies.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
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
    ):
        """
        Args:
            model_name: Name of the timm model ['resnet', 'convnext_tiny', 'efficientnet_b0', etc.].
            num_classes: Number of output classes.
            pretrained: Whether to use ImageNet pretrained weights.
            learning_rate: Base learning rate for optimizer.
            weight_decay: L2 regularization weight.
            use_patches: Whether input is patch-based [B, num_patches, C, H, W].
            patch_aggregation: Aggregation method:
                - 'mean': Simple mean pooling
                - 'max': Max pooling
                - 'attention': Simple attention with softmax
                - 'gated_attention': Gated attention
                - 'clam': CLAM attention
                - 'transmil': Transformer MIL
                - 'multihead': Multi-head self-attention
            optimizer_name: Optimizer to use:
                - 'adamw': AdamW
                - 'lion': Lion
                - 'ranger': RAdam
            dropout_rate: Dropout probability before classifier.
            label_smoothing: Label smoothing factor for cross-entropy.
            class_weights: Optional class weights for imbalanced datasets.
            warmup_epochs: Number of warmup epochs for learning rate.
            freeze_backbone_epochs: Number of epochs to freeze backbone, default 0 (no freezing).
            mixup_alpha: Alpha parameter for Mixup augmentation (0 = no Mixup).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        # Store hyperparameters
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

        # Validate optimizer choice
        valid_optimizers = ["adamw", "lion", "ranger"]
        if self.optimizer_name not in valid_optimizers:
            raise ValueError(
                f"Unknown optimizer: {optimizer_name}. Choose from: {valid_optimizers}"
            )

        # Check if Lion is available when requested
        if self.optimizer_name == "lion" and Lion is None:
            raise ImportError(
                "Lion optimizer requires lion-pytorch package. "
                "Install with: pip install lion-pytorch"
            )

        # Build model architecture
        self._build_model(model_name, pretrained, dropout_rate)

        # Initialize loss and metrics
        self._setup_loss()
        self._setup_metrics()

        # Freeze backbone if requested
        if freeze_backbone_epochs > 0:
            self._freeze_backbone()

    def _build_model(self, model_name: str, pretrained: bool, dropout_rate: float):
        """Build the model architecture."""
        # Create backbone using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=0.0,
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Build aggregation module for patches
        if self.use_patches:
            self.aggregation = self._build_aggregation_module()
        else:
            self.aggregation = None

        # Build classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(
                self.feature_dim
            ),  # LayerNorm often better than BatchNorm for MIL
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),  # GELU often better than ReLU
            nn.LayerNorm(self.feature_dim // 2),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(self.feature_dim // 2, self.num_classes),
        )

    def _build_aggregation_module(self) -> Optional[nn.Module]:
        """Build patch aggregation module based on strategy."""

        if self.patch_aggregation in ["mean", "max"]:
            return None

        elif self.patch_aggregation == "attention":
            return SimpleAttention(self.feature_dim)

        elif self.patch_aggregation == "gated_attention":
            return GatedAttention(self.feature_dim)

        elif self.patch_aggregation == "clam":
            return CLAMAttention(
                self.feature_dim,
                num_classes=self.num_classes,
            )

        elif self.patch_aggregation == "transmil":
            return TransMIL(self.feature_dim)

        elif self.patch_aggregation == "multihead":
            return MultiHeadAttentionMIL(self.feature_dim)

        else:
            raise ValueError(
                f"Unknown aggregation: {self.patch_aggregation}. "
                f"Choose from: mean, max, attention, gated_attention, clam, transmil, multihead"
            )

    def _setup_loss(self):
        """Initialize loss function."""
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    def _setup_metrics(self):
        """Initialize metrics for each stage."""
        metric_kwargs = {"task": "multiclass", "num_classes": self.num_classes}

        # Training metrics
        self.train_acc = Accuracy(**metric_kwargs)
        self.train_f1 = F1Score(**metric_kwargs, average="macro")

        # Validation metrics
        self.val_acc = Accuracy(**metric_kwargs)
        self.val_f1 = F1Score(**metric_kwargs, average="macro")
        self.val_auroc = AUROC(**metric_kwargs)

        # Test metrics
        self.test_acc = Accuracy(**metric_kwargs)
        self.test_f1 = F1Score(**metric_kwargs, average="macro")
        self.test_auroc = AUROC(**metric_kwargs)
        self.test_confmat = ConfusionMatrix(**metric_kwargs)

    def _freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Backbone frozen for {self.freeze_backbone_epochs} epochs")

    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")

    def _apply_mixup(self, x: torch.Tensor, y: torch.Tensor):
        """
        Applies Mixup augmentation to the batch.
        Returns:
            mixed_x: The mixed input tensor
            target_a: Original labels
            target_b: Shuffled labels
            lam: The mixing coefficient (lambda)
        """
        # 1. Sample lambda from Beta distribution
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0

        # 2. Generate permutation indices
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        # 3. Create mixed inputs
        # This works for both [B, C, H, W] and [B, Num_Patches, C, H, W]
        mixed_x = lam * x + (1 - lam) * x[index, :]

        # 4. Get pair of targets
        target_a, target_b = y, y[index]

        return mixed_x, target_a, target_b, lam

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for single images."""
        return self.backbone(x)

    def _forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for patch-based images."""
        batch_size, num_patches, c, h, w = x.shape

        # Reshape to process all patches
        x = x.view(batch_size * num_patches, c, h, w)

        # Extract features
        features = self.backbone(x)  # [B * num_patches, feature_dim]

        # Reshape back
        features = features.view(batch_size, num_patches, -1)

        # Aggregate patches
        features = self._aggregate_patches(features)

        return features

    def _aggregate_patches(self, features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate patch features.

        Args:
            features: [B, num_patches, feature_dim]

        Returns:
            Aggregated features [B, feature_dim]
        """
        if self.patch_aggregation == "mean":
            return features.mean(dim=1)

        elif self.patch_aggregation == "max":
            return features.max(dim=1)[0]

        else:
            return self.aggregation(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
               - If use_patches=False: [B, C, H, W]
               - If use_patches=True: [B, num_patches, C, H, W]

        Returns:
            Logits of shape [B, num_classes]
        """
        if self.use_patches and x.dim() == 5:
            features = self._forward_patches(x)
        else:
            features = self._forward_single(x)

        logits = self.classifier(features)
        return logits

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Training step with Mixup support."""
        x, y = batch

        # Check if we should apply mixup
        # We generally only apply mixup if alpha > 0
        if self.mixup_alpha > 0:
            mixed_x, target_a, target_b, lam = self._apply_mixup(x, y)

            # Forward pass with mixed input
            logits = self(mixed_x)

            # Mixup Loss: weighted sum of loss against both targets
            loss_a = self.criterion(logits, target_a)
            loss_b = self.criterion(logits, target_b)
            loss = lam * loss_a + (1 - lam) * loss_b

            # For logging accuracy, we look at the 'dominant' label (optional)
            # Or we can simply skip accuracy logging for mixup steps as it's noisy.
            # Here, we calculate acc against the target with higher weight.
            if lam >= 0.5:
                preds = torch.argmax(logits, dim=1)
                self.train_acc(preds, target_a)
                self.train_f1(preds, target_a)
            else:
                preds = torch.argmax(logits, dim=1)
                self.train_acc(preds, target_b)
                self.train_f1(preds, target_b)

        else:
            # Standard training (No Mixup)
            logits = self(x)
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            self.train_acc(preds, y)
            self.train_f1(preds, y)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Tuple, batch_idx: int):
        """Validation step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_auroc(probs, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True)

    def test_step(self, batch: Tuple, batch_idx: int):
        """Test step."""
        x, y = batch
        logits = self(x)
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
        Prediction step with Test Time Augmentation (TTA).
        Averages predictions across: Original, Horizontal Flip, Vertical Flip, and Rotations.
        """
        x, sample_ids = batch
        # x shape is either [B, C, H, W] or [B, N, C, H, W]
        # augment the spatial dims: H and W (the last two dimensions)
        spatial_dims = [-2, -1]

        # Define the list of augmentations to apply
        augmentations = [
            lambda t: t,
            lambda t: torch.flip(t, dims=[-1]),  # Horizontal Flip
            lambda t: torch.flip(t, dims=[-2]),  # Vertical Flip
            lambda t: torch.rot90(t, k=1, dims=spatial_dims),  # 90 degree rotation
        ]

        logits_sum = 0

        # Loop through augmentations
        for aug_func in augmentations:
            aug_x = aug_func(x)
            logits = self(aug_x)
            logits_sum += logits

        # Average the logits
        avg_logits = logits_sum / len(augmentations)

        # Compute final probabilities and predictions
        probs = F.softmax(avg_logits, dim=1)
        preds = torch.argmax(avg_logits, dim=1)

        return {
            "sample_ids": sample_ids,
            "predictions": preds,
            "probabilities": probs,
            "logits": avg_logits,
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        # Separate parameters for differential learning rates
        backbone_params = list(self.backbone.parameters())
        classifier_params = list(self.classifier.parameters())
        if self.aggregation is not None:
            classifier_params += list(self.aggregation.parameters())

        # Adjust hyperparameters based on optimizer
        if self.optimizer_name == "lion":
            backbone_lr = self.learning_rate * 0.03  # Even lower for backbone
            classifier_lr = self.learning_rate * 0.3
            wd = self.weight_decay * 10
        else:
            backbone_lr = self.learning_rate * 0.1
            classifier_lr = self.learning_rate
            wd = self.weight_decay

        # Differential learning rates
        param_groups = [
            {
                "params": backbone_params,
                "lr": backbone_lr,
                "name": "backbone",
            },
            {
                "params": classifier_params,
                "lr": classifier_lr,
                "name": "classifier",
            },
        ]

        # Create optimizer based on selection
        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=wd,
            )
        elif self.optimizer_name == "lion":
            optimizer = Lion(
                param_groups,
                weight_decay=wd,
                betas=(0.9, 0.99),
            )
        elif self.optimizer_name == "ranger":
            optimizer = Ranger(
                param_groups,
                weight_decay=wd,
                k=6,  # Lookahead step
                alpha=0.5,  # Lookahead alpha
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Log optimizer info
        # print(f"\n{'=' * 60}")
        # print(f"Optimizer: {self.optimizer_name.upper()}")
        # print(f"Backbone LR: {backbone_lr:.2e}")
        # print(f"Classifier LR: {classifier_lr:.2e}")
        # print(f"Weight Decay: {wd:.2e}")
        # print(f"{'=' * 60}\n")

        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            return 0.5 * (
                1
                + torch.cos(
                    torch.tensor(
                        (epoch - self.warmup_epochs)
                        / (50 - self.warmup_epochs)
                        * 3.14159
                    )
                ).item()
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        if (
            self.freeze_backbone_epochs > 0
            and self.current_epoch == self.freeze_backbone_epochs
        ):
            self._unfreeze_backbone()
            self.trainer.strategy.setup_optimizers(self.trainer)


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


class DualStreamPathologyModel(L.LightningModule):
    """
    Lightning Module for pathology image classification with Dual-Stream support.
    Stream 1: Deep Backbone (RGB)
    Stream 2: Simple CNN (Mask)
    """

    def __init__(
        self,
        model_name: str = "resnet50",
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
        use_dual_stream: bool = False,
    ):
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

        # Validate optimizer
        valid_optimizers = ["adamw", "lion", "ranger"]
        if self.optimizer_name not in valid_optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        if self.optimizer_name == "lion" and Lion is None:
            raise ImportError("Lion optimizer requires lion-pytorch package.")

        self._build_model(model_name, pretrained, dropout_rate)
        self._setup_loss()
        self._setup_metrics()

        if freeze_backbone_epochs > 0:
            self._freeze_backbone()

    def _build_model(self, model_name: str, pretrained: bool, dropout_rate: float):
        """Build the model architecture."""

        # --- STREAM 1: RGB Image (Deep Backbone) ---
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, drop_rate=0.0, in_chans=3
        )
        self.feature_dim = self.backbone.num_features

        # --- STREAM 2: Mask (Simple CNN) ---
        if self.use_dual_stream:
            # Simple CNN outputting 128 features
            mask_out_dim = 128
            self.mask_backbone = SimpleCNN(
                in_chans=1, base_filters=32, output_dim=mask_out_dim
            )

            # Fusion: Concatenation (RGB features + Mask features)
            self.feature_dim += mask_out_dim

        # Build aggregation module for patches
        if self.use_patches:
            self.aggregation = self._build_aggregation_module()
        else:
            self.aggregation = None

        # Build classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.LayerNorm(self.feature_dim // 2),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(self.feature_dim // 2, self.num_classes),
        )

    def _build_aggregation_module(self) -> Optional[nn.Module]:
        if self.patch_aggregation in ["mean", "max"]:
            return None
        elif self.patch_aggregation == "attention":
            return SimpleAttention(self.feature_dim)
        elif self.patch_aggregation == "gated_attention":
            return GatedAttention(self.feature_dim)
        elif self.patch_aggregation == "clam":
            return CLAMAttention(self.feature_dim, num_classes=self.num_classes)
        elif self.patch_aggregation == "transmil":
            return TransMIL(self.feature_dim)
        elif self.patch_aggregation == "multihead":
            return MultiHeadAttentionMIL(self.feature_dim)
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
        # Only freeze the pretrained RGB backbone, usually we keep training the custom SimpleCNN
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"RGB Backbone frozen for {self.freeze_backbone_epochs} epochs")

    def _unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("RGB Backbone unfrozen")

    def _apply_mixup(
        self, x: torch.Tensor, mask: Optional[torch.Tensor], y: torch.Tensor
    ):
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1 - lam) * x[index]

        mixed_mask = None
        if mask is not None:
            mixed_mask = lam * mask + (1 - lam) * mask[index]

        target_a, target_b = y, y[index]
        return mixed_x, mixed_mask, target_a, target_b, lam

    def _forward_single(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        features = self.backbone(x)

        if self.use_dual_stream:
            if mask is None:
                raise ValueError("Dual stream active but mask is None")
            mask_features = self.mask_backbone(mask)
            features = torch.cat([features, mask_features], dim=1)

        return features

    def _forward_patches(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, num_patches, c, h, w = x.shape

        # 1. RGB Processing
        x_flat = x.view(batch_size * num_patches, c, h, w)
        rgb_feat = self.backbone(x_flat)

        # 2. Mask Processing (Simple CNN)
        if self.use_dual_stream:
            if mask is None:
                raise ValueError("Dual stream active but mask is None")

            mask_flat = mask.view(batch_size * num_patches, 1, h, w)
            mask_feat = self.mask_backbone(mask_flat)

            # Late Fusion
            features = torch.cat([rgb_feat, mask_feat], dim=1)
        else:
            features = rgb_feat

        # 3. Reshape and Aggregate
        features = features.view(batch_size, num_patches, -1)
        features = self._aggregate_patches(features)

        return features

    def _aggregate_patches(self, features: torch.Tensor) -> torch.Tensor:
        if self.patch_aggregation == "mean":
            return features.mean(dim=1)
        elif self.patch_aggregation == "max":
            return features.max(dim=1)[0]
        else:
            return self.aggregation(features)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        is_patches = x.dim() == 5

        if self.use_dual_stream and mask is None:
            raise ValueError("Dual stream requires mask input.")

        if is_patches:
            features = self._forward_patches(x, mask)
        else:
            features = self._forward_single(x, mask)

        logits = self.classifier(features)
        return logits

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        if self.use_dual_stream:
            x, mask, y = batch
        else:
            x, y = batch
            mask = None

        if self.mixup_alpha > 0:
            mixed_x, mixed_mask, target_a, target_b, lam = self._apply_mixup(x, mask, y)
            logits = self(mixed_x, mixed_mask)
            loss = lam * self.criterion(logits, target_a) + (1 - lam) * self.criterion(
                logits, target_b
            )

            # Metric logging
            target_metric = target_a if lam >= 0.5 else target_b
            preds = torch.argmax(logits, dim=1)
            self.train_acc(preds, target_metric)
            self.train_f1(preds, target_metric)
        else:
            logits = self(x, mask)
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            self.train_acc(preds, y)
            self.train_f1(preds, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple, batch_idx: int):
        if self.use_dual_stream:
            x, mask, y = batch
        else:
            x, y = batch
            mask = None

        logits = self(x, mask)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_auroc(probs, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True)

    def test_step(self, batch: Tuple, batch_idx: int):
        if self.use_dual_stream:
            x, mask, y = batch
        else:
            x, y = batch
            mask = None

        logits = self(x, mask)
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
        if self.use_dual_stream:
            x, mask, sample_ids = batch
        else:
            x, sample_ids = batch
            mask = None

        spatial_dims = [-2, -1]
        augmentations = [
            lambda t, m: (t, m),
            lambda t, m: (
                torch.flip(t, dims=[-1]),
                torch.flip(m, dims=[-1]) if m is not None else None,
            ),
            lambda t, m: (
                torch.flip(t, dims=[-2]),
                torch.flip(m, dims=[-2]) if m is not None else None,
            ),
            lambda t, m: (
                torch.rot90(t, k=1, dims=spatial_dims),
                torch.rot90(m, k=1, dims=spatial_dims) if m is not None else None,
            ),
        ]

        logits_sum = 0
        for aug_func in augmentations:
            aug_x, aug_mask = aug_func(x, mask)
            logits = self(aug_x, aug_mask)
            logits_sum += logits

        avg_logits = logits_sum / len(augmentations)
        probs = F.softmax(avg_logits, dim=1)
        preds = torch.argmax(avg_logits, dim=1)

        return {"sample_ids": sample_ids, "predictions": preds, "probabilities": probs}

    def configure_optimizers(self) -> Dict[str, Any]:
        # RGB Backbone Params
        backbone_params = list(self.backbone.parameters())

        # Classifier & Aggregation Params
        classifier_params = list(self.classifier.parameters())
        if self.aggregation is not None:
            classifier_params += list(self.aggregation.parameters())

        # Mask CNN Params (Always treat as part of 'classifier' group or its own group with higher LR)
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
                return (epoch + 1) / self.warmup_epochs
            return 0.5 * (
                1
                + torch.cos(
                    torch.tensor(
                        (epoch - self.warmup_epochs)
                        / (50 - self.warmup_epochs)
                        * 3.14159
                    )
                ).item()
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
