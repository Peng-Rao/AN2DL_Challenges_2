#!/usr/bin/env python
"""
Hyperparameter Tuning Script for Pathology Classification Model
Using Optuna for single-machine optimization

Usage:
    python run_hparam_tuning.py

Requirements:
    pip install optuna
"""

import warnings
from typing import Dict

warnings.filterwarnings("ignore")

import albumentations as A
import lightning as L
import optuna
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
)
from notebooks.models import PathologyModel
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Import model and datamodule
from notebooks.PathologyDataModule import PathologyDataModule


# =============================================================================
# Configuration
# =============================================================================
class Config:
    """Data and training configuration."""

    DATA_DIR = "./data"
    TRAIN_DATA_DIR = "./data/train_data"
    TEST_DATA_DIR = "./data/test_data"
    TRAIN_LABELS_PATH = "./data/train_labels.csv"
    TRASH_LIST_PATH = "./data/trash_list.txt"
    CLASSES = ["Luminal A", "Luminal B", "HER2(+)", "Triple negative"]
    NUM_CLASSES = 4


class TuneConfig:
    """Tuning configuration."""

    N_TRIALS = 20  # Number of hyperparameter combinations to try
    MAX_EPOCHS = 50  # Maximum epochs per trial
    PRUNING_WARMUP = 5  # Epochs before pruning can start
    STUDY_NAME = "pathology_hparam_tuning"
    STORAGE = "sqlite:///optuna_study.db"  # Persistent storage


# =============================================================================
# Optuna Pruning Callback
# =============================================================================
class OptunaPruningCallback(Callback):
    """Callback to report metrics to Optuna and handle pruning."""

    def __init__(self, trial: optuna.Trial, monitor: str = "val/acc"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        current_score = metrics.get(self.monitor)

        if current_score is not None:
            score = (
                current_score.item()
                if hasattr(current_score, "item")
                else current_score
            )
            self.trial.report(score, epoch)

            # Prune unpromising trials
            if self.trial.should_prune():
                raise optuna.TrialPruned()


# =============================================================================
# Objective Function
# =============================================================================
def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object.

    Returns:
        Validation accuracy (metric to maximize).
    """
    # Sample hyperparameters
    config = {
        "patch_aggregation": trial.suggest_categorical(
            "patch_aggregation",
            ["clam", "transmil", "gated_attention", "attention", "multihead", "mean"],
        ),
        "optimizer_name": trial.suggest_categorical(
            "optimizer_name", ["adamw", "lion", "ranger"]
        ),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "mixup_alpha": trial.suggest_categorical("mixup_alpha", [0.0, 0.2, 0.4, 0.6]),
        "patch_size": trial.suggest_categorical("patch_size", [128, 192, 224, 256]),
        "num_patches": trial.suggest_categorical("num_patches", [8, 12, 16, 20, 24]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
    }

    L.seed_everything(42)

    img_size = config["patch_size"]

    # Transforms
    train_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1, p=0.8
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # DataModule
    datamodule = PathologyDataModule(
        train_data_dir=Config.TRAIN_DATA_DIR,
        test_data_dir=Config.TEST_DATA_DIR,
        train_labels_path=Config.TRAIN_LABELS_PATH,
        trash_list_path=Config.TRASH_LIST_PATH,
        use_mask=True,
        use_patches=True,
        patch_size=config["patch_size"],
        num_patches=config["num_patches"],
        img_size=img_size,
        batch_size=config["batch_size"],
        min_annotation_pixels=1,
        train_transform=train_transform,
        val_transform=val_transform,
        classes=Config.CLASSES,
    )

    # Model
    model = PathologyModel(
        model_name="convnext_tiny",
        num_classes=Config.NUM_CLASSES,
        use_patches=True,
        patch_aggregation=config["patch_aggregation"],
        optimizer_name=config["optimizer_name"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        dropout_rate=config["dropout_rate"],
        label_smoothing=config["label_smoothing"],
        mixup_alpha=config["mixup_alpha"],
    )

    # Setup datamodule manually to avoid Lightning inheritance check issues
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # Trainer
    trainer = L.Trainer(
        max_epochs=TuneConfig.MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        callbacks=[
            OptunaPruningCallback(trial, monitor="val/acc"),
            EarlyStopping(monitor="val/acc", patience=10, mode="max", verbose=False),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        precision="16-mixed",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=True,
    )

    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return 0.0

    # Return best validation accuracy
    val_acc = trainer.callback_metrics.get("val/acc")
    if val_acc is not None:
        return val_acc.item() if hasattr(val_acc, "item") else val_acc
    return 0.0


# =============================================================================
# Main
# =============================================================================
def run_tuning():
    """Run hyperparameter tuning with Optuna."""

    print("=" * 70)
    print("🔬 Pathology Model Hyperparameter Tuning (Optuna)")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  - Number of trials: {TuneConfig.N_TRIALS}")
    print(f"  - Max epochs per trial: {TuneConfig.MAX_EPOCHS}")
    print(f"  - Pruning warmup epochs: {TuneConfig.PRUNING_WARMUP}")
    print("=" * 70)

    # Create study
    study = optuna.create_study(
        study_name=TuneConfig.STUDY_NAME,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=TuneConfig.PRUNING_WARMUP),
        storage=TuneConfig.STORAGE,
        load_if_exists=True,
    )

    print("\n🚀 Starting hyperparameter search...\n")

    # Run optimization
    study.optimize(
        objective,
        n_trials=TuneConfig.N_TRIALS,
        show_progress_bar=True,
    )

    return study


def analyze_results(study: optuna.Study):
    """Analyze and print tuning results."""

    print("\n" + "=" * 70)
    print("🏆 BEST HYPERPARAMETERS FOUND")
    print("=" * 70)

    print("\n📊 Best Configuration:")
    print("-" * 40)
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\n📈 Best Metrics:")
    print("-" * 40)
    print(f"  Best Validation Accuracy: {study.best_value:.4f}")
    print(f"  Best Trial Number: {study.best_trial.number}")

    # Top 5 trials
    print("\n📋 Top 5 Trials:")
    print("-" * 40)

    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value", ascending=False)

    cols = [
        "number",
        "value",
        "params_patch_aggregation",
        "params_optimizer_name",
        "params_learning_rate",
        "params_batch_size",
    ]
    available_cols = [c for c in cols if c in trials_df.columns]

    print(trials_df.head(5)[available_cols].to_string())

    print("\n" + "=" * 70)

    # Save best config
    save_best_config(study.best_params, study.best_value)

    return study.best_params


def save_best_config(config: Dict, best_value: float):
    """Save best configuration to a Python file."""

    output_file = "best_hparams.py"

    with open(output_file, "w") as f:
        f.write('"""Best hyperparameters found by Optuna."""\n\n')
        f.write("BEST_CONFIG = {\n")
        for key, value in config.items():
            if isinstance(value, str):
                f.write(f'    "{key}": "{value}",\n')
            elif isinstance(value, float):
                f.write(f'    "{key}": {value:.6f},\n')
            else:
                f.write(f'    "{key}": {value},\n')
        f.write("}\n\n")
        f.write(f"BEST_VAL_ACC = {best_value:.4f}\n")

    print(f"\n💾 Best configuration saved to: {output_file}")


if __name__ == "__main__":
    study = run_tuning()
    best_config = analyze_results(study)

    print("\n✅ Hyperparameter tuning complete!")
    print("\nTo train the final model with best hyperparameters:")
    print("  from best_hparams import BEST_CONFIG")
    print(
        "  model = PathologyModel(model_name='convnext_tiny', use_patches=True, **BEST_CONFIG)"
    )
