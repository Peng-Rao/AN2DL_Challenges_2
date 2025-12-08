"""
Test script for PathologyDataModule with visualization of transformed images.

This script:
1. Initializes the PathologyDataModule with real PathologyDataset
2. Sets up train, val, and test datasets
3. Visualizes sample images from each split after transformations

Usage:
    python test_datamodule.py --train_dir ./data/train --test_dir ./data/test --labels ./data/train.csv
    python test_datamodule.py --train_dir ./data/train --test_dir ./data/test --labels ./data/train.csv --use_patches
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import lightning.pytorch as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# =============================================================================
# Configuration
# =============================================================================
class Config:
    # Data paths
    DATA_DIR = "./data"
    TRAIN_DATA_DIR = "./data/train_data"
    TEST_DATA_DIR = "./data/test_data"
    TRAIN_LABELS_PATH = "./data/train_labels.csv"
    OUTPUT_PATH = "./predictions.csv"
    # Class labels
    CLASSES = ["Luminal A", "Luminal B", "HER2(+)", "Triple negative"]
    NUM_CLASSES = 4

    # Image settings
    IMG_SIZE = 512  # Larger size for histopathology
    USE_MASK = True

    # Tissue detection settings
    TISSUE_THRESHOLD = 0.8  # Threshold for tissue detection (lower = more sensitive)
    MIN_TISSUE_AREA = 0.05  # Minimum tissue area ratio
    PADDING = 50  # Padding around tissue bounding box

    # Patch-based settings (for very large images)
    USE_PATCHES = True
    PATCH_SIZE = 224
    NUM_PATCHES = 8  # Number of patches to sample per image

    # Stain normalization
    USE_STAIN_NORMALIZATION = False

    # Training settings
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    MAX_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Validation split
    VAL_SPLIT = 0.2
    RANDOM_SEED = 42


# =============================================================================
# Tissue Extractor (simplified version)
# =============================================================================
class TissueExtractor:
    """Extract tissue patches from histopathology images."""

    def __init__(self, patch_size: int = 224, min_tissue_ratio: float = 0.05):
        self.patch_size = patch_size
        self.min_tissue_ratio = min_tissue_ratio

    def get_valid_patches(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        num_patches: int = 8,
        strategy: str = "random",
        stride: Optional[int] = None,
        shuffle: bool = False,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Extract valid patches from image based on mask."""
        h, w = img.shape[:2]
        patches = []
        coords = []

        if strategy == "grid":
            stride = stride or self.patch_size // 2
            for y in range(0, h - self.patch_size + 1, stride):
                for x in range(0, w - self.patch_size + 1, stride):
                    patch_mask = mask[y : y + self.patch_size, x : x + self.patch_size]
                    tissue_ratio = np.mean(patch_mask > 0)

                    if tissue_ratio >= self.min_tissue_ratio:
                        patch = img[y : y + self.patch_size, x : x + self.patch_size]
                        patches.append(patch)
                        coords.append((y, x))
        else:  # random
            attempts = 0
            max_attempts = num_patches * 20

            while len(patches) < num_patches and attempts < max_attempts:
                if h > self.patch_size and w > self.patch_size:
                    y = np.random.randint(0, h - self.patch_size)
                    x = np.random.randint(0, w - self.patch_size)
                else:
                    y, x = 0, 0

                y_end = min(y + self.patch_size, h)
                x_end = min(x + self.patch_size, w)

                patch_mask = mask[y:y_end, x:x_end]
                tissue_ratio = np.mean(patch_mask > 0)

                if tissue_ratio >= self.min_tissue_ratio:
                    patch = img[y:y_end, x:x_end]
                    if (
                        patch.shape[0] == self.patch_size
                        and patch.shape[1] == self.patch_size
                    ):
                        patches.append(patch)
                        coords.append((y, x))

                attempts += 1

        if shuffle and patches:
            combined = list(zip(patches, coords))
            np.random.shuffle(combined)
            patches, coords = zip(*combined)
            patches, coords = list(patches), list(coords)

        return patches[:num_patches], coords[:num_patches]


# =============================================================================
# PathologyDataset (from your provided code)
# =============================================================================
class PathologyDataset(Dataset):
    """Dataset optimized for histopathology images."""

    def __init__(
        self,
        data_dir: str,
        labels_df: Optional[pd.DataFrame] = None,
        transform: Optional[transforms.Compose] = None,
        use_mask: bool = True,
        use_patches: bool = False,
        patch_size: int = 224,
        num_patches: int = 8,
        patch_strategy: str = "random",
        min_tissue_ratio: float = 0.05,
        use_stain_norm: bool = False,
        is_test: bool = False,
        label_encoder: Optional[LabelEncoder] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_mask = use_mask
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_strategy = patch_strategy
        self.use_stain_norm = use_stain_norm
        self.is_test = is_test
        self.label_encoder = label_encoder

        # Initialize helpers
        self.tissue_extractor = TissueExtractor(
            patch_size=patch_size,
            min_tissue_ratio=min_tissue_ratio,
        )
        self.stain_normalizer = None

        if is_test:
            self.samples = self._get_test_samples()
            self.labels = None
            self.encoded_labels = None
        else:
            if labels_df is None:
                raise ValueError("labels_df must be provided for training/validation.")

            self.samples = [
                self._clean_sample_idx(str(idx))
                for idx in labels_df["sample_index"].tolist()
            ]
            self.labels = labels_df["label"].tolist()

            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(Config.CLASSES)
            self.encoded_labels = self.label_encoder.transform(self.labels)

    def _clean_sample_idx(self, sample_idx: str) -> str:
        """Clean sample index by removing prefix and suffix."""
        sample_idx = str(sample_idx)
        if sample_idx.startswith("img_"):
            sample_idx = sample_idx[4:]
        if sample_idx.endswith(".png"):
            sample_idx = sample_idx[:-4]
        return sample_idx

    def _get_test_samples(self) -> List[str]:
        """Get list of sample indices from test directory."""
        samples = []
        for f in sorted(self.data_dir.glob("img_*.png")):
            sample_idx = self._clean_sample_idx(f.stem)
            samples.append(sample_idx)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_and_mask(
        self, sample_idx: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load image and optionally its mask."""
        img_path = self.data_dir / f"img_{sample_idx}.png"
        img = np.array(Image.open(img_path).convert("RGB"))

        mask = None
        if self.use_mask:
            mask_path = self.data_dir / f"mask_{sample_idx}.png"
            if mask_path.exists():
                mask = np.array(Image.open(mask_path).convert("L"))

        return img, mask

    def _crop_to_tissue_bbox(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Crop image to bounding box of tissue region."""
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)

        if not rows.any() or not cols.any():
            return img

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        padding = 10
        y_min = max(0, y_min - padding)
        y_max = min(img.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(img.shape[1], x_max + padding)

        return img[y_min:y_max, x_min:x_max]

    def _apply_stain_normalization(self, img: np.ndarray) -> np.ndarray:
        """Apply stain normalization to image."""
        if self.use_stain_norm and self.stain_normalizer is not None:
            try:
                return self.stain_normalizer.normalize(img)
            except Exception:
                pass
        return img

    def _load_and_preprocess(self, sample_idx: str) -> np.ndarray:
        """Load and preprocess full image with optional tissue cropping."""
        img, mask = self._load_image_and_mask(sample_idx)

        if mask is not None:
            img = self._crop_to_tissue_bbox(img, mask)

        if self.use_stain_norm:
            img = self._apply_stain_normalization(img)

        return img

    def _load_patches(self, sample_idx: str) -> List[np.ndarray]:
        """Load image as patches using TissueExtractor."""
        img, mask = self._load_image_and_mask(sample_idx)

        if mask is None:
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

        patches, _ = self.tissue_extractor.get_valid_patches(
            img=img,
            mask=mask,
            num_patches=self.num_patches,
            strategy=self.patch_strategy,
            stride=self.patch_size // 2 if self.patch_strategy == "grid" else None,
            shuffle=False,
        )

        if len(patches) == 0:
            h, w = img.shape[:2]
            cy, cx = h // 2, w // 2
            half = self.patch_size // 2
            y1 = max(0, cy - half)
            x1 = max(0, cx - half)
            y2 = min(h, y1 + self.patch_size)
            x2 = min(w, x1 + self.patch_size)
            fallback_patch = img[y1:y2, x1:x2]
            fallback_patch = cv2.resize(
                fallback_patch, (self.patch_size, self.patch_size)
            )
            patches = [fallback_patch] * self.num_patches

        elif len(patches) < self.num_patches:
            while len(patches) < self.num_patches:
                patches.append(patches[len(patches) % len(patches)])

        normalized_patches = []
        for patch in patches:
            patch = self._apply_stain_normalization(patch)
            normalized_patches.append(patch)

        return normalized_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        sample_idx = self.samples[idx]

        if self.use_patches:
            patches = self._load_patches(sample_idx)

            transformed_patches = []
            for patch in patches:
                patch_pil = Image.fromarray(patch)
                if self.transform:
                    patch_tensor = self.transform(patch_pil)
                else:
                    patch_tensor = transforms.ToTensor()(patch_pil)
                transformed_patches.append(patch_tensor)

            img_tensor = torch.stack(transformed_patches)
        else:
            img = self._load_and_preprocess(sample_idx)
            img_pil = Image.fromarray(img)

            if self.transform:
                img_tensor = self.transform(img_pil)
            else:
                img_tensor = transforms.ToTensor()(img_pil)

        if self.is_test:
            return img_tensor, sample_idx
        else:
            label = self.encoded_labels[idx]
            return img_tensor, torch.tensor(label, dtype=torch.long)


# =============================================================================
# PathologyDataModule
# =============================================================================
class PathologyDataModule(L.LightningDataModule):
    """Lightning DataModule for histopathology image classification.

    Args:
        train_data_dir: Directory containing training images and masks.
        test_data_dir: Directory containing test images and masks.
        train_labels_path: Path to CSV file with training labels.
        batch_size: Batch size for dataloaders.
        num_workers: Number of workers for dataloaders.
        img_size: Target image size (used when not using patches).
        use_mask: Whether to use masks for tissue extraction.
        use_patches: Whether to use patch-based loading.
        patch_size: Size of patches to extract.
        num_patches: Number of patches per image.
        min_tissue_ratio: Minimum tissue ratio for valid patches.
        use_stain_norm: Whether to apply stain normalization.
        val_split: Fraction of training data to use for validation.
        random_seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        train_data_dir: str = Config.TRAIN_DATA_DIR,
        test_data_dir: str = Config.TEST_DATA_DIR,
        train_labels_path: str = Config.TRAIN_LABELS_PATH,
        trash_list_path: str = "data/trash_list.txt",
        batch_size: int = Config.BATCH_SIZE,
        num_workers: int = Config.NUM_WORKERS,
        img_size: int = Config.IMG_SIZE,
        use_mask: bool = Config.USE_MASK,
        use_patches: bool = Config.USE_PATCHES,
        patch_size: int = Config.PATCH_SIZE,
        num_patches: int = Config.NUM_PATCHES,
        min_tissue_ratio: float = Config.MIN_TISSUE_AREA,
        use_stain_norm: bool = Config.USE_STAIN_NORMALIZATION,
        val_split: float = Config.VAL_SPLIT,
        random_seed: int = Config.RANDOM_SEED,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.train_labels_path = train_labels_path
        self.trash_list_path = trash_list_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.use_mask = use_mask
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.min_tissue_ratio = min_tissue_ratio
        self.use_stain_norm = use_stain_norm
        self.val_split = val_split
        self.random_seed = random_seed

        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(Config.CLASSES)

        # Will be set in setup()
        self.train_df = None
        self.val_df = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_train_transforms(self) -> transforms.Compose:
        """Get augmentation transforms for training."""
        target_size = self.patch_size if self.use_patches else self.img_size
        return transforms.Compose(
            [
                transforms.Resize((target_size, target_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _get_val_transforms(self) -> transforms.Compose:
        """Get transforms for validation/test (no augmentation)."""
        target_size = self.patch_size if self.use_patches else self.img_size
        return transforms.Compose(
            [
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        if stage == "fit" or stage is None:
            # Load and split training data
            full_df = pd.read_csv(self.train_labels_path)

            # --- TRASH FILTERING START (DEBUG VERSION) ---
            trash_path = Path(self.trash_list_path)
            if trash_path.exists():
                print(f"Loading trash list from {trash_path}...")
                with open(trash_path, "r") as f:
                    trash_files = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]

                print(f"Total lines in trash_list.txt: {len(trash_files)}")

                # Normalize trash filenames to IDs
                trash_ids = set()
                for t_file in trash_files:
                    clean_id = t_file.replace("img_", "").replace(".png", "")
                    trash_ids.add(clean_id)

                print(
                    f"Unique IDs in trash list (after deduplication): {len(trash_ids)}"
                )

                # Helper to clean DataFrame IDs
                def clean_df_id(x):
                    return str(x).replace("img_", "").replace(".png", "")

                # Get all IDs currently in the CSV
                csv_ids = set(full_df["sample_index"].apply(clean_df_id))

                # Calculate intersection and difference
                ids_to_remove = trash_ids.intersection(csv_ids)
                ids_not_found = trash_ids - csv_ids

                print(f"IDs from trash list FOUND in CSV: {len(ids_to_remove)}")
                print(f"IDs from trash list NOT FOUND in CSV: {len(ids_not_found)}")

                if len(ids_not_found) > 0:
                    print(f"Example missing IDs: {list(ids_not_found)[:5]}")

                # Apply the filter
                initial_count = len(full_df)
                mask = full_df["sample_index"].apply(clean_df_id).isin(trash_ids)
                full_df = full_df[~mask].reset_index(drop=True)

                dropped_count = initial_count - len(full_df)
                print(f"Final check: Removed {dropped_count} rows from dataframe.")
                print(f"Remaining samples: {len(full_df)}")
            else:
                print("No trash_list.txt found, skipping filtering.")
            # --- TRASH FILTERING END ---

            self.train_df, self.val_df = train_test_split(
                full_df,
                test_size=self.val_split,
                stratify=full_df["label"],
                random_state=self.random_seed,
            )

            # Training dataset: random strategy with half overlap
            self.train_dataset = PathologyDataset(
                data_dir=self.train_data_dir,
                labels_df=self.train_df,
                transform=self._get_train_transforms(),
                use_mask=self.use_mask,
                use_patches=self.use_patches,
                patch_size=self.patch_size,
                num_patches=self.num_patches,
                patch_strategy="random",  # Random for training
                min_tissue_ratio=self.min_tissue_ratio,
                use_stain_norm=self.use_stain_norm,
                is_test=False,
                label_encoder=self.label_encoder,
            )

            # Validation dataset: grid strategy with no overlap
            self.val_dataset = PathologyDataset(
                data_dir=self.train_data_dir,
                labels_df=self.val_df,
                transform=self._get_val_transforms(),
                use_mask=self.use_mask,
                use_patches=self.use_patches,
                patch_size=self.patch_size,
                num_patches=self.num_patches,
                patch_strategy="grid",  # Grid for validation
                min_tissue_ratio=self.min_tissue_ratio,
                use_stain_norm=self.use_stain_norm,
                is_test=False,
                label_encoder=self.label_encoder,
            )

        if stage == "test" or stage == "predict" or stage is None:
            # Test dataset: grid strategy with no overlap
            self.test_dataset = PathologyDataset(
                data_dir=self.test_data_dir,
                labels_df=None,
                transform=self._get_val_transforms(),
                use_mask=self.use_mask,
                use_patches=self.use_patches,
                patch_size=self.patch_size,
                num_patches=self.num_patches,
                patch_strategy="grid",
                min_tissue_ratio=self.min_tissue_ratio,
                use_stain_norm=self.use_stain_norm,
                is_test=True,
                label_encoder=self.label_encoder,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


# =============================================================================
# Visualization Functions
# =============================================================================
def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize image tensor for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    tensor = tensor.cpu().clone()
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)

    img = tensor.permute(1, 2, 0).numpy()
    return img


def plot_dataset_samples(
    dataset,
    dataset_name: str,
    num_samples: int = 4,
    use_patches: bool = False,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    is_test: bool = False,
):
    """Plot sample images from a dataset."""
    if class_names is None:
        class_names = Config.CLASSES

    indices = np.random.choice(
        len(dataset), min(num_samples, len(dataset)), replace=False
    )

    if use_patches:
        fig, axes = plt.subplots(num_samples, 8, figsize=(20, 3 * num_samples))
        fig.suptitle(
            f"{dataset_name} Dataset - Patch Samples (After Transforms)",
            fontsize=16,
            y=1.02,
        )

        for i, idx in enumerate(indices):
            result = dataset[idx]
            img_tensor = result[0]

            if is_test:
                sample_id = result[1]
                title = f"ID: {sample_id}"
            else:
                label = result[1]
                label_name = class_names[label.item()]
                title = f"Label: {label_name}"

            for j in range(min(8, img_tensor.shape[0])):
                ax = axes[i, j] if num_samples > 1 else axes[j]
                patch = denormalize(img_tensor[j])
                ax.imshow(patch)
                ax.axis("off")
                if j == 0:
                    ax.set_title(title, fontsize=10)
    else:
        cols = min(4, num_samples)
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        fig.suptitle(
            f"{dataset_name} Dataset - Image Samples (After Transforms)",
            fontsize=16,
            y=1.02,
        )

        if num_samples == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            result = dataset[idx]
            img_tensor = result[0]

            if is_test:
                sample_id = result[1]
                title = f"ID: {sample_id}"
            else:
                label = result[1]
                label_name = class_names[label.item()]
                title = f"Label: {label_name}"

            img = denormalize(img_tensor)
            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=10)
            axes[i].axis("off")

        for i in range(len(indices), len(axes)):
            axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    # Quiet, non-blocking draw
    plt.show(block=False)
    plt.pause(0.001)
    plt.close()


def plot_class_distribution(
    dataset, dataset_name: str, save_path: Optional[str] = None
):
    """Plot class distribution in the dataset."""
    if dataset.encoded_labels is None:
        print(f"No labels available for {dataset_name}")
        return

    labels = dataset.labels
    unique, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
    bars = ax.bar(unique, counts, color=colors, edgecolor="black")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{dataset_name} - Class Distribution", fontsize=14)
    plt.xticks(rotation=45, ha="right")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show(block=False)
    plt.pause(0.001)
    plt.close()


def plot_augmentation_comparison(
    datamodule: PathologyDataModule,
    num_samples: int = 4,
    save_path: Optional[str] = None,
):
    """Compare images with and without augmentation."""
    val_transform = datamodule._get_val_transforms()
    train_transform = datamodule._get_train_transforms()

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    fig.suptitle("Augmentation Comparison: Original vs Augmented", fontsize=14, y=1.02)

    indices = np.random.choice(
        len(datamodule.train_dataset),
        min(num_samples, len(datamodule.train_dataset)),
        replace=False,
    )

    for i, idx in enumerate(indices):
        sample_idx = datamodule.train_dataset.samples[idx]
        label = datamodule.train_dataset.labels[idx]

        # Load raw image
        img = datamodule.train_dataset._load_and_preprocess(sample_idx)
        img_pil = Image.fromarray(img)

        # Apply transforms
        img_no_aug = val_transform(img_pil)
        img_with_aug = train_transform(img_pil)

        ax_orig = axes[i, 0] if num_samples > 1 else axes[0]
        ax_aug = axes[i, 1] if num_samples > 1 else axes[1]

        ax_orig.imshow(denormalize(img_no_aug))
        ax_orig.set_title(f"Original - {label}", fontsize=10)
        ax_orig.axis("off")

        ax_aug.imshow(denormalize(img_with_aug))
        ax_aug.set_title(f"Augmented - {label}", fontsize=10)
        ax_aug.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show(block=False)
    plt.pause(0.001)
    plt.close()


def plot_batch_grid(
    dataloader: DataLoader,
    dataset_name: str,
    save_path: Optional[str] = None,
    is_test: bool = False,
):
    """Plot a grid of images from a single batch."""
    batch = next(iter(dataloader))

    if is_test:
        images, sample_ids = batch
    else:
        images, labels = batch

    # Handle patch-based batches
    if images.dim() == 5:  # [B, num_patches, C, H, W]
        images = images[:, 0, :, :, :]

    batch_size = images.shape[0]
    cols = min(4, batch_size)
    rows = (batch_size + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(f"{dataset_name} - Batch Visualization", fontsize=14, y=1.02)

    axes = np.array(axes).flatten() if batch_size > 1 else [axes]

    for i in range(batch_size):
        img = denormalize(images[i])
        axes[i].imshow(img)

        if is_test:
            axes[i].set_title(f"ID: {sample_ids[i]}", fontsize=10)
        else:
            label_name = Config.CLASSES[labels[i].item()]
            axes[i].set_title(f"{label_name}", fontsize=10)
        axes[i].axis("off")

    for i in range(batch_size, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show(block=False)
    plt.pause(0.001)
    plt.close()


# =============================================================================
# Test Functions
# =============================================================================
def test_datamodule(
    train_data_dir: str,
    test_data_dir: str,
    labels_path: str,
    use_patches: bool = True,
    output_dir: str = ".",
):
    """Test the PathologyDataModule and visualize samples."""
    print("=" * 60)
    print("Testing PathologyDataModule")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize DataModule
    datamodule = PathologyDataModule(
        train_data_dir=train_data_dir,
        test_data_dir=test_data_dir,
        train_labels_path=labels_path,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        img_size=Config.IMG_SIZE,
        use_patches=use_patches,
        patch_size=Config.PATCH_SIZE,
        num_patches=Config.NUM_PATCHES,
    )

    # Setup datasets
    datamodule.setup()

    print(f"\n{'=' * 60}")
    print("Dataset Summary:")
    print(f"{'=' * 60}")
    print(f"  Train samples: {len(datamodule.train_dataset)}")
    print(f"  Val samples:   {len(datamodule.val_dataset)}")
    print(f"  Test samples:  {len(datamodule.test_dataset)}")
    print(f"  Use patches:   {use_patches}")

    # Test dataloaders
    print(f"\n{'=' * 60}")
    print("Testing Dataloaders:")
    print(f"{'=' * 60}")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # Get batch shapes
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    print(
        f"  Train batch - Images: {train_batch[0].shape}, Labels: {train_batch[1].shape}"
    )
    print(train_batch[1])
    print(f"  Val batch   - Images: {val_batch[0].shape}, Labels: {val_batch[1].shape}")
    print(f"  Test batch  - Images: {test_batch[0].shape}, IDs: {len(test_batch[1])}")

    # Visualizations
    print(f"\n{'=' * 60}")
    print("Generating Visualizations:")
    print(f"{'=' * 60}")

    # 1. Plot training samples
    print("\n1. Training samples...")
    plot_dataset_samples(
        datamodule.train_dataset,
        "Training",
        num_samples=4,
        use_patches=use_patches,
        save_path=str(output_dir / "train_samples.png"),
        is_test=False,
    )

    # 2. Plot validation samples
    print("\n2. Validation samples...")
    plot_dataset_samples(
        datamodule.val_dataset,
        "Validation",
        num_samples=4,
        use_patches=use_patches,
        save_path=str(output_dir / "val_samples.png"),
        is_test=False,
    )

    # 3. Plot test samples
    print("\n3. Test samples...")
    plot_dataset_samples(
        datamodule.test_dataset,
        "Test",
        num_samples=4,
        use_patches=use_patches,
        save_path=str(output_dir / "test_samples.png"),
        is_test=True,
    )

    # 4. Plot class distribution
    print("\n4. Class distribution...")
    plot_class_distribution(
        datamodule.train_dataset,
        "Training",
        save_path=str(output_dir / "class_distribution.png"),
    )

    # 5. Plot augmentation comparison
    if not use_patches:
        print("\n5. Augmentation comparison...")
        plot_augmentation_comparison(
            datamodule,
            num_samples=4,
            save_path=str(output_dir / "augmentation_comparison.png"),
        )

    # 6. Plot batch grids
    print("\n6. Batch visualizations...")
    plot_batch_grid(
        train_loader,
        "Training Batch",
        save_path=str(output_dir / "train_batch.png"),
        is_test=False,
    )
    plot_batch_grid(
        val_loader,
        "Validation Batch",
        save_path=str(output_dir / "val_batch.png"),
        is_test=False,
    )
    plot_batch_grid(
        test_loader,
        "Test Batch",
        save_path=str(output_dir / "test_batch.png"),
        is_test=True,
    )

    print(f"\n{'=' * 60}")
    print("Test completed successfully!")
    print(f"{'=' * 60}")

    return datamodule


def main():
    """Main function to run all tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test PathologyDataModule")
    parser.add_argument(
        "--train_dir",
        type=str,
        default="./data/train_data",
        help="Path to training data directory",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="./data/test_data",
        help="Path to test data directory",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="./data/train_labels.csv",
        help="Path to labels CSV file",
    )
    parser.add_argument(
        "--use_patches",
        default=True,
        action="store_true",
        help="Use patch-based loading",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Directory to save output images",
    )

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Run tests
    datamodule = test_datamodule(
        train_data_dir=args.train_dir,
        test_data_dir=args.test_dir,
        labels_path=args.labels,
        use_patches=args.use_patches,
        output_dir=args.output_dir,
    )

    print("\nGenerated files:")
    print(f"  - {args.output_dir}/train_samples.png")
    print(f"  - {args.output_dir}/val_samples.png")
    print(f"  - {args.output_dir}/test_samples.png")
    print(f"  - {args.output_dir}/class_distribution.png")
    print(f"  - {args.output_dir}/augmentation_comparison.png")
    print(f"  - {args.output_dir}/train_batch.png")
    print(f"  - {args.output_dir}/val_batch.png")
    print(f"  - {args.output_dir}/test_batch.png")


if __name__ == "__main__":
    main()
