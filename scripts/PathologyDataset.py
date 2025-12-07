from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms

from utils.TissueExtractor import TissueExtractor


class PathologyDataset(Dataset):
    """Dataset optimized for histopathology images.

    Args:
        data_dir: Directory containing images and masks.
        labels_df: DataFrame with 'sample_index' and 'label' columns for training/validation.
        transform: torchvision transforms to apply to images.
        img_size: Target size to resize images to (img_size x img_size). If using patches, this is ignored.
        use_mask: Whether to use existing masks for tissue extraction.
        use_patches: Whether to load images as patches.
        patch_size: Size of each patch if using patches.
        num_patches: Number of patches to extract per image.
        patch_strategy: Strategy for patch extraction ('random' or 'grid').
        min_tissue_ratio: Minimum tissue ratio for valid patches.
        use_stain_norm: Whether to apply stain normalization.
        is_test: Whether the dataset is for testing (no labels).
        label_encoder: Optional LabelEncoder for encoding labels.
    """

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
        use_stain_norm: bool = True,
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
        self.stain_normalizer = None if use_stain_norm else None

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
                self.label_encoder.fit(
                    ["Luminal A", "Luminal B", "HER2(+)", "Triple negative"]
                )
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

        if self.use_mask:
            mask_path = self.data_dir / f"mask_{sample_idx}.png"
            if mask_path.exists():
                mask = np.array(Image.open(mask_path).convert("L"))

        return img, mask

    def _crop_to_tissue_bbox(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Crop image to bounding box of tissue region."""
        # Find bounding box of tissue
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)

        if not rows.any() or not cols.any():
            return img  # No tissue found, return original

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Add small padding
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
                pass  # Skip normalization if it fails
        return img

    def _load_and_preprocess(self, sample_idx: str) -> np.ndarray:
        """Load and preprocess full image with optional tissue cropping."""
        img, mask = self._load_image_and_mask(sample_idx)

        # Crop to tissue region if mask is available
        if mask is not None:
            img = self._crop_to_tissue_bbox(img, mask)

        # Stain normalization
        if self.use_stain_norm:
            img = self._apply_stain_normalization(img)

        return img

    def _load_patches(self, sample_idx: str) -> List[np.ndarray]:
        """Load image as patches using TissueExtractor."""
        img, mask = self._load_image_and_mask(sample_idx)

        if mask is None:
            # If no mask, create a simple one (all tissue)
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

        # Extract patches using TissueExtractor
        patches, _ = self.tissue_extractor.get_valid_patches(
            img=img,
            mask=mask,
            num_patches=self.num_patches,
            strategy=self.patch_strategy,
            stride=self.patch_size // 2,
            shuffle=False,
        )

        # Handle case where fewer patches are found than requested
        if len(patches) == 0:
            # Fallback: extract center crop
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
            # Duplicate existing patches to reach num_patches
            while len(patches) < self.num_patches:
                patches.append(patches[len(patches) % len(patches)])

        # Apply stain normalization to each patch
        normalized_patches = []
        for patch in patches:
            patch = self._apply_stain_normalization(patch)
            normalized_patches.append(patch)

        return normalized_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        sample_idx = self.samples[idx]

        if self.use_patches:
            patches = self._load_patches(sample_idx)

            # Transform each patch
            transformed_patches = []
            for patch in patches:
                patch_pil = Image.fromarray(patch)
                if self.transform:
                    patch_tensor = self.transform(patch_pil)
                else:
                    patch_tensor = transforms.ToTensor()(patch_pil)
                transformed_patches.append(patch_tensor)

            # Stack patches [num_patches, C, H, W]
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


# Example usage
if __name__ == "__main__":
    TRAIN_DATA_DIR = "./data/train_data"
    TEST_DATA_DIR = "./data/test_data"
    TRAIN_LABELS_PATH = "./data/train_labels.csv"

    # Load training labels
    train_df = pd.read_csv(TRAIN_LABELS_PATH)

    # Define simple transforms for testing
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Test 1: Without patches (whole images)
    print("=" * 50)
    print("Test 1: Whole Images (no patches)")
    print("=" * 50)

    dataset_whole = PathologyDataset(
        data_dir=TRAIN_DATA_DIR,
        labels_df=train_df,
        transform=test_transform,
        use_mask=True,
        use_patches=False,
        use_stain_norm=False,
        is_test=False,
    )

    # Use a fixed sample index for all tests
    fixed_idx = 0
    fixed_sample_id = dataset_whole.samples[fixed_idx]
    fixed_label_idx = dataset_whole.encoded_labels[fixed_idx]
    fixed_label_name = dataset_whole.label_encoder.inverse_transform([fixed_label_idx])[
        0
    ]

    # Build a small batch around the fixed sample for visualization
    # Non-blocking drawing enabled
    import matplotlib.pyplot as plt

    # Gather 4 images (repeat fixed sample if needed)
    imgs = []
    labels = []
    for i in range(4):
        img_tensor, label_tensor = dataset_whole[
            fixed_idx if i == 0 else i % len(dataset_whole)
        ]
        imgs.append(img_tensor)
        labels.append(label_tensor)

    # Print shapes for Test 1
    print(f"Test 1 - single image tensor shape: {imgs[0].shape}")  # [3, 224, 224]
    print(f"Test 1 - label tensor shape: {labels[0].shape}")  # torch.Size([])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        img = imgs[i].permute(1, 2, 0).numpy()
        label_name_i = dataset_whole.label_encoder.inverse_transform(
            [labels[i].item()]
        )[0]
        sample_id_i = dataset_whole.samples[
            fixed_idx if i == 0 else i % len(dataset_whole)
        ]
        axes[i].imshow(img)
        axes[i].set_title(f"{label_name_i}\n{sample_id_i}", fontsize=10)
        axes[i].axis("off")
    plt.suptitle(
        "PathologyDataset - Whole Images (First Batch, fixed sample in slot 0)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    # Test 2: With patches (same fixed sample)
    print("\n" + "=" * 50)
    print("Test 2: Patch-based Images (same sample)")
    print("=" * 50)

    dataset_patches = PathologyDataset(
        data_dir=TRAIN_DATA_DIR,
        labels_df=train_df,
        transform=test_transform,
        use_mask=True,
        use_patches=True,
        patch_size=224,
        num_patches=8,
        patch_strategy="random",
        min_tissue_ratio=0.05,
        use_stain_norm=False,
        is_test=False,
    )

    # Fetch patches for the same fixed sample
    patches_batch, label_tensor = dataset_patches[fixed_idx]  # [num_patches, C, H, W]

    # Print shapes for Test 2
    print(f"Test 2 - patches tensor shape: {patches_batch.shape}")  # [8, 3, 224, 224]
    print(f"Test 2 - label tensor shape: {label_tensor.shape}")  # torch.Size([])

    fig, axes = plt.subplots(2, 8, figsize=(20, 5))

    # Row labels for non-blocking display
    row_labels = [
        f"{fixed_label_name}\n{fixed_sample_id}",
        f"{fixed_label_name}\n{fixed_sample_id}",
    ]

    # Show two rows (repeat first sample for row 2 to fill grid)
    for sample_row in range(2):
        for patch_idx in range(8):
            img = patches_batch[patch_idx].permute(1, 2, 0).numpy()
            axes[sample_row, patch_idx].imshow(img)
            axes[sample_row, patch_idx].set_title(f"P{patch_idx}", fontsize=8)
            axes[sample_row, patch_idx].axis("off")

    # Add row labels on the figure (left side)
    fig.text(0.02, 0.75, row_labels[0], va="center", ha="left", fontsize=10)
    fig.text(0.02, 0.25, row_labels[1], va="center", ha="left", fontsize=10)

    plt.suptitle("PathologyDataset - Patches (Fixed Sample)", fontsize=14)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    # Test 3: Grid-based patch extraction (same fixed sample)
    print("\n" + "=" * 50)
    print("Test 3: Grid-based Patch Extraction (same sample)")
    print("=" * 50)

    dataset_grid_patches = PathologyDataset(
        data_dir=TRAIN_DATA_DIR,
        labels_df=train_df,
        transform=test_transform,
        use_mask=True,
        use_patches=True,
        patch_size=224,
        num_patches=9,  # 3x3 grid
        patch_strategy="grid",
        min_tissue_ratio=0.05,
        use_stain_norm=False,
        is_test=False,
    )

    grid_patches, grid_label_tensor = dataset_grid_patches[fixed_idx]  # [9, C, H, W]

    # Print shapes for Test 3
    print(
        f"Test 3 - grid patches tensor shape: {grid_patches.shape}"
    )  # [9, 3, 224, 224]
    print(f"Test 3 - label tensor shape: {grid_label_tensor.shape}")  # torch.Size([])

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for patch_idx in range(9):
        img = grid_patches[patch_idx].permute(1, 2, 0).numpy()
        row = patch_idx // 3
        col = patch_idx % 3
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"P{patch_idx}", fontsize=10)
        axes[row, col].axis("off")

    plt.suptitle(
        f"PathologyDataset - Grid-based Patches\n{fixed_label_name} - {fixed_sample_id}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(20)
