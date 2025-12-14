import random
from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from TissueExtractor import TissueExtractor
from torch.utils.data import Dataset


class PathologyDataset(Dataset):
    """Dataset optimized for histopathology images."""

    def __init__(
        self,
        data_dir: str,
        labels_df: Optional[pd.DataFrame] = None,
        transform: Optional[A.Compose] = None,
        use_mask: bool = True,
        use_patches: bool = False,
        patch_size: int = 224,
        num_patches: int = 8,
        patch_strategy: str = "random",
        stride: Optional[int] = None,
        min_annotation_pixels: int = 100,
        is_test: bool = False,
        label_encoder: Optional[LabelEncoder] = None,
        use_dual_stream: bool = False,
    ):
        """
        Args:
            data_dir: Directory with images and masks.
            labels_df: DataFrame with 'sample_index' and 'label' columns (None for test).
            transform: torchvision transforms to apply to images.
            use_mask: Whether to load and use masks.
            use_patches: Whether to extract patches or use full images.
            patch_size: Size of square patches to extract.
            num_patches: Number of patches to extract per image.
            patch_strategy: 'random' or 'grid' strategy for patch extraction.
            stride: Step size for grid strategy. Defaults to patch_size (no overlap).
            min_annotation_pixels: Minimum annotation pixels required in patch.
            is_test: Whether the dataset is for testing (no labels).
            label_encoder: Pre-fitted LabelEncoder (if None, will fit on training labels).
            use_dual_stream: Whether to return masks alongside images.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_mask = use_mask
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_strategy = patch_strategy
        self.stride = stride
        self.min_annotation_pixels = min_annotation_pixels
        self.is_test = is_test
        self.label_encoder = label_encoder
        self.use_dual_stream = use_dual_stream

        if self.transform is None:
            self.transform = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

        self.tissue_extractor = TissueExtractor(
            patch_size=patch_size,
            min_annotation_pixels=min_annotation_pixels,
        )

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
                self.label_encoder.fit(self.labels)
            self.encoded_labels = self.label_encoder.transform(self.labels)

    def _clean_sample_idx(self, sample_idx: str) -> str:
        sample_idx = str(sample_idx)
        if sample_idx.startswith("img_"):
            sample_idx = sample_idx[4:]
        if sample_idx.endswith(".png"):
            sample_idx = sample_idx[:-4]
        return sample_idx

    def _get_test_samples(self) -> List[str]:
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
        img_path = self.data_dir / f"img_{sample_idx}.png"
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = None
        if self.use_mask:
            mask_path = self.data_dir / f"mask_{sample_idx}.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        return img, mask

    def _load_patches(
        self, sample_idx: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load image patches AND mask patches.
        Ensures both lists stay synchronized during padding/augmentation.
        """
        img, mask = self._load_image_and_mask(sample_idx)

        if mask is None:
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

        # Extract patches (TissueExtractor returns both)
        patches, mask_patches = self.tissue_extractor.get_valid_patches(
            img=img,
            mask=mask,
            num_patches=self.num_patches,
            strategy=self.patch_strategy,
            stride=self.patch_size // 2,
            shuffle=False,
            min_distance=32,
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

            fallback_mask = mask[y1:y2, x1:x2]
            fallback_mask = cv2.resize(
                fallback_mask,
                (self.patch_size, self.patch_size),
                interpolation=cv2.INTER_NEAREST,
            )

            patches = [fallback_patch.copy() for _ in range(self.num_patches)]
            mask_patches = [fallback_mask.copy() for _ in range(self.num_patches)]

        elif len(patches) < self.num_patches:
            num_missing = self.num_patches - len(patches)
            indices = [random.randint(0, len(patches) - 1) for _ in range(num_missing)]

            for idx in indices:
                patch = patches[idx].copy()
                mask_patch = mask_patches[idx].copy()

                if random.random() > 0.5:
                    patch = cv2.flip(patch, 1)
                    mask_patch = cv2.flip(mask_patch, 1)

                patches.append(patch)
                mask_patches.append(mask_patch)

        return patches, mask_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        sample_idx = self.samples[idx]

        processed_images = []
        processed_masks = []

        if self.use_patches:
            images_np, masks_np = self._load_patches(sample_idx)
        else:
            img, mask = self._load_image_and_mask(sample_idx)
            images_np = [img]
            masks_np = [mask] if mask is not None else []

        for i in range(len(images_np)):
            curr_img = images_np[i]
            transform_args = {"image": curr_img}
            has_mask = (
                self.use_dual_stream
                and (i < len(masks_np))
                and (masks_np[i] is not None)
            )

            if has_mask:
                transform_args["mask"] = masks_np[i]

            augmented = self.transform(**transform_args)

            processed_images.append(augmented["image"])

            if has_mask:
                m_tensor = augmented["mask"]
                if m_tensor.ndim == 2:
                    m_tensor = m_tensor.unsqueeze(0)
                m_tensor = m_tensor.float() / 255.0
                processed_masks.append(m_tensor)

        if self.use_patches:
            img_tensor = torch.stack(processed_images)  # [Num_Patches, C, H, W]
            mask_tensor = (
                torch.stack(processed_masks) if processed_masks else None
            )  # [Num_Patches, 1, H, W]
        else:
            img_tensor = processed_images[0]
            mask_tensor = processed_masks[0] if processed_masks else None

        if self.is_test:
            if self.use_dual_stream and mask_tensor is not None:
                return img_tensor, mask_tensor, sample_idx
            return img_tensor, sample_idx
        else:
            label = self.encoded_labels[idx]
            label_t = torch.tensor(label, dtype=torch.long)
            if self.use_dual_stream and mask_tensor is not None:
                return img_tensor, mask_tensor, label_t
            return img_tensor, label_t


if __name__ == "__main__":
    train_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
