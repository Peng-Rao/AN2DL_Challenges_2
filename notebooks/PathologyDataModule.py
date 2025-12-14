from pathlib import Path
from typing import List, Optional

import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import torch
from PathologyDataset import PathologyDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader


class PathologyDataModule(L.LightningDataModule):
    """Lightning DataModule for histopathology image classification.
    Updated to support Global-Local Architecture (img_size propagation).
    """

    def __init__(
        self,
        train_data_dir: str,
        test_data_dir: str,
        train_labels_path: str,
        trash_list_path: str,
        batch_size: int = 16,
        num_workers: int = 2,
        img_size: int = 224,  # Size for Global View
        use_mask: bool = True,
        use_patches: bool = True,
        patch_size: int = 64,  # Size for Local View (Patches)
        num_patches: int = 10,
        min_annotation_pixels: int = 50,
        val_split: float = 0.2,
        random_seed: int = 42,
        train_transform: Optional[A.Compose] = None,
        val_transform: Optional[A.Compose] = None,
        classes: Optional[List[str]] = None,
        use_dual_stream: bool = False,
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
        self.min_annotation_pixels = min_annotation_pixels
        self.val_split = val_split
        self.random_seed = random_seed
        self.use_dual_stream = use_dual_stream

        self.train_transform = train_transform
        self.val_transform = val_transform

        self.classes = (
            classes
            if classes is not None
            else ["Luminal A", "Luminal B", "HER2(+)", "Triple negative"]
        )

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)

        self.train_df = None
        self.val_df = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.sample_weights = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        if stage == "fit" or stage is None:
            full_df = pd.read_csv(self.train_labels_path)

            trash_path = Path(self.trash_list_path)
            if trash_path.exists():
                with open(trash_path, "r") as f:
                    trash_files = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]

                trash_ids = set()
                for t_file in trash_files:
                    clean_id = t_file.replace("img_", "").replace(".png", "")
                    trash_ids.add(clean_id)

                def clean_df_id(x):
                    return str(x).replace("img_", "").replace(".png", "")

                mask = full_df["sample_index"].apply(clean_df_id).isin(trash_ids)
                full_df = full_df[~mask].reset_index(drop=True)

            self.train_df, self.val_df = train_test_split(
                full_df,
                test_size=self.val_split,
                random_state=self.random_seed,
            )

            _, self.sample_weights = self._compute_sample_weights(self.train_df)

            # Training dataset
            self.train_dataset = PathologyDataset(
                data_dir=self.train_data_dir,
                labels_df=self.train_df,
                transform=self.train_transform,
                use_mask=self.use_mask,
                use_patches=self.use_patches,
                patch_size=self.patch_size,
                img_size=self.img_size,  # <--- Added explicit pass
                num_patches=self.num_patches,
                patch_strategy="random",
                min_annotation_pixels=self.min_annotation_pixels,
                is_test=False,
                label_encoder=self.label_encoder,
                use_dual_stream=self.use_dual_stream,
            )

            # Validation dataset
            self.val_dataset = PathologyDataset(
                data_dir=self.train_data_dir,
                labels_df=self.val_df,
                transform=self.val_transform,
                use_mask=self.use_mask,
                use_patches=self.use_patches,
                patch_size=self.patch_size,
                img_size=self.img_size,  # <--- Added explicit pass
                num_patches=self.num_patches,
                patch_strategy="grid",
                stride=self.patch_size // 2,
                min_annotation_pixels=self.min_annotation_pixels,
                is_test=False,
                label_encoder=self.label_encoder,
                use_dual_stream=self.use_dual_stream,
            )

        if stage == "test" or stage == "predict" or stage is None:
            # Test dataset
            self.test_dataset = PathologyDataset(
                data_dir=self.test_data_dir,
                labels_df=None,
                transform=self.val_transform,
                use_mask=self.use_mask,
                use_patches=self.use_patches,
                patch_size=self.patch_size,
                img_size=self.img_size,  # <--- Added explicit pass
                num_patches=self.num_patches,
                patch_strategy="grid",
                stride=self.patch_size // 2,
                min_annotation_pixels=self.min_annotation_pixels,
                is_test=True,
                label_encoder=self.label_encoder,
                use_dual_stream=self.use_dual_stream,
            )

    def _compute_sample_weights(self, df: pd.DataFrame):
        labels = self.label_encoder.transform(df["label"].values)
        class_counts = np.bincount(labels, minlength=len(self.classes))
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(self.classes)
        sample_weights = class_weights[labels]
        return class_weights, sample_weights

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=torch.utils.data.WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True,
            ),
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
