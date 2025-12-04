from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler


# ==========================================
# Define Custom Dataset
class SubtypeDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        mode: str,
        transform=None,
        train_labels_path: str = None,
    ):
        """
        Args:
            img_dir (str): Directory with all the images.
            train_labels_path (str): Path to the CSV file with training labels.
            mode (str): One of 'train' or 'test'.
            transform: Optional transform to be applied on a sample.
        """
        self.img_dir = Path(img_dir)
        self.transform = transform

        assert mode in ["train", "test"], "mode must be 'train', or 'test'"
        self.mode = mode

        self.label_to_idx = {}
        self.idx_to_label = {}

        if not self.mode == "test":
            # If in training mode, load labels
            if train_labels_path is None:
                raise ValueError("Training mode requires a train_labels_path!")

            df = pd.read_csv(train_labels_path)
            self.img_ids = df.iloc[:, 0].values
            self.labels = df.iloc[:, 1].values

            # Create label mappings
            unique_labels = sorted(list(set(self.labels)))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

            print(f"Label Mapping: {self.label_to_idx}")

        else:
            # Test mode: load all image ids from directory
            self.img_ids = sorted(
                [
                    f.name
                    for f in self.img_dir.iterdir()
                    if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
                    and not f.name.startswith("mask_")
                ]
            )

    def __len__(self):
        return len(self.img_ids)

    def _load_masked_image(self, img_name):
        """Load image and apply mask to remove background"""
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert("RGB")

        # Remove "img_" prefix if present
        if img_name.startswith("img_"):
            img_name = img_name[4:]

        # Find corresponding mask
        mask_name = f"mask_{img_name}"
        mask_path = self.img_dir / mask_name

        if mask_path.exists():
            try:
                mask = Image.open(mask_path).convert("L")
                if mask.size != image.size:
                    mask = mask.resize(image.size, resample=Image.NEAREST)

                mask_np = np.array(mask)
                mask_binary = (mask_np > 100).astype(np.uint8)
                mask_3ch = np.stack([mask_binary] * 3, axis=-1)

                image_np = np.array(image)
                image_masked = image_np * mask_3ch
                image = Image.fromarray(image_masked)
            except Exception as e:
                print(f"Error applying mask for {img_name}: {e}")

        return image

    def __getitem__(self, idx):
        img_name = self.img_ids[idx]

        # Load and apply mask
        image = self._load_masked_image(img_name)

        # Apply Transforms (Resize, Tensor, Norm)
        if self.transform:
            image = self.transform(image)

        if self.mode == "test":
            return image, img_name
        else:
            label_str = self.labels[idx]
            label_idx = self.label_to_idx[label_str]
            return image, torch.tensor(label_idx, dtype=torch.long)


class KFoldDataModule(L.LightningDataModule):
    def __init__(self, dataset, train_idx, val_idx, batch_size=16):
        super().__init__()
        self.dataset = dataset
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.batch_size = batch_size

    def train_dataloader(self):
        sampler = SubsetRandomSampler(self.train_idx)
        return DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=sampler, num_workers=2
        )

    def val_dataloader(self):
        subset = Subset(self.dataset, self.val_idx)
        return DataLoader(
            subset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

    def predict_dataloader(self):
        return self.val_dataloader()
