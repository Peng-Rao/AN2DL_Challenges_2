import warnings
from typing import List, Optional, Tuple

import numpy as np


class TissueExtractor:
    """
    Extract patches from images using existing masks.
    Designed for workflow where ground truth masks are already available.
    """

    def __init__(self, patch_size: int = 224, min_tissue_ratio: float = 0.05):
        self.patch_size = patch_size
        self.min_tissue_ratio = min_tissue_ratio

    def _validate_inputs(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Validate inputs and return processed mask."""
        # Check dimensions match
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image shape {img.shape[:2]} doesn't match mask shape {mask.shape[:2]}"
            )

        # Check if image is large enough
        h, w = img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            raise ValueError(
                f"Image dimensions ({h}, {w}) smaller than patch_size ({self.patch_size})"
            )

        # Convert multi-channel mask to single channel
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Handle float masks (threshold at 0.5 if float, otherwise use > 0)
        if mask.dtype in [np.float32, np.float64]:
            warnings.warn("Float mask detected, thresholding at 0.5")
            mask = (mask > 0.5).astype(np.uint8)

        return mask

    def get_valid_patches(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        num_patches: int = 8,
        strategy: str = "random",
        stride: Optional[int] = None,
        shuffle: bool = True,
        min_distance: int = None,  # NEW: minimum distance between random patches
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract patches from image based on mask.

        Args:
            img: RGB image (H, W, 3)
            mask: Binary or Index mask (H, W). Assumes tissue > 0.
            num_patches: Number of patches to extract per image.
            strategy: 'random' samples points from mask; 'grid' slides across image.
            stride: Step size for grid strategy. Defaults to patch_size (no overlap).
            shuffle: Whether to shuffle grid patches before selecting.
            min_distance: Minimum pixel distance between patch centers (random strategy).

        Returns:
            images: List of RGB patches
            masks: List of corresponding Mask patches
        """
        # Validate inputs
        mask = self._validate_inputs(img, mask)
        h, w = img.shape[:2]

        # Find all tissue pixel indices
        tissue_indices = np.where(mask > 0)

        if len(tissue_indices[0]) == 0:
            warnings.warn("No tissue found in mask!")
            return [], []

        if strategy == "random":
            patches_img, patches_mask = self._extract_random(
                img, mask, tissue_indices, num_patches, h, w, min_distance
            )
        elif strategy == "grid":
            patches_img, patches_mask = self._extract_grid(
                img, mask, num_patches, h, w, stride, shuffle
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'random' or 'grid'.")

        # Warn if fewer patches than requested
        if len(patches_img) < num_patches:
            warnings.warn(
                f"Only {len(patches_img)} patches extracted (requested {num_patches})"
            )

        return patches_img, patches_mask

    def _extract_random(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        tissue_indices: Tuple[np.ndarray, np.ndarray],
        num_patches: int,
        h: int,
        w: int,
        min_distance: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Random sampling strategy with optional minimum distance between patches."""

        patches_img = []
        patches_mask = []
        selected_centers = []  # Track selected patch centers

        if min_distance is None:
            min_distance = self.patch_size // 2  # Default: half patch size

        attempts = 0
        max_attempts = num_patches * 100  # Increased for distance constraint

        while len(patches_img) < num_patches and attempts < max_attempts:
            attempts += 1

            idx = np.random.randint(len(tissue_indices[0]))
            cy, cx = tissue_indices[0][idx], tissue_indices[1][idx]

            # Check minimum distance from existing patches
            if min_distance > 0 and selected_centers:
                too_close = False
                for prev_cy, prev_cx in selected_centers:
                    dist = np.sqrt((cy - prev_cy) ** 2 + (cx - prev_cx) ** 2)
                    if dist < min_distance:
                        too_close = True
                        break
                if too_close:
                    continue

            # Calculate patch bounds (centered on the selected pixel)
            half_size = self.patch_size // 2
            y_min = cy - half_size
            x_min = cx - half_size
            y_max = y_min + self.patch_size
            x_max = x_min + self.patch_size

            # Boundary check
            if y_min < 0 or x_min < 0 or y_max > h or x_max > w:
                continue

            # Extract and validate mask patch
            mask_patch = mask[y_min:y_max, x_min:x_max]
            current_ratio = np.count_nonzero(mask_patch) / mask_patch.size

            if current_ratio >= self.min_tissue_ratio:
                img_patch = img[y_min:y_max, x_min:x_max]
                patches_img.append(img_patch)
                patches_mask.append(mask_patch)
                selected_centers.append((cy, cx))

        return patches_img, patches_mask

    def _extract_grid(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        num_patches: int,
        h: int,
        w: int,
        stride: Optional[int] = None,
        shuffle: bool = True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Grid strategy: systematically slide across the TISSUE REGION only."""

        if stride is None:
            stride = self.patch_size

        if stride <= 0:
            raise ValueError(f"Stride must be positive, got {stride}")

        tissue_rows = np.any(mask > 0, axis=1)
        tissue_cols = np.any(mask > 0, axis=0)

        if not tissue_rows.any() or not tissue_cols.any():
            return [], []

        y_min_tissue, y_max_tissue = np.where(tissue_rows)[0][[0, -1]]
        x_min_tissue, x_max_tissue = np.where(tissue_cols)[0][[0, -1]]

        # Add padding (half patch size) to ensure we cover edges
        padding = self.patch_size // 2
        y_start = max(0, y_min_tissue - padding)
        y_end = min(h, y_max_tissue + padding)
        x_start = max(0, x_min_tissue - padding)
        x_end = min(w, x_max_tissue + padding)

        # ===== Grid within tissue bounding box =====
        y_positions = list(range(y_start, y_end - self.patch_size + 1, stride))
        x_positions = list(range(x_start, x_end - self.patch_size + 1, stride))

        valid_patches = []

        for y_min in y_positions:
            for x_min in x_positions:
                y_max = y_min + self.patch_size
                x_max = x_min + self.patch_size

                mask_patch = mask[y_min:y_max, x_min:x_max]
                tissue_ratio = np.count_nonzero(mask_patch) / mask_patch.size

                if tissue_ratio >= self.min_tissue_ratio:
                    valid_patches.append((y_min, x_min, tissue_ratio))

        if shuffle:
            np.random.shuffle(valid_patches)
        else:
            # Sort by tissue ratio (highest first) for deterministic selection
            valid_patches.sort(key=lambda x: x[2], reverse=True)

        patches_img = []
        patches_mask = []

        for y_min, x_min, _ in valid_patches[:num_patches]:
            y_max = y_min + self.patch_size
            x_max = x_min + self.patch_size

            patches_img.append(img[y_min:y_max, x_min:x_max])
            patches_mask.append(mask[y_min:y_max, x_min:x_max])

        return patches_img, patches_mask

    def get_all_valid_patches(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        stride: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
        """Extract ALL valid patches from the image using grid strategy."""

        mask = self._validate_inputs(img, mask)
        h, w = img.shape[:2]

        if stride is None:
            stride = self.patch_size

        y_positions = list(range(0, h - self.patch_size + 1, stride))
        x_positions = list(range(0, w - self.patch_size + 1, stride))

        patches_img = []
        patches_mask = []
        coordinates = []

        for y_min in y_positions:
            for x_min in x_positions:
                y_max = y_min + self.patch_size
                x_max = x_min + self.patch_size

                mask_patch = mask[y_min:y_max, x_min:x_max]
                tissue_ratio = np.count_nonzero(mask_patch) / mask_patch.size

                if tissue_ratio >= self.min_tissue_ratio:
                    patches_img.append(img[y_min:y_max, x_min:x_max])
                    patches_mask.append(mask_patch)
                    coordinates.append((y_min, x_min))

        return patches_img, patches_mask, coordinates
