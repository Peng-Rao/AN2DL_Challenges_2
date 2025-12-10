import warnings
from typing import List, Optional, Tuple

import numpy as np


class TissueExtractor:
    """
    Extract patches from images centered around cancer point annotations.
    Designed for workflow where annotations mark cancer locations, and we want
    to extract surrounding tissue context.
    """

    def __init__(self, patch_size: int = 224, min_annotation_pixels: int = 1):
        """
        Args:
            patch_size: Size of square patches to extract.
            min_annotation_pixels: Minimum number of annotation pixels required in patch.
        """
        self.patch_size = patch_size
        self.min_annotation_pixels = min_annotation_pixels

    def _validate_inputs(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Validate inputs and return processed mask."""
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image shape {img.shape[:2]} doesn't match mask shape {mask.shape[:2]}"
            )

        h, w = img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            raise ValueError(
                f"Image dimensions ({h}, {w}) smaller than patch_size ({self.patch_size})"
            )

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

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
        min_distance: int = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract tissue patches centered around cancer annotations in mask.

        Args:
            img: RGB image (H, W, 3) - the full tissue image
            mask: Annotation mask (H, W) - cancer point annotations (sparse)
            num_patches: Number of patches to extract per image.
            strategy: 'random' samples from annotation points; 'grid' finds patches containing annotations.
            stride: Step size for grid strategy. Defaults to patch_size (no overlap).
            shuffle: Whether to shuffle grid patches before selecting.
            min_distance: Minimum pixel distance between patch centers (random strategy).

        Returns:
            images: List of RGB patches (surrounding tissue context)
            masks: List of corresponding annotation patches (sparse cancer markers)
        """
        mask = self._validate_inputs(img, mask)
        h, w = img.shape[:2]

        # Find all annotation pixel indices
        annotation_indices = np.where(mask > 0)

        if len(annotation_indices[0]) == 0:
            warnings.warn("No annotations found in mask!")
            return [], []

        if strategy == "random":
            patches_img, patches_mask = self._extract_random(
                img, mask, annotation_indices, num_patches, h, w, min_distance
            )
        elif strategy == "grid":
            patches_img, patches_mask = self._extract_grid(
                img, mask, num_patches, h, w, stride, shuffle
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'random' or 'grid'.")

        if len(patches_img) < num_patches:
            warnings.warn(
                f"Only {len(patches_img)} patches extracted (requested {num_patches})"
            )

        return patches_img, patches_mask

    def _extract_random(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        annotation_indices: Tuple[np.ndarray, np.ndarray],
        num_patches: int,
        h: int,
        w: int,
        min_distance: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Random sampling: center patches on annotation points to capture surrounding tissue.
        """
        patches_img = []
        patches_mask = []
        selected_centers = []

        if min_distance is None:
            min_distance = self.patch_size // 2

        attempts = 0
        max_attempts = num_patches * 100

        while len(patches_img) < num_patches and attempts < max_attempts:
            attempts += 1

            # Sample a random annotation point
            idx = np.random.randint(len(annotation_indices[0]))
            cy, cx = annotation_indices[0][idx], annotation_indices[1][idx]

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

            # Calculate patch bounds centered on annotation point
            half_size = self.patch_size // 2
            y_min = cy - half_size
            x_min = cx - half_size
            y_max = y_min + self.patch_size
            x_max = x_min + self.patch_size

            # Boundary check
            if y_min < 0 or x_min < 0 or y_max > h or x_max > w:
                continue

            # Extract patches - image contains surrounding tissue, mask contains annotation
            img_patch = img[y_min:y_max, x_min:x_max]
            mask_patch = mask[y_min:y_max, x_min:x_max]

            # Check minimum annotation pixels (at least some annotation in patch)
            if np.count_nonzero(mask_patch) >= self.min_annotation_pixels:
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
        """
        Grid strategy: find patches that contain annotation points.
        Prioritizes patches with more annotation pixels.
        """
        if stride is None:
            stride = self.patch_size

        if stride <= 0:
            raise ValueError(f"Stride must be positive, got {stride}")

        # Find annotation bounding box to focus search
        annotation_rows = np.any(mask > 0, axis=1)
        annotation_cols = np.any(mask > 0, axis=0)

        if not annotation_rows.any() or not annotation_cols.any():
            return [], []

        y_min_ann, y_max_ann = np.where(annotation_rows)[0][[0, -1]]
        x_min_ann, x_max_ann = np.where(annotation_cols)[0][[0, -1]]

        # Expand search region to capture surrounding tissue
        padding = self.patch_size
        y_start = max(0, y_min_ann - padding)
        y_end = min(h, y_max_ann + padding)
        x_start = max(0, x_min_ann - padding)
        x_end = min(w, x_max_ann + padding)

        y_positions = list(range(y_start, y_end - self.patch_size + 1, stride))
        x_positions = list(range(x_start, x_end - self.patch_size + 1, stride))

        valid_patches = []

        for y_min in y_positions:
            for x_min in x_positions:
                y_max = y_min + self.patch_size
                x_max = x_min + self.patch_size

                mask_patch = mask[y_min:y_max, x_min:x_max]
                annotation_count = np.count_nonzero(mask_patch)

                # Only include patches that contain annotations
                if annotation_count >= self.min_annotation_pixels:
                    valid_patches.append((y_min, x_min, annotation_count))

        if shuffle:
            np.random.shuffle(valid_patches)
        else:
            # Sort by annotation count (highest first) for deterministic selection
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
        """Extract ALL patches containing annotations from the image."""

        mask = self._validate_inputs(img, mask)
        h, w = img.shape[:2]

        if stride is None:
            stride = self.patch_size

        # Focus on region around annotations
        annotation_rows = np.any(mask > 0, axis=1)
        annotation_cols = np.any(mask > 0, axis=0)

        if not annotation_rows.any() or not annotation_cols.any():
            return [], [], []

        y_min_ann, y_max_ann = np.where(annotation_rows)[0][[0, -1]]
        x_min_ann, x_max_ann = np.where(annotation_cols)[0][[0, -1]]

        padding = self.patch_size
        y_start = max(0, y_min_ann - padding)
        y_end = min(h, y_max_ann + padding)
        x_start = max(0, x_min_ann - padding)
        x_end = min(w, x_max_ann + padding)

        y_positions = list(range(y_start, y_end - self.patch_size + 1, stride))
        x_positions = list(range(x_start, x_end - self.patch_size + 1, stride))

        patches_img = []
        patches_mask = []
        coordinates = []

        for y_min in y_positions:
            for x_min in x_positions:
                y_max = y_min + self.patch_size
                x_max = x_min + self.patch_size

                mask_patch = mask[y_min:y_max, x_min:x_max]
                annotation_count = np.count_nonzero(mask_patch)

                if annotation_count >= self.min_annotation_pixels:
                    patches_img.append(img[y_min:y_max, x_min:x_max])
                    patches_mask.append(mask_patch)
                    coordinates.append((y_min, x_min))

        return patches_img, patches_mask, coordinates
