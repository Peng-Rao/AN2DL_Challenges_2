import warnings
from typing import List, Optional, Tuple

import numpy as np


class TissueExtractor:
    """
    Extract patches from images centered around cancer point annotations.
    If cancer annotations are insufficient to fill the requested number of patches,
    it pads the result with normal tissue patches (areas with no annotations).
    """

    def __init__(self, patch_size: int = 224, min_annotation_pixels: int = 1):
        """
        Args:
            patch_size: Size of square patches to extract.
            min_annotation_pixels: Minimum number of annotation pixels required in a CANCER patch.
        """
        self.patch_size = patch_size
        self.min_annotation_pixels = min_annotation_pixels

    def _validate_inputs(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Validate inputs and return processed mask.
        """
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
        Extract tissue patches.
        1. Tries to fill num_patches with cancer-positive patches first.
        2. If not enough cancer patches are found, fills the remainder with normal tissue.

        Args:
            img: RGB image (H, W, 3).
            mask: Annotation mask (H, W).
            num_patches: Total number of patches desired.
            strategy: 'random' or 'grid' (applies to cancer extraction).
            stride: Step size for grid strategy.
            shuffle: Whether to shuffle grid patches.
            min_distance: Minimum pixel distance between cancer patch centers.

        Returns:
            images: List of RGB patches.
            masks: List of corresponding mask patches.
        """
        mask = self._validate_inputs(img, mask)
        h, w = img.shape[:2]

        # --- 1. Extract Cancer Patches ---
        annotation_indices = np.where(mask > 0)
        has_annotations = len(annotation_indices[0]) > 0

        patches_img = []
        patches_mask = []

        if has_annotations:
            if strategy == "random":
                patches_img, patches_mask = self._extract_random(
                    img, mask, annotation_indices, num_patches, h, w, min_distance
                )
            elif strategy == "grid":
                patches_img, patches_mask = self._extract_grid(
                    img, mask, num_patches, h, w, stride, shuffle
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        # --- 2. Check Deficit and Pad with Normal Tissue ---
        num_extracted = len(patches_img)

        if num_extracted < num_patches:
            needed = num_patches - num_extracted
            # warnings.warn(
            #     f"Found only {num_extracted} cancer patches. Padding with {needed} normal patches."
            # )

            normal_imgs, normal_masks = self._extract_normal_patches(
                img, mask, needed, h, w
            )

            patches_img.extend(normal_imgs)
            patches_mask.extend(normal_masks)

        # Final check if we still failed to meet quota (e.g. image too small or empty)
        if len(patches_img) < num_patches:
            warnings.warn(
                f"Could only extract {len(patches_img)} total patches (requested {num_patches})."
            )

        return patches_img, patches_mask

    def _extract_normal_patches(
        self, img: np.ndarray, mask: np.ndarray, num_patches: int, h: int, w: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract patches that contain NO cancer annotations (normal tissue).
        Tries to avoid pure white background if possible.
        """
        normal_imgs = []
        normal_masks = []

        attempts = 0
        # High max_attempts because finding tissue in a sparse slide can be hard
        max_attempts = num_patches * 200
        # Valid range for patch centers
        y_range = h - self.patch_size
        x_range = w - self.patch_size

        if y_range <= 0 or x_range <= 0:
            return [], []

        while len(normal_imgs) < num_patches and attempts < max_attempts:
            attempts += 1

            # Random top-left corner
            y_min = np.random.randint(0, y_range)
            x_min = np.random.randint(0, x_range)
            y_max = y_min + self.patch_size
            x_max = x_min + self.patch_size

            mask_patch = mask[y_min:y_max, x_min:x_max]

            # CRITICAL: Normal tissue means mask must be empty (0 annotations)
            if np.count_nonzero(mask_patch) == 0:
                img_patch = img[y_min:y_max, x_min:x_max]
                mean_intensity = np.mean(img_patch)
                if mean_intensity < 235:
                    normal_imgs.append(img_patch)
                    normal_masks.append(mask_patch)

        return normal_imgs, normal_masks

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
        Random sampling centered on positive annotation points.
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

            idx = np.random.randint(len(annotation_indices[0]))
            cy, cx = annotation_indices[0][idx], annotation_indices[1][idx]

            if min_distance > 0 and selected_centers:
                # Vectorized distance check for speed
                centers_array = np.array(selected_centers)
                dists = np.sqrt(np.sum((centers_array - [cy, cx]) ** 2, axis=1))
                if np.any(dists < min_distance):
                    continue

            half_size = self.patch_size // 2
            y_min = cy - half_size
            x_min = cx - half_size
            y_max = y_min + self.patch_size
            x_max = x_min + self.patch_size

            if y_min < 0 or x_min < 0 or y_max > h or x_max > w:
                continue

            img_patch = img[y_min:y_max, x_min:x_max]
            mask_patch = mask[y_min:y_max, x_min:x_max]

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
        Grid strategy: find patches containing annotations.
        """
        if stride is None:
            stride = self.patch_size

        if stride <= 0:
            raise ValueError(f"Stride must be positive, got {stride}")

        annotation_rows = np.any(mask > 0, axis=1)
        annotation_cols = np.any(mask > 0, axis=0)

        if not annotation_rows.any() or not annotation_cols.any():
            return [], []

        y_min_ann, y_max_ann = np.where(annotation_rows)[0][[0, -1]]
        x_min_ann, x_max_ann = np.where(annotation_cols)[0][[0, -1]]

        padding = self.patch_size
        y_start = max(0, y_min_ann - padding)
        y_end = min(h, y_max_ann + padding)
        x_start = max(0, x_min_ann - padding)
        x_end = min(w, x_max_ann + padding)

        y_positions = range(y_start, y_end - self.patch_size + 1, stride)
        x_positions = range(x_start, x_end - self.patch_size + 1, stride)

        valid_patches = []

        for y_min in y_positions:
            for x_min in x_positions:
                y_max = y_min + self.patch_size
                x_max = x_min + self.patch_size

                mask_patch = mask[y_min:y_max, x_min:x_max]
                annotation_count = np.count_nonzero(mask_patch)

                if annotation_count >= self.min_annotation_pixels:
                    valid_patches.append((y_min, x_min, annotation_count))

        if shuffle:
            np.random.shuffle(valid_patches)
        else:
            valid_patches.sort(key=lambda x: x[2], reverse=True)

        patches_img = []
        patches_mask = []

        for y_min, x_min, _ in valid_patches[:num_patches]:
            y_max = y_min + self.patch_size
            x_max = x_min + self.patch_size

            patches_img.append(img[y_min:y_max, x_min:x_max])
            patches_mask.append(mask[y_min:y_max, x_min:x_max])

        return patches_img, patches_mask
