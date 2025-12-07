from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import PIL


class TissueExtractor:
    """
    Extract patches from images using existing masks.
    Designed for workflow where ground truth masks are already available.

    Args:
        patch_size: Size of the square patch to extract.
        min_tissue_ratio: Minimum ratio of tissue pixels required in a patch.
    """

    def __init__(self, patch_size: int = 224, min_tissue_ratio: float = 0.05):
        self.patch_size = patch_size
        self.min_tissue_ratio = min_tissue_ratio

    def get_valid_patches(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        num_patches: int = 8,
        strategy: str = "random",  # 'random' or 'grid'
        stride: int = None,  # For grid strategy: step size between patches
        shuffle: bool = True,  # For grid strategy: shuffle valid patches before selecting
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Args:
            img: RGB image (H, W, 3)
            mask: Binary or Index mask (H, W). Assumes tissue > 0.
            num_patches: Number of patches to extract per image.
            strategy: 'random' samples points from mask; 'grid' slides across image.
            stride: Step size for grid strategy. Defaults to patch_size (no overlap).
            shuffle: Whether to shuffle grid patches before selecting (for diversity).

        Returns:
            images: List of RGB patches
            masks: List of corresponding Mask patches
        """

        h, w = img.shape[:2]

        # if mask has 3 channels, convert to single channel
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Find all tissue pixel indices
        tissue_indices = np.where(mask > 0)

        # If no tissue found, return empty lists
        if len(tissue_indices[0]) == 0:
            print("Warning: No tissue found in mask!")
            return [], []

        if strategy == "random":
            return self._extract_random(img, mask, tissue_indices, num_patches, h, w)
        elif strategy == "grid":
            return self._extract_grid(img, mask, num_patches, h, w, stride, shuffle)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'random' or 'grid'.")

    def _extract_random(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        tissue_indices: Tuple[np.ndarray, np.ndarray],
        num_patches: int,
        h: int,
        w: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Random sampling strategy: randomly select tissue pixels as patch centers."""

        patches_img = []
        patches_mask = []

        # Protection mechanism: limit the number of attempts to avoid infinite loop
        attempts = 0
        max_attempts = num_patches * 50

        while len(patches_img) < num_patches and attempts < max_attempts:
            attempts += 1

            # Randomly select a tissue pixel as center
            idx = np.random.randint(len(tissue_indices[0]))
            cy, cx = tissue_indices[0][idx], tissue_indices[1][idx]

            # Calculate top-left and bottom-right corners to ensure no out-of-bounds
            half_size = self.patch_size // 2

            # Simple center cropping logic
            y_min = int(cy - half_size)
            x_min = int(cx - half_size)
            y_max = y_min + self.patch_size
            x_max = x_min + self.patch_size

            # Boundary check: if the patch goes out of image bounds, skip and retry
            if y_min < 0 or x_min < 0 or y_max > h or x_max > w:
                continue

            # Extract Mask Patch for validation
            mask_patch = mask[y_min:y_max, x_min:x_max]

            # Calculate the proportion of non-zero pixels
            current_ratio = np.count_nonzero(mask_patch) / mask_patch.size

            if current_ratio >= self.min_tissue_ratio:
                # Extract Image Patch
                img_patch = img[y_min:y_max, x_min:x_max]

                patches_img.append(img_patch)
                patches_mask.append(mask_patch)

        return patches_img, patches_mask

    def _extract_grid(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        num_patches: int,
        h: int,
        w: int,
        stride: int = None,
        shuffle: bool = True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Grid strategy: systematically slide across the image.

        Args:
            img: RGB image
            mask: Binary mask
            num_patches: Maximum number of patches to extract
            h, w: Image dimensions
            stride: Step size between patches. Defaults to patch_size (no overlap).
            shuffle: If True, shuffle valid patches before selecting to add diversity.
        """

        if stride is None:
            stride = self.patch_size  # No overlap by default

        # Calculate all valid grid positions
        y_positions = list(range(0, h - self.patch_size + 1, stride))
        x_positions = list(range(0, w - self.patch_size + 1, stride))

        # Collect all valid patches first
        valid_patches = []  # List of (y_min, x_min, tissue_ratio)

        for y_min in y_positions:
            for x_min in x_positions:
                y_max = y_min + self.patch_size
                x_max = x_min + self.patch_size

                # Extract mask patch for validation
                mask_patch = mask[y_min:y_max, x_min:x_max]

                # Calculate tissue ratio
                tissue_ratio = np.count_nonzero(mask_patch) / mask_patch.size

                if tissue_ratio >= self.min_tissue_ratio:
                    valid_patches.append((y_min, x_min, tissue_ratio))

        # Shuffle or sort based on preference
        if shuffle:
            np.random.shuffle(valid_patches)
        else:
            # Sort by tissue ratio (descending) to prioritize patches with more tissue
            valid_patches.sort(key=lambda x: x[2], reverse=True)

        # Extract the requested number of patches
        patches_img = []
        patches_mask = []

        for y_min, x_min, _ in valid_patches[:num_patches]:
            y_max = y_min + self.patch_size
            x_max = x_min + self.patch_size

            img_patch = img[y_min:y_max, x_min:x_max]
            mask_patch = mask[y_min:y_max, x_min:x_max]

            patches_img.append(img_patch)
            patches_mask.append(mask_patch)

        return patches_img, patches_mask

    def get_all_valid_patches(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        stride: int = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extract ALL valid patches from the image using grid strategy.
        Useful for inference or when you need complete coverage.

        Args:
            img: RGB image (H, W, 3)
            mask: Binary or Index mask (H, W)
            stride: Step size between patches. Defaults to patch_size.

        Returns:
            images: List of RGB patches
            masks: List of corresponding mask patches
            coordinates: List of (y_min, x_min) for each patch
        """
        h, w = img.shape[:2]

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

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
                    img_patch = img[y_min:y_max, x_min:x_max]
                    patches_img.append(img_patch)
                    patches_mask.append(mask_patch)
                    coordinates.append((y_min, x_min))

        return patches_img, patches_mask, coordinates


if __name__ == "__main__":
    test_img = np.array(PIL.Image.open("data/train_data/img_0000.png").convert("RGB"))
    test_mask = np.array(PIL.Image.open("data/train_data/mask_0000.png").convert("L"))

    extractor = TissueExtractor(patch_size=64, min_tissue_ratio=0.1)

    # Test random strategy
    print("\n--- Testing RANDOM strategy ---")
    images_random, masks_random = extractor.get_valid_patches(
        test_img,
        test_mask,
        num_patches=10,
        strategy="random",
    )
    print(f"Random: Extracted {len(images_random)} patches")

    # Test grid strategy with default stride (no overlap)
    print("\n--- Testing GRID strategy (no overlap) ---")
    images_grid, masks_grid = extractor.get_valid_patches(
        test_img,
        test_mask,
        num_patches=10,
        strategy="grid",
        stride=None,  # defaults to patch_size
        shuffle=True,
    )
    print(f"Grid (no overlap): Extracted {len(images_grid)} patches")

    # Test grid strategy with overlap
    print("\n--- Testing GRID strategy (with 50% overlap) ---")
    images_grid_overlap, masks_grid_overlap = extractor.get_valid_patches(
        test_img,
        test_mask,
        num_patches=10,
        strategy="grid",
        stride=32,  # 50% overlap with patch_size=64
        shuffle=False,  # sorted by tissue ratio
    )
    print(f"Grid (50% overlap): Extracted {len(images_grid_overlap)} patches")

    # Test get_all_valid_patches
    print("\n--- Testing get_all_valid_patches ---")
    all_imgs, all_masks, coords = extractor.get_all_valid_patches(
        test_img,
        test_mask,
        stride=64,
    )
    print(f"All valid patches: {len(all_imgs)} patches extracted")
    print(f"First 5 coordinates: {coords[:5]}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Show original image and mask
    axes[0, 0].imshow(test_img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(test_mask, cmap="gray")
    axes[0, 1].set_title("Mask")
    axes[0, 1].axis("off")

    # Show grid coverage
    coverage_map = np.zeros_like(test_mask, dtype=float)
    for y_min, x_min in coords:
        coverage_map[y_min : y_min + 64, x_min : x_min + 64] += 1
    axes[0, 2].imshow(coverage_map, cmap="hot")
    axes[0, 2].set_title(f"Grid Coverage ({len(coords)} patches)")
    axes[0, 2].axis("off")

    # Show sample patches
    for i, (img_patch, mask_patch) in enumerate(zip(images_grid[:3], masks_grid[:3])):
        axes[1, i].imshow(img_patch)
        ratio = np.count_nonzero(mask_patch) / mask_patch.size
        axes[1, i].set_title(f"Grid Patch {i} ({ratio:.1%})")
        axes[1, i].axis("off")

    plt.suptitle("TissueExtractor Demo: Random vs Grid Strategy", fontsize=14)
    plt.tight_layout()
    plt.show()
