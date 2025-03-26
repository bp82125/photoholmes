import glob
import os
from typing import List, Optional, Tuple

from torch import Tensor

from .base import BaseDataset


class InTheWildDataset(BaseDataset):
    """
    Class for the In-the-Wild forensics dataset.

    Directory structure:
    img_dir (In-the-Wild forensics dataset)
    ├── images
    │   ├── [images in JPG]
    └── masks
        ├── [masks in PNG]
    """

    IMAGES_DIR: str = "images"
    MASKS_DIR: str = "masks"
    IMAGE_EXTENSION: str = ".jpg"
    MASK_EXTENSION: str = ".png"

    def _get_paths(
        self, dataset_path: str, tampered_only: bool
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        """
        Get the paths of the images and masks in the dataset.

        Args:
            dataset_path (str): Path to the dataset.
            tampered_only (bool): Whether to load only the tampered images.

        Returns:
            Tuple[List[str], List[str] | List[str | None]]: Paths of the images and
                masks.
        """
        # Get all image paths
        image_paths = glob.glob(
            os.path.join(dataset_path, self.IMAGES_DIR, f"*{self.IMAGE_EXTENSION}")
        )

        image_mask_pairs = []
        mask_paths = []

        for image_path in image_paths:
            mask_path = self._get_mask_path(image_path)
            full_mask_path = os.path.join(dataset_path, mask_path)

            if os.path.exists(full_mask_path):
                image_mask_pairs.append((image_path, full_mask_path))
            elif not tampered_only:

                image_mask_pairs.append((image_path, None))

        if image_mask_pairs:
            image_paths, mask_paths = zip(*image_mask_pairs)
            image_paths = list(image_paths)
            mask_paths = list(mask_paths)
        else:
            image_paths = []
            mask_paths = []

        return image_paths, mask_paths

    def _get_mask_path(self, image_path: str) -> str:
        """
        Get the path of the mask for the given image path.

        Args:
            image_path (str): Path to the image.

        Returns:
            str: Path to the mask.
        """
        image_filename = os.path.basename(image_path)
        image_name = os.path.splitext(image_filename)[0]
        mask_filename = image_name + self.MASK_EXTENSION
        return os.path.join(self.MASKS_DIR, mask_filename)

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        """
        Binarize the mask. Assumes mask is already in binary format
        where positive values represent tampered regions.

        Args:
            mask_image (Tensor): Mask image.

        Returns:
            Tensor: Binarized mask image.
        """
        if mask_image.dim() > 2 and mask_image.shape[0] > 1:          # Take first channel (or you could sum across channels)
            return mask_image[0, :, :] > 0

        return mask_image > 0
