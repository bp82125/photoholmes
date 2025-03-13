import numpy as np
import torch
from typing import Tuple
import cv2

from photoholmes.methods.base import BaseMethod, BenchmarkOutput
from photoholmes.postprocessing.resizing import ResizeToOriginal
from photoholmes.utils.image import tensor2numpy
from photoholmes.utils.patched_image import PatchedImage


class BlockingArtifacts(BaseMethod):
    """
    Implementation of blocking artifacts detection method. This method detects image manipulations
    by analyzing compression artifacts in the pixel domain, particularly focusing on detecting
    inconsistent blocking artifacts which may indicate tampering.
    """

    def __init__(
        self, diff_threshold: int = 50, kernel_size: int = 33, block_size: int = 8, **kwargs
    ) -> None:
        """
        Initializes the BlockingArtifacts class with specified parameters.

        Args:
            diff_threshold (int): Threshold for pixel differences. Defaults to 50.
            kernel_size (int): Size of kernel for filtering operations. Defaults to 33.
            block_size (int): Size of blocks for analysis. Defaults to 8.
        """
        super().__init__(**kwargs)
        self.diff_threshold = diff_threshold
        self.kernel_size = kernel_size
        self.block_size = block_size

    def predict(self, image: np.ndarray, image_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Predicts the blocking artifact heatmap from image data.

        Args:
            image (np.ndarray): Input image in BGR format.
            image_size (Tuple[int, int], optional): Original image size (height, width) for resizing.
                                                   If None, the input image size is used.

        Returns:
            np.ndarray: The predicted heatmap indicating potential manipulations.
        """
        if image is None:
            raise ValueError("Image could not be processed.")

        luminance = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]

        vertical_artifacts, horizontal_artifacts = self._detect_edge_artifacts(
            luminance)

        vertical_artifacts = cv2.resize(
            vertical_artifacts, (luminance.shape[1], luminance.shape[0]))
        horizontal_artifacts = cv2.resize(
            horizontal_artifacts, (luminance.shape[1], luminance.shape[0]))

        combined_artifacts = vertical_artifacts + horizontal_artifacts
        artifact_map = self._analyze_blocks(combined_artifacts)

        target_size = image_size if image_size is not None else (image.shape[0], image.shape[1])
        
        resizer = ResizeToOriginal(interpolation='bilinear')
        heatmap = resizer(artifact_map, target_size)

        return heatmap

    def benchmark(self, image: np.ndarray) -> BenchmarkOutput:
        """
        Benchmarks the Blocking Artifacts method using provided image.

        Args:
            image (np.ndarray): Input image for analysis.

        Returns:
            BenchmarkOutput: Contains the heatmap and placeholders for mask and detection.
        """
        heatmap = self.predict(image)
        return {
            "heatmap": torch.from_numpy(heatmap),
            "mask": None,
            "detection": None
        }

    def _detect_edge_artifacts(self, luminance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects edge artifacts in the luminance channel of an image.

        Args:
            luminance (np.ndarray): Luminance channel of the image.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Vertical and horizontal edge artifacts.
        """
        vertical_diffs = self._calculate_pixel_differences(luminance, axis=0)
        horizontal_diffs = self._calculate_pixel_differences(luminance, axis=1)

        vertical_artifacts = self._extract_edge_patterns(vertical_diffs, axis=0)
        horizontal_artifacts = self._extract_edge_patterns(horizontal_diffs, axis=1)

        return vertical_artifacts, horizontal_artifacts

    def _calculate_pixel_differences(self, image: np.ndarray, axis: int) -> np.ndarray:
        """
        Calculate the absolute differences between adjacent pixels along a specified axis.

        Args:
            image (np.ndarray): Input image (luminance channel).
            axis (int): Axis along which to calculate differences (0 for vertical, 1 for horizontal).

        Returns:
            np.ndarray: Array of differences between adjacent pixels.
        """
        pad_top_bottom, pad_left_right = (self.kernel_size // 2, self.kernel_size // 2)
        pad_width = ((pad_top_bottom, pad_top_bottom), (0, 0)) if axis == 0 else \
                    ((0, 0), (pad_left_right, pad_left_right))

        padded_image = np.pad(image, pad_width, mode="reflect")

        if axis == 0:  # Vertical differences
            diff = np.abs(2 * padded_image[1:-1] - padded_image[:-2] - padded_image[2:])
        else:  # Horizontal differences
            diff = np.abs(2 * padded_image[:, 1:-1]
                          - padded_image[:, :-2] - padded_image[:, 2:])

        return np.clip(diff, 0, self.diff_threshold)

    def _extract_edge_patterns(self, edge_diff: np.ndarray, axis: int) -> np.ndarray:
        """
        Extract edge patterns from the difference image.

        Args:
            edge_diff (np.ndarray): Difference image.
            axis (int): Axis along which to extract patterns (0 for vertical, 1 for horizontal).

        Returns:
            np.ndarray: Extracted edge patterns.
        """
        kernel_shape = (self.kernel_size, 1) if axis == 0 else (1, self.kernel_size)
        summed_edges = cv2.boxFilter(edge_diff, -1, kernel_shape, normalize=False)
        mid_filtered = cv2.medianBlur(summed_edges.astype(np.uint8), self.kernel_size)
        return summed_edges - mid_filtered

    def _analyze_blocks(self, image: np.ndarray) -> np.ndarray:
        """
        Analyze image blocks for artifact detection.

        Args:
            image (np.ndarray): Input image with combined artifacts.

        Returns:
            np.ndarray: Array of block scores indicating artifact presence.
        """
        # Use PatchedImage for efficient block processing
        tensor_image = torch.from_numpy(image).float()
        patched_img = PatchedImage(tensor_image.unsqueeze(0),
                                   self.block_size, self.block_size)

        blocks = (image.shape[0] // self.block_size, image.shape[1] // self.block_size)
        block_scores = np.zeros(blocks)

        for i in range(blocks[0]):
            for j in range(blocks[1]):
                block = tensor2numpy(patched_img.get_patch(i, j).squeeze(0))
                block_scores[i, j] = self._calculate_artifact_score(block)

        return block_scores

    def _calculate_artifact_score(self, block: np.ndarray) -> float:
        """
        Calculate the artifact score for a single block.

        Args:
            block (np.ndarray): Image block to analyze.

        Returns:
            float: Score indicating likelihood of artifacts in the block.
        """
        block = block.astype(np.float64)

        row_sum = np.sum(block[1:-1, 1:-1], axis=1)
        col_sum = np.sum(block[1:-1, 1:-1], axis=0)
        row_edge = [np.sum(block[1:-1, 0]), np.sum(block[1:-1, -1])]
        col_edge = [np.sum(block[0, 1:-1]), np.sum(block[-1, 1:-1])]

        return np.max(row_sum) + np.max(col_sum) - np.min(row_edge) - np.min(col_edge)
