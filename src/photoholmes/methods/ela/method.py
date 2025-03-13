from typing import Tuple

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from photoholmes.methods.base import BaseMethod, BenchmarkOutput


class ELA(BaseMethod):
    """
    Implementation of ELA (Error Level Analysis). The method detects forgeries
    by analyzing compression artifacts between the original image and its recompressed version.
    """

    def __init__(self, quality: int = 75, display_multiplier: int = 20, **kwargs) -> None:
        """
        Initializes the ELA class with specified parameters.

        Args:
            quality (int): JPEG compression quality to use for recompression.
                Defaults to 75.
            display_multiplier (int): Multiplier for enhancing the visual differences.
                Defaults to 20.
        """
        super().__init__(**kwargs)
        self.quality = quality
        self.display_multiplier = display_multiplier

    def predict(self, image: NDArray, **kwargs) -> NDArray:
        """
        Predicts the error level analysis heatmap from the input image.

        Args:
            image (NDArray): Input image as a numpy array.

        Returns:
            NDArray: The ELA heatmap as a numpy array in RGB format.
        """
        recompressed_image = self._recompress_image(image)
        image_difference = self._get_image_difference(image, recompressed_image)

        int_difference = np.sqrt(image_difference) * self.display_multiplier
        int_difference = np.clip(int_difference, 0, 255).astype(np.uint8)

        return int_difference

    def benchmark(self, image: NDArray, **kwargs) -> BenchmarkOutput:
        """
        Benchmarks the ELA method using the provided image.

        Args:
            image (NDArray): Input image to analyze.

        Returns:
            BenchmarkOutput: Contains the heatmap and placeholders for mask and
                detection.
        """
        heatmap = self.predict(image)
        return {"heatmap": torch.from_numpy(heatmap), "mask": None, "detection": None}

    def _recompress_image(self, image: NDArray) -> NDArray:
        """
        Recompresses the image using JPEG compression.

        Args:
            image (NDArray): The input image to recompress.

        Returns:
            NDArray: The recompressed image in RGB format.
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        if not result:
            raise ValueError("Could not recompress the image.")
        recompressed_image = cv2.imdecode(encimg, 1)
        return recompressed_image

    def _get_image_difference(self, image1: NDArray, image2: NDArray) -> NDArray:
        """
        Calculates the absolute difference between two images.

        Args:
            image1 (NDArray): First image.
            image2 (NDArray): Second image.

        Returns:
            NDArray: The absolute difference between the images.
        """
        diff = image1.astype(np.float32) - image2.astype(np.float32)
        diff = np.abs(diff)
        return diff
