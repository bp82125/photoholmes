from typing import Tuple

import cv2
import numpy as np
import pywt
import torch
from numpy.typing import NDArray

from photoholmes.methods.base import BaseMethod, BenchmarkOutput
from photoholmes.postprocessing.resizing import ResizeToOriginal


class Wavelet(BaseMethod):
    """
    Implementation of Wavelet-based noise analysis method. This method detects forgeries
    by analyzing the high-frequency noise patterns in wavelet domain.
    """

    def __init__(self, block_size: int = 8, **kwargs) -> None:
        """
        Initializes the Wavelet class with specified parameters.

        Args:
            block_size (int): Size of blocks for noise estimation.
                Defaults to 8.
        """
        super().__init__(**kwargs)
        self.block_size = block_size
        self.resizer = ResizeToOriginal()

    def predict(
        self, image: NDArray, image_size: Tuple[int, int] = None
    ) -> NDArray:
        """
        Predicts the noise map using wavelet-based analysis.

        Args:
            image (np.ndarray): The input image.
            image_size (Tuple[int, int], optional): Tuple representing the dimensions of the
                image. If None, the original image size is used.

        Returns:
            NDArray: The predicted noise map.
        """
        # Extract color channels
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)

        # Process each channel
        noise_Y = self._process_channel(Y, self.block_size)
        noise_Cr = self._process_channel(Cr, self.block_size)
        noise_Cb = self._process_channel(Cb, self.block_size)

        combined_noise_map = np.maximum.reduce([noise_Y, noise_Cr, noise_Cb])

        target_size = image_size if image_size is not None else (
            image.shape[0], image.shape[1])

        heatmap = self.resizer(combined_noise_map, target_size)

        return heatmap

    def benchmark(
        self, image: NDArray, image_size: Tuple[int, int] = None
    ) -> BenchmarkOutput:
        """
        Benchmarks the Wavelet method using the provided image.

        Args:
            image (np.ndarray): The input image.
            image_size (Tuple[int, int], optional): Dimensions of the image.
                If None, the original image size is used.

        Returns:
            BenchmarkOutput: Contains the heatmap and placeholders for mask and
                detection.
        """
        heatmap = self.predict(image, image_size)
        return {"heatmap": torch.from_numpy(heatmap), "mask": None, "detection": None}

    def _estimate_noise_std(self, block: NDArray) -> float:
        """
        Estimates the noise standard deviation using Median Absolute Deviation (MAD).

        Args:
            block (NDArray): Small block of wavelet coefficients.

        Returns:
            float: Estimated noise standard deviation.
        """
        MAD_CONSTANT = 0.6745
        return np.median(np.abs(block - np.median(block))) / MAD_CONSTANT

    def _process_channel(self, channel: NDArray, block_size: int = 8) -> NDArray:
        """
        Processes a single image channel to estimate the noise map.

        Args:
            channel (NDArray): Single channel of the image.
            block_size (int): Size of blocks for noise estimation. Defaults to 8.

        Returns:
            NDArray: Noise map for the channel.
        """
        coeffs2 = pywt.dwt2(channel, 'db8')
        _, (_, _, HH) = coeffs2

        height, width = HH.shape
        noise_map = np.zeros((height, width))

        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = HH[i:i + block_size, j:j + block_size]
                if block.shape == (block_size, block_size):
                    noise_map[i:i + block_size, j:j
                              + block_size] = self._estimate_noise_std(block)

        return noise_map
