from typing import Optional, Tuple
from photoholmes.postprocessing.resizing import ResizeToOriginal

from .backbones.convnext import ConvNeXt
from .backbones.segformer import MixVisionTransformer
from .extractors.low_frequency_feature_extraction import LowDctFrequencyExtractor
from .extractors.high_frequency_feature_extraction import HighDctFrequencyExtractor

import torch

import torch.nn as nn
import torch.nn.functional as F
import sys
from photoholmes.methods.base import BaseTorchMethod, BenchmarkOutput

sys.path.append('.')


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class UpsampleConcatConv(nn.Module):
    def __init__(self):
        super(UpsampleConcatConv, self).__init__()
        self.upsamplec2 = nn.ConvTranspose2d(
            192, 96, kernel_size=4, stride=2, padding=1)

        self.upsamples2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1)

        self.upsamplec3 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        )

        self.upsamples3 = nn.Sequential(
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )

        self.upsamplec4 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        )

        self.upsamples4 = nn.Sequential(
            nn.ConvTranspose2d(512, 320, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, inputs):
        c1, c2, c3, c4, s1, s2, s3, s4 = inputs

        c2 = self.upsamplec2(c2)
        c3 = self.upsamplec3(c3)
        c4 = self.upsamplec4(c4)
        s2 = self.upsamples2(s2)
        s3 = self.upsamples3(s3)
        s4 = self.upsamples4(s4)

        x = torch.cat([c1, c2, c3, c4, s1, s2, s3, s4], dim=1)
        features = [c1, c2, c3, c4, s1, s2, s3, s4]

        return x, features


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ScoreNetwork(nn.Module):
    def __init__(self):
        super(ScoreNetwork, self).__init__()
        self.conv1 = nn.Conv2d(9, 192, kernel_size=7, stride=2, padding=3)
        self.invert = nn.Sequential(LayerNorm2d(192),
                                    nn.Conv2d(192, 192, kernel_size=3,
                                              stride=1, padding=1),
                                    nn.Conv2d(192, 768, kernel_size=1),
                                    nn.Conv2d(768, 192, kernel_size=1),
                                    nn.GELU())
        self.conv2 = nn.Conv2d(192, 8, kernel_size=7, stride=2, padding=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        short_cut = x
        x = self.invert(x)
        x = short_cut + x
        x = self.conv2(x)
        x = x.float()
        x = self.softmax(x)
        return x


class Mesorch(BaseTorchMethod):
    """
    Mesorch method for image manipulation detection.
    Uses both ConvNeXt and Segformer architectures with frequency domain features.
    """

    def __init__(
        self,
        seg_pretrain_path: Optional[str] = None,
        conv_pretrain: bool = False,
        device: str = "cpu"
    ):
        super().__init__()
        self.convnext = ConvNeXt(conv_pretrain)
        self.segformer = MixVisionTransformer(seg_pretrain_path)
        self.upsample = UpsampleConcatConv()
        self.low_dct = LowDctFrequencyExtractor()
        self.high_dct = HighDctFrequencyExtractor()

        self.inverse = nn.ModuleList([nn.Conv2d(96, 1, 1) for _ in range(4)]
                                     + [nn.Conv2d(64, 1, 1) for _ in range(4)])
        self.gate = ScoreNetwork()

        self.resize = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.resize_to_original = ResizeToOriginal(interpolation='bilinear')

        self.to_device(device)
        self.eval()

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Mesorch model.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted mask and gate outputs.
        """
        high_freq = self.high_dct(image)
        low_freq = self.low_dct(image)
        input_high = torch.concat([image, high_freq], dim=1)
        input_low = torch.concat([image, low_freq], dim=1)
        input_all = torch.concat([image, high_freq, low_freq], dim=1)

        _, outs1 = self.convnext(input_high)
        _, outs2 = self.segformer(input_low)

        inputs = outs1 + outs2
        x, features = self.upsample(inputs)
        gate_outputs = self.gate(input_all)

        reduced = torch.cat([self.inverse[i](features[i]) for i in range(8)], dim=1)
        pred_mask = torch.sum(gate_outputs * reduced, dim=1, keepdim=True)

        pred_mask = self.resize(pred_mask)
        pred_mask = pred_mask.float()
        mask_pred = torch.sigmoid(pred_mask)

        return mask_pred, gate_outputs

    def predict(self, image: torch.Tensor, original_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Runs Mesorch on an image.

        Args:
            image (torch.Tensor): Input image tensor.
            original_size (Optional[Tuple[int, int]]): Original size of the image (height, width).
                If provided, the output mask will be resized to this size.

        Returns:
            torch.Tensor: Predicted manipulation mask.
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Move image to the same device as the model
        if image.device != self.device:
            image = image.to(self.device)

        with torch.no_grad():
            mask_pred, _ = self.forward(image)

        mask_pred = mask_pred.squeeze(0).squeeze(0).cpu()

        # Resize to original dimensions if requested
        if original_size is not None:
            mask_pred = self.resize_to_original(mask_pred, original_size)

        return mask_pred

    def benchmark(self, image: torch.Tensor, original_size: Optional[Tuple[int, int]] = None) -> BenchmarkOutput:
        """
        Benchmarks the Mesorch method using the provided image.

        Args:
            image (torch.Tensor): Input image tensor.
            original_size (Optional[Tuple[int, int]]): Original size to resize the output to.

        Returns:
            BenchmarkOutput: Contains the heatmap and placeholder for mask and detection.
        """
        mask = self.predict(image, original_size)
        return {"heatmap": mask, "mask": None, "detection": None}
