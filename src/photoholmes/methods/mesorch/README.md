# Mesoscopic Insights: Orchestrating Multi-scale & Hybrid Architecture for Image Manipulation Localization

This is the implementation of the method by Zhu et al that can be found [here](https://arxiv.org/pdf/2412.13753).

The code contained in this library was derived from [the original implementation](https://github.com/scu-zjz/Mesorch), making only minor changes to fit the PhotoHolmes library structure.

This is a deep learning based method, the checkpoints can be found [here](https://drive.google.com/drive/folders/1jwYv-S3HAZqzz0YxM9bJynBiPv-O9-6x). The authors provide two checkpoints files: `mesorch_p-118.pth` (pruned version) and `mesorch-98.pth` (original version). This implementation uses the original checkpoints (`mesorch-98.pth`). We last checked this information on May 15 2025, please refer to the authors of the original paper if the weights cannot be found.

## Description

The Mesorch Framework employs a novel multi-scale parallel architecture to effectively process input images, setting a new benchmark in image manipulation localization. By leveraging distinct frequency components and feature hierarchies, it captures both local manipulations and global inconsistencies. Its adaptive weighting mechanism ensures precise and comprehensive results, making it a robust solution for image manipulation localization tasks.
## Full overview

The Mesorch (Mesoscopic-Orchestration) framework addresses image manipulation localization (IML) by orchestrating microscopic and macroscopic features to capture artifacts at the mesoscopic level. It leverages Discrete Cosine Transform (DCT) to extract high-frequency (microscopic) and low-frequency (macroscopic) features, enhancing input images for processing. A hybrid encoder combines CNNs (ConvNeXt) for local feature extraction with Transformers (SegFormer) for global semantic understanding, operating in parallel to maximize the strengths of both architectures. The framework employs a multi-scale decoder to generate predictions at four scales, utilizing an adaptive weighting module to dynamically prioritize significant scales for improved accuracy.

A pruning method reduces parameters and FLOPs by eliminating less significant scales post-convergence, ensuring computational efficiency. Mesorch produces a pixel-level binary mask highlighting tampered regions, achieving an average F1 score of 0.6771 (0.6762 with pruning), outperforming models like TruFor, CAT-Net, and MVSS-Net in F1, AUC, IOU, and robustness under perturbations such as Gaussian noise, blur, and JPEG compression. With 62.235M parameters and 64.821G FLOPs (pruned), it is highly efficient. The model was trained on the Protocol-CAT dataset, including CASIA v2, FantasticReality, and tampered COCO/RAISE, over 150 epochs using the AdamW optimizer, a cosine learning rate schedule (1e-4 to 5e-7), and four NVIDIA 4090 GPUs.

## Usage

```python
import torch

from photoholmes.methods.mesorch import Mesorch, mesorch_preprocessing

# Read an image
from photoholmes.utils.image import read_image
path_to_image = "path_to_image"
image = read_image(path_to_image)

# Assign the image to a dictionary and preprocess the image
image_data = {"image": image}
input = mesorch_preprocessing(**image_data)

# Declare the method and use the .to_device if you want to run it on cuda or mps instead of cpu
path_to_checkpoint = "path_to_checkpoint"
model = Mesorch(seg_pretrain_path=None, conv_pretrain=False)
checkpoint = torch.load(
    checkpoint_path, map_location=device, weights_only=False)

model.load_state_dict(checkpoint['model'])

device = "cpu"
model.to_device(device)

# Use predict to get the final result
output = method.predict(**input)
```

## Citation
``` bibtex
@inproceedings{zhu2025mesoscopic,
  title={Mesoscopic insights: orchestrating multi-scale \& hybrid architecture for image manipulation localization},
  author={Zhu, Xuekang and Ma, Xiaochen and Su, Lei and Jiang, Zhuohang and Du, Bo and Wang, Xiwen and Lei, Zeyu and Feng, Wentao and Pun, Chi-Man and Zhou, Ji-Zhe},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={10},
  pages={11022--11030},
  year={2025}
}
```