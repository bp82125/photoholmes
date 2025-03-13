from photoholmes.preprocessing.image import GetImageSize, ToNumpy
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

wavelet_preprocessing = PreProcessingPipeline(
    inputs=["image"],
    outputs_keys=["image", "image_size"],
    transforms=[
        GetImageSize(),
        ToNumpy(),
    ],
)
