from photoholmes.preprocessing.image import GetImageSize, ToNumpy
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

ela_preprocessing = PreProcessingPipeline(
    inputs=["image"],
    outputs_keys=["image", "image_size"],
    transforms=[
        GetImageSize(),
        ToNumpy(),
    ],
)
