from photoholmes.preprocessing.image import GetImageSize, ToNumpy, RGBtoBGR
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

blocking_preprocessing = PreProcessingPipeline(
    inputs=["image"],
    outputs_keys=["image", "image_size"],
    transforms=[
        GetImageSize(),
        ToNumpy(),
    ],
)
