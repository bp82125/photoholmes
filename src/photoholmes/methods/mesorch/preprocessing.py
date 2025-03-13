from photoholmes.preprocessing import (
    EnsureFloatTensor,
    ZeroOneRange,
    Resize,
    ToTensor,
    StoreOriginalSize,
    PreProcessingPipeline
)

mesorch_preprocessing = PreProcessingPipeline(
    inputs=["image"],
    outputs_keys=["image", "original_size"],
    transforms=[
        ToTensor(),
        ZeroOneRange(),
        StoreOriginalSize(),
        Resize(target_size=(512, 512)),
        EnsureFloatTensor() 
    ]
)
