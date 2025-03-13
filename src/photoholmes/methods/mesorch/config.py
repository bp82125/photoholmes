from pydantic import BaseModel, Field
from typing import Optional, Union, Literal


class MesorchArchConfig(BaseModel):
    """Configuration for Mesorch architecture."""
    seg_pretrain_path: Optional[str] = None
    conv_pretrain: bool = False

pretrained_arch = MesorchArchConfig(
    seg_pretrain_path=None,
    conv_pretrain=False
)


class MesorchConfig(BaseModel):
    """Configuration for Mesorch model."""
    arch: Union[MesorchArchConfig, Literal["pretrained"]] = "pretrained"
    weights: Optional[str] = Field(
        default="/home/nhat82125/photoholmes/weights/mesorch/mesorch.pth",
        description="Path to the weights file"
    )
    device: str = "cuda"


DEFAULT_CONFIG = MesorchConfig(
    arch="pretrained",
    weights="/home/nhat82125/photoholmes/weights/mesorch/mesorch.pth",
    device="cuda"
)
