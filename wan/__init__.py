# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from . import configs, distributed, modules

# Lazy import pipelines only when accessed
def __getattr__(name):
    if name == "WanI2V":
        from .image2video import WanI2V
        return WanI2V
    if name == "WanS2V":
        from .speech2video import WanS2V
        return WanS2V
    if name == "WanT2V":
        from .text2video import WanT2V
        return WanT2V
    if name == "WanTI2V":
        from .textimage2video import WanTI2V
        return WanTI2V
    if name == "WanAnimate":
        from .animate import WanAnimate
        return WanAnimate

    raise AttributeError(f"module 'wan' has no attribute '{name}'")
