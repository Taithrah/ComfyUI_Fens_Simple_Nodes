from .nodes.FensTokenCounter import FensTokenCounter
from .nodes.OptiEmptyLatent import OptiEmptyLatent
from .nodes.OptiEmptyLatentAdvanced import OptiEmptyLatentAdvanced

NODE_CLASS_MAPPINGS = {
    "FensTokenCounter": FensTokenCounter,
    "OptiEmptyLatent": OptiEmptyLatent,
    "OptiEmptyLatentAdvanced": OptiEmptyLatentAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FensTokenCounter": "Fens Token Counter",
    "OptiEmptyLatent": "Optimal Empty Latent",
    "OptiEmptyLatentAdvanced": "Optimal Empty Latent (Advanced)",
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
