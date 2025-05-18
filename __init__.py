__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

from .nodes.FensTokenCounter import FensTokenCounter
from .nodes.OptiEmptyLatent import OptiEmptyLatent

NODE_CLASS_MAPPINGS = {
    "FensTokenCounter": FensTokenCounter,
    "OptiEmptyLatent": OptiEmptyLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FensTokenCounter": "Fens Token Counter",
    "OptiEmptyLatent": "Optimal Empty Latent",
}

WEB_DIRECTORY = "./web"
