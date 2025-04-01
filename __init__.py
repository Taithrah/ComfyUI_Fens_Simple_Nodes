__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

from .src.FensTokenCounter.FensTokenCounter import FensTokenCounter

NODE_CLASS_MAPPINGS = {
    "FensTokenCounter": FensTokenCounter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FensTokenCounter": "Fens Token Counter",
}

WEB_DIRECTORY = "./web"
