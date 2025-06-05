import math

import torch
from comfy.model_management import intermediate_device


class OptiEmptyLatent:
    """ComfyUI node: choose closest WxH for a given aspect ratio & MP target."""

    MODEL_CONFIG = {
        # Standard
        "SD1 (512px)": {"block": 8, "target_mp": 0.26},
        "SD2 (768px)": {"block": 8, "target_mp": 0.59},
        "SDXL (1024px)": {"block": 64, "target_mp": 1.05},
        # Experimental
        "SDXL EXPT-A": {"block": 16, "target_mp": 1.05},
        "SDXL EXPT-B": {"block": 8, "target_mp": 1.05},
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ratio": (
                    "STRING",
                    {
                        "default": "1:1",
                        "tooltip": "The ratio of the latent images. Can be W:H, WxH, or decimal.",
                    },
                ),
                "swap_ratio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Yes",
                        "label_off": "No",
                        "tooltip": "Swap W x H",
                    },
                ),
                "latent_alignment": (
                    list(cls.MODEL_CONFIG.keys()),
                    {"default": "SDXL (1024px)", "tooltip": "Optimization Preset."},
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "The number of latent images in the batch.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    CATEGORY = "Fens_Simple_Nodes/Latent"
    FUNCTION = "opti_generate"

    @staticmethod
    def parse_ratio(ratio: str) -> float:
        """
        Parse an aspect‐ratio string into a float (width/height).

        Supported formats:
          - "W:H"   (e.g. "16:9")
          - "WxH"   (e.g. "1920x1080" or "1920X1080")
          - decimal (e.g. "1.777")

        Raises:
          ValueError if the format is unrecognized or height == 0.
        """
        s = ratio.strip()
        # W:H
        if ":" in s:
            parts = s.split(":", 1)
        # WxH (case‐insensitive)
        elif "x" in s.lower():
            parts = s.lower().split("x", 1)
        else:
            # try bare decimal
            try:
                return float(s)
            except ValueError:
                raise ValueError(f"Invalid ratio format: '{ratio}'")
        # parse and compute
        w, h = map(float, parts)
        if h == 0:
            raise ValueError(f"Height cannot be zero in ratio '{ratio}'")
        return w / h

    def __init__(self):
        self.device = intermediate_device()

    def opti_generate(
        self,
        ratio: str,
        latent_alignment: str,
        batch_size: int = 1,
        swap_ratio: bool = False,
    ):
        # user‐friendly catch for bad ratio input & ratio limits
        try:
            ar = self.parse_ratio(ratio)
        except ValueError as e:
            error_msg = f"⚠️ Invalid ratio: {str(e)}"
            print(error_msg)
            return {"ui": {"text": [error_msg]}, "result": ({"samples": None}, 0, 0)}

        if swap_ratio:
            ar = 1.0 / ar

        try:
            cfg = self.MODEL_CONFIG[latent_alignment]
            w, h = self._find_resolution(ar, cfg["target_mp"], cfg["block"])
        except ValueError as e:
            error_msg = f"⚠️ Resolution error: {str(e)}"
            print(error_msg)
            return {"ui": {"text": [error_msg]}, "result": ({"samples": None}, 0, 0)}

        latent = self._make_latent(w, h, batch_size)
        return {"ui": {"text": [f"Optimal: {w}×{h}"]}, "result": (latent, w, h)}

    def _error_result(self, message: str):
        print(f"OptiEmptyLatent Error: {message}")
        return {"ui": {"text": [message]}, "result": ({"samples": None}, 0, 0)}

    def _success_result(self, latent, width: int, height: int):
        return {
            "ui": {"text": [f"Optimal: {width}×{height}"]},
            "result": (latent, width, height),
        }

    def _align(self, value: float, block: int) -> int:
        """Round to nearest multiple of block, min block."""
        val = max(block, int(round(value / block)) * block)
        return val

    def _find_resolution(self, ar: float, target_mp: float, block: int):
        ideal_px = target_mp * 1e6
        raw_h = math.sqrt(ideal_px / ar)
        best = None
        for delta in range(-2, 3):
            h_try = raw_h + delta * block
            w_try = ar * h_try
            w = self._align(w_try, block)
            h = self._align(h_try, block)
            if w < block or h < block:
                continue
            mp = (w * h) / 1e6
            diff = abs(mp - target_mp)
            ar_err = abs((w / h) - ar)
            score = (diff, ar_err, -w * h)
            if best is None or score < best[0]:
                best = (score, w, h)
        if not best:
            raise ValueError("No valid resolution found")
        return best[1], best[2]

    def _make_latent(self, w: int, h: int, bs: int):
        # latent dims are 1/8 of image
        shape = (bs, 4, h // 8, w // 8)
        return {"samples": torch.zeros(shape, device=self.device)}
