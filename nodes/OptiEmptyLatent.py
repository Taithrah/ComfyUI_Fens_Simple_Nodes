import math

import torch
from comfy.model_management import intermediate_device


class OptiEmptyLatent:
    """
    ComfyUI node: Choose optimal WxH for a given aspect ratio & MP target.

    Supports SD1, SD2, SDXL, and other SD-like architectures. Model configs include
    latent block size, target megapixels, channel count, and recommended aspect ratio ranges.

    Model config options:
    block: Minimum multiple for width/height (latent spatial block size for model, e.g. 8 or 64).
    target_mp: Target image area in megapixels (width * height / 1e6) for optimal generation.
    channels: Number of latent channels (usually 4 for Stable Diffusion models).
    min_ar: Minimum recommended aspect ratio (width/height) for this model.
    max_ar: Maximum recommended aspect ratio (width/height) for this model.
    """

    MODEL_CONFIG = {
        # Standard Stable Diffusion versions
        "SD1 (512px)": {
            "block": 8,
            "target_mp": 0.262144,
            "channels": 4,
            "min_ar": 0.5,
            "max_ar": 3.5,
            "desc": "SD1.x, 512x512, 4-channel latent, block 8",
        },
        "SD2 (768px)": {
            "block": 8,
            "target_mp": 0.589824,
            "channels": 4,
            "min_ar": 0.5,
            "max_ar": 3.5,
            "desc": "SD2.x, 768x768, 4-channel latent, block 8",
        },
        "SDXL (1024px)": {
            "block": 64,
            "target_mp": 1.048576,
            "channels": 4,
            "min_ar": 0.5,
            "max_ar": 3.5,
            "desc": "SDXL, 1024x1024, 4-channel latent, block 64",
        },
        # Experimental variants
        "SDXL (Div-32)": {
            "block": 32,
            "target_mp": 1.048576,
            "channels": 4,
            "min_ar": 0.5,
            "max_ar": 3.5,
            "desc": "SDXL experimental, block 16",
        },
        "SDXL (Div-16)": {
            "block": 16,
            "target_mp": 1.048576,
            "channels": 4,
            "min_ar": 0.5,
            "max_ar": 3.5,
            "desc": "SDXL experimental, block 16",
        },
        "SDXL (Div-8)": {
            "block": 8,
            "target_mp": 1.048576,
            "channels": 4,
            "min_ar": 0.5,
            "max_ar": 3.5,
            "desc": "SDXL experimental, block 8",
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        """
        Returns:
            Dictionary of input fields and tooltips for ComfyUI.
        """
        return {
            "required": {
                "ratio": (
                    "STRING",
                    {
                        "default": "1:1",
                        "tooltip": (
                            "Aspect ratio of latent images. Formats: W:H (e.g. 16:9), WxH (e.g. 1280x720), or decimal (e.g. 1.777)."
                        ),
                    },
                ),
                "swap_ratio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Yes",
                        "label_off": "No",
                        "tooltip": "Swap width and height (invert aspect ratio, e.g. 16:9 > 9:16).",
                    },
                ),
                "latent_alignment": (
                    list(cls.MODEL_CONFIG.keys()),
                    {
                        "default": "SDXL (1024px)",
                        "tooltip": (
                            "Optimization preset for model type. See documentation for model-specific constraints."
                        ),
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": (
                            "Number of latent images in batch (VRAM usage increases with batch size)."
                        ),
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
        if ":" in s:
            parts = s.split(":", 1)
        elif "x" in s.lower():
            parts = s.lower().split("x", 1)
        else:
            try:
                return float(s)
            except ValueError:
                raise ValueError(f"Invalid ratio format: '{ratio}'")
        w, h = map(float, parts)
        if h == 0:
            raise ValueError(f"Height cannot be zero in ratio '{ratio}'")
        return w / h

    def __init__(self):
        """
        Initialize node and set the device for latent generation.
        """
        self.device = intermediate_device()

    def opti_generate(
        self,
        ratio: str,
        latent_alignment: str,
        batch_size: int = 1,
        swap_ratio: bool = False,
    ):
        """
        Main node function: calculates optimal latent shape for requested ratio and model.
        Handles aspect ratio clamping, user feedback, and latent generation.
        Returns detailed UI output for user clarity.
        """
        # Parse aspect ratio
        try:
            ar = self.parse_ratio(ratio)
        except ValueError as e:
            error_msg = f"⚠️ Invalid ratio: {str(e)}"
            print(error_msg)
            return self._error_result(error_msg)

        if swap_ratio:
            ar = 1.0 / ar

        # Get model config and clamp aspect ratio if necessary
        cfg = self.MODEL_CONFIG[latent_alignment]
        min_ar, max_ar = cfg.get("min_ar", 0.4), cfg.get("max_ar", 2.5)

        # Swap min/max bounds if ratio was swapped
        if swap_ratio:
            min_ar, max_ar = 1.0 / max_ar, 1.0 / min_ar

        clamp_warning = ""
        if not (min_ar <= ar <= max_ar):
            clamp_warning = (
                f"⚠️ Ratio {ar:.3f} is outside recommended range for {latent_alignment} "
                f"({min_ar}–{max_ar}). Clamping for best results."
            )
            ar = max(min_ar, min(ar, max_ar))

        try:
            w, h = self._find_resolution(ar, cfg["target_mp"], cfg["block"])
        except ValueError as e:
            error_msg = f"⚠️ Resolution error: {str(e)}"
            print(error_msg)
            return self._error_result(error_msg)

        latent = self._make_latent(w, h, batch_size, cfg.get("channels", 4))

        actual_mp = (w * h) / 1e6
        actual_ar = round(w / h, 4)
        details = (
            f"Optimal: {w}×{h} px\n"
            f"Aspect Ratio: {actual_ar} (requested: {ratio})\n"
            f"Target MP: {cfg['target_mp']}, Actual MP: {actual_mp:.3f}\n"
            f"Block Size: {cfg['block']}, Channels: {cfg.get('channels', 4)}\n"
            f"Model: {cfg.get('desc', latent_alignment)}"
        )
        if clamp_warning:
            details = clamp_warning + "\n" + details

        return self._success_result(latent, w, h, details)

    def _error_result(self, message: str):
        print(f"OptiEmptyLatent Error: {message}")
        return {"ui": {"text": [message]}, "result": ({"samples": None}, 0, 0)}

    def _success_result(self, latent, width: int, height: int, details: str):
        return {
            "ui": {"text": [details]},
            "result": (latent, width, height),
        }

    def _align(self, value: float, block: int) -> int:
        """
        Round to nearest multiple of block size, min block.
        """
        val = max(block, int(round(value / block)) * block)
        return val

    def _find_resolution(self, ar: float, target_mp: float, block: int):
        """
        Find optimal width and height (multiples of block) for given aspect ratio and megapixel target.
        """
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

    def _make_latent(self, w: int, h: int, bs: int, channels: int = 4):
        """
        Create a latent tensor of shape (batch_size, channels, h//8, w//8).
        """
        # latent dims are 1/8 of image
        shape = (bs, channels, h // 8, w // 8)
        return {"samples": torch.zeros(shape, device=self.device)}
