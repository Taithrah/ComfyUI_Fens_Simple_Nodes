import math
import os

import torch
import yaml
from comfy.model_management import intermediate_device


class OptiEmptyLatent:
    """
    ComfyUI node: Choose optimal WxH for a given aspect ratio & MP target.
    Supports exact resolution input when optimized resolution is disabled.
    """

    config_path = os.path.join(os.path.dirname(__file__), "model_config.yaml")
    with open(config_path, "r") as f:
        MODEL_CONFIG = yaml.safe_load(f)

    @classmethod
    def INPUT_TYPES(cls):
        """
        Returns:
            Dictionary of input fields and tooltips for ComfyUI.
        """
        preset_names = list(cls.MODEL_CONFIG.keys())
        alignment_options = preset_names + ["Custom"]

        return {
            "required": {
                "dimensions": (
                    "STRING",
                    {
                        "default": "1:1",
                        "tooltip": (
                            "Formats: W:H (e.g. 16:9), WxH (e.g. 1280x720), or decimal (e.g. 1.777).\n"
                            "Use WxH when 'Optimization' is FALSE."
                        ),
                    },
                ),
                "invert": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "TRUE",
                        "label_off": "FALSE",
                        "tooltip": "Swap width and height (invert aspect ratio, e.g. 16:9 > 9:16).",
                    },
                ),
                "optimization": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "TRUE",
                        "label_off": "FALSE",
                        "tooltip": (
                            "TRUE: Automatically calculates best resolution for your aspect ratio.\n"
                            "FALSE: Use your own resolution (WxH format) - will be model spec."
                        ),
                    },
                ),
                "latent_alignment": (
                    alignment_options,
                    {
                        "default": "SDXL (1024px)",
                        "tooltip": "Optimization preset for model type. Select 'Custom' to set your own block size and target MP.",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "Number of latent images in batch (VRAM usage increases with batch size).",
                    },
                ),
                # I think there is a limitation with conditional options, so added tooltip clarification
                "block_size": (
                    "INT",
                    {
                        "default": 64,
                        "min": 8,
                        "max": 128,
                        "step": 8,
                        "tooltip": "Block size (multiple for width/height). (Only used when 'Custom' is selected.)",
                    },
                ),
                "target_mp": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": "Target megapixels. (Only used when 'Custom' is selected.)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height", "block_size")
    CATEGORY = "Fens_Simple_Nodes/Latent"
    FUNCTION = "opti_generate"

    def __init__(self):
        """
        Initialize node and set the device for latent generation.
        """
        self.device = intermediate_device()

    @staticmethod
    def parse_ratio(dimensions: str) -> float:
        """
        Parse an aspect-ratio string into a float (width/height).
        Supported formats: "W:H", "WxH", decimal.
        """
        s = dimensions.strip()
        if ":" in s:
            parts = s.split(":", 1)
        elif "x" in s.lower():
            parts = s.lower().split("x", 1)
        else:
            try:
                return float(s)
            except ValueError:
                raise ValueError(f"Invalid dimensions format: '{dimensions}'")
        w, h = map(float, parts)
        if h == 0:
            raise ValueError(f"Height cannot be zero in dimensions '{dimensions}'")
        return w / h

    def opti_generate(
        self,
        dimensions: str,
        latent_alignment: str,
        batch_size: int = 1,
        invert: bool = False,
        optimization: bool = True,
        block_size: int = 64,  # Ignored unless custom is selected
        target_mp: float = 1.0,  # Ignored unless custom is selected
    ):
        """
        Main node function: calculates optimal latent shape for requested dimensions and model.
        Handles aspect ratio clamping, user feedback, and latent generation.
        Returns detailed UI output for user clarity.
        """
        # Get model config or use custom values
        if latent_alignment == "Custom":
            cfg = {
                "block": block_size,
                "target_mp": target_mp,
                "min_ar": 0.5,
                "max_ar": 3.5,
                "channels": 4,
                "desc": f"Custom (Block: {block_size}, Target: {target_mp}MP)",
            }
        else:
            cfg = self.MODEL_CONFIG[latent_alignment]

        block = cfg["block"]

        # Handle exact resolution mode (when optimized is disabled)
        if not optimization:
            try:
                if "x" in dimensions.lower():
                    parts = dimensions.lower().split("x", 1)
                elif ":" in dimensions:
                    parts = dimensions.split(":", 1)
                else:
                    raise ValueError("Must use WxH or W:H format for exact resolution")
                if len(parts) != 2:
                    raise ValueError("Invalid format: expected WxH or W:H")
                w_val = int(parts[0].strip())
                h_val = int(parts[1].strip())
                if invert:
                    w_val, h_val = h_val, w_val
                w = self._align(w_val, block)
                h = self._align(h_val, block)
                latent = self._make_latent(w, h, batch_size, cfg.get("channels", 4))
                actual_ar = round(w / h, 4)
                details = (
                    f"Exact Resolution: {w}x{h} px\n"
                    f"Aspect Ratio: {actual_ar}\n"
                    f"Block Size: {block}, Channels: {cfg.get('channels', 4)}\n"
                    f"Model: {cfg.get('desc', latent_alignment)}"
                )
                return self._success_result(latent, w, h, block, details)
            except Exception as e:
                error_msg = f"⚠️ Exact resolution error: {str(e)}"
                print(error_msg)
                return self._error_result(error_msg, block)

        # Optimized resolution mode
        try:
            ar = self.parse_ratio(dimensions)
        except ValueError as e:
            error_msg = f"⚠️ Invalid dimensions: {str(e)}"
            print(error_msg)
            return self._error_result(error_msg, block)

        if invert:
            ar = 1.0 / ar

        min_ar, max_ar = cfg.get("min_ar", 0.4), cfg.get("max_ar", 2.5)
        if invert:
            min_ar, max_ar = 1.0 / max_ar, 1.0 / min_ar

        clamp_warning = ""
        if not (min_ar <= ar <= max_ar):
            clamp_warning = (
                f"⚠️ Dimensions {ar:.3f} are outside recommended range for {latent_alignment} "
                f"({min_ar:.2f}-{max_ar:.2f}). Clamping for best results."
            )
            ar = max(min_ar, min(ar, max_ar))

        try:
            w, h = self._find_resolution(ar, cfg["target_mp"], block)
        except ValueError as e:
            error_msg = f"⚠️ Resolution error: {str(e)}"
            print(error_msg)
            return self._error_result(error_msg, block)

        latent = self._make_latent(w, h, batch_size, cfg.get("channels", 4))
        actual_mp = (w * h) / 1e6
        actual_ar = round(w / h, 4)
        details = (
            f"Optimized Resolution: {w}x{h} px\n"
            f"Aspect Ratio: {actual_ar} (requested: {dimensions})\n"
            f"Target MP: {cfg['target_mp']}, Actual MP: {actual_mp:.3f}\n"
            f"Block Size: {block}, Channels: {cfg.get('channels', 4)}\n"
            f"Model: {cfg.get('desc', latent_alignment)}"
        )
        if clamp_warning:
            details = clamp_warning + "\n" + details

        return self._success_result(latent, w, h, block, details)

    def _error_result(self, message: str, block_size: int = 8):
        print(f"OptiEmptyLatent Error: {message}")
        return {
            "ui": {"text": [message]},
            "result": ({"samples": None}, 0, 0, block_size),
        }

    def _success_result(
        self, latent, width: int, height: int, block_size: int, details: str
    ):
        return {
            "ui": {"text": [details]},
            "result": (latent, width, height, block_size),
        }

    def _align(self, value: float, block: int) -> int:
        """
        Round to nearest multiple of block size, min block.
        """
        return max(block, int(round(value / block)) * block)

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
