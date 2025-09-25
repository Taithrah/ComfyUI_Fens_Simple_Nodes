import math
import os
from typing import Any, Dict

import torch
import yaml
from comfy.model_management import intermediate_device
from comfy_api.latest import io


class OptiEmptyLatent(io.ComfyNode):
    """
    Choose optimal WxH for a given aspect ratio & MP target.
    Supports exact resolution input when optimized resolution is disabled.
    """

    config_path = os.path.join(os.path.dirname(__file__), "model_config.yaml")
    with open(config_path, "r") as f:
        MODEL_CONFIG = yaml.safe_load(f)

    device = intermediate_device()

    @classmethod
    def define_schema(cls) -> io.Schema:
        preset_names = list(cls.MODEL_CONFIG.keys())
        alignment_options = preset_names + ["Custom"]
        return io.Schema(
            node_id="OptiEmptyLatent",
            display_name="OptiEmptyLatent",
            category="Fens_Simple_Nodes/Latent",
            description="Choose optimal WxH for a given aspect ratio & MP target. Supports exact resolution input when optimized resolution is disabled.",
            inputs=[
                io.String.Input(
                    "dimensions",
                    default="1:1",
                    tooltip="Formats: W:H (e.g. 16:9), WxH (e.g. 1280x720), or decimal (e.g. 1.777). Use WxH when 'Optimization' is FALSE.",
                ),
                io.Boolean.Input(
                    "invert",
                    default=False,
                    tooltip="Swap width and height (invert aspect ratio, e.g. 16:9 > 9:16).",
                ),
                io.Boolean.Input(
                    "optimization",
                    default=True,
                    tooltip="TRUE: Automatically calculates best resolution for your aspect ratio. FALSE: Use your own resolution (WxH format) - will be model spec.",
                ),
                io.Combo.Input(
                    "latent_alignment",
                    options=alignment_options,
                    default="SDXL (1024px)",
                    tooltip="Optimization preset for model type. Select 'Custom' to set your own block size and target MP.",
                ),
                io.Int.Input(
                    "batch_size",
                    default=1,
                    min=1,
                    max=4096,
                    tooltip="Number of latent images in batch (VRAM usage increases with batch size).",
                ),
                io.Int.Input(
                    "vae_scale_factor",
                    default=8,
                    min=1,
                    max=64,
                    tooltip="The VAE's total downsampling factor. (Only used when 'Custom' is selected.)",
                ),
                io.Float.Input(
                    "target_mp",
                    default=1.048576,
                    min=0.1,
                    max=16.0,
                    step=0.000001,
                    tooltip="Target megapixels. (Only used when 'Custom' is selected.)",
                ),
            ],
            outputs=[
                io.AnyType.Output(display_name="latent", tooltip="Latent tensor"),
                io.Int.Output(display_name="width", tooltip="Width"),
                io.Int.Output(display_name="height", tooltip="Height"),
                io.Int.Output(
                    display_name="vae_scale_factor", tooltip="VAE Scale Factor"
                ),
                io.String.Output(
                    display_name="details", tooltip="Details about the calculation"
                ),
            ],
        )

    @staticmethod
    def parse_ratio(dimensions: str) -> float:
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

    @staticmethod
    def _parse_exact_dimensions(dimensions: str) -> tuple[int, int]:
        """Parses 'WxH' or 'W:H' strings into integer tuples."""
        s = dimensions.strip().lower()
        if "x" in s:
            parts = s.split("x", 1)
        elif ":" in s:
            parts = s.split(":", 1)
        else:
            raise ValueError("Use WxH or W:H format for exact resolution")

        if len(parts) != 2:
            raise ValueError("Invalid format. Use WxH or W:H")

        w, h = map(int, map(str.strip, parts))
        return w, h

    @staticmethod
    def _align(value: float, block: int) -> int:
        return max(block, int(round(value / block)) * block)

    @classmethod
    def _find_resolution(
        cls, ar: float, target_mp: float, block: int, model_cfg: Dict[str, Any]
    ):
        ideal_px = target_mp * 1e6
        raw_h = math.sqrt(ideal_px / ar)
        search_range = int(model_cfg.get("search_range", 10 if block >= 32 else 5))
        rel_ar_tol = float(model_cfg.get("rel_ar_tol", 0.001))
        align = cls._align

        best_exact = None
        for delta in range(-search_range, search_range + 1):
            h_try = raw_h + delta * block
            w_try = ar * h_try
            w = align(w_try, block)
            h = align(h_try, block)
            if w < block or h < block:
                continue
            actual_ar = w / h
            ar_err_rel = abs(actual_ar / ar - 1.0)
            if ar_err_rel <= rel_ar_tol:
                mp = (w * h) / 1e6
                mp_err = abs(mp - target_mp)
                score = (mp_err, ar_err_rel, -w * h)
                if best_exact is None or score < best_exact[0]:
                    best_exact = (score, w, h)
        if best_exact:
            return best_exact[1], best_exact[2]

        best = None
        for delta in range(-search_range, search_range + 1):
            h_try = raw_h + delta * block
            w_try = ar * h_try
            w = align(w_try, block)
            h = align(h_try, block)
            if w < block or h < block:
                continue
            mp = (w * h) / 1e6
            mp_err = abs(mp - target_mp)
            ar_err = abs((w / h) - ar)
            score = (mp_err, ar_err, -w * h)
            if best is None or score < best[0]:
                best = (score, w, h)
        if not best:
            raise ValueError("No valid resolution found")
        return best[1], best[2]

    @classmethod
    def _make_latent(
        cls, w: int, h: int, bs: int, channels: int = 4, vae_scale_factor: int = 8
    ):
        shape = (bs, channels, h // vae_scale_factor, w // vae_scale_factor)
        return {"samples": torch.zeros(shape, device=cls.device)}

    @classmethod
    def _execute_exact(
        cls, dimensions: str, invert: bool, batch_size: int, cfg: Dict[str, Any]
    ) -> io.NodeOutput:
        """Handles exact resolution mode."""
        scale_factor = cfg["vae_scale_factor"]
        try:
            w_val, h_val = cls._parse_exact_dimensions(dimensions)
            if invert:
                w_val, h_val = h_val, w_val

            w = cls._align(w_val, scale_factor)
            h = cls._align(h_val, scale_factor)

            latent = cls._make_latent(
                w, h, batch_size, cfg.get("channels", 4), scale_factor
            )
            actual_ar = round(w / h, 4)
            details = (
                f"Exact Resolution: {w}x{h} px\n"
                f"Aspect Ratio: {actual_ar}\n"
                f"VAE Scale Factor: {scale_factor}, Channels: {cfg.get('channels', 4)}\n"
                f"Model: {cfg.get('desc', 'Custom')}"
            )
            return io.NodeOutput(latent, w, h, scale_factor, details)
        except Exception as e:
            error_msg = f"⚠️ Exact resolution error: {e}"
            print(error_msg)
            return io.NodeOutput(None, 0, 0, scale_factor, error_msg)

    @classmethod
    def _execute_optimized(
        cls,
        dimensions: str,
        invert: bool,
        batch_size: int,
        cfg: Dict[str, Any],
        latent_alignment: str,
    ) -> io.NodeOutput:
        """Handles optimized resolution mode."""
        scale_factor = cfg["vae_scale_factor"]
        try:
            ar = cls.parse_ratio(dimensions)
        except ValueError as e:
            error_msg = f"⚠️ Invalid dimensions: {e}"
            print(error_msg)
            return io.NodeOutput(None, 0, 0, scale_factor, error_msg)

        min_ar, max_ar = cfg.get("min_ar", 0.4), cfg.get("max_ar", 2.5)

        clamp_warning = ""
        if not (min_ar <= ar <= max_ar):
            clamp_warning = (
                f"⚠️ Dimensions {ar:.3f} are outside recommended range for {latent_alignment} "
                f"({min_ar:.2f}-{max_ar:.2f}). Clamping for best results."
            )
            ar = max(min_ar, min(ar, max_ar))

        try:
            w, h = cls._find_resolution(
                ar, cfg["target_mp"], scale_factor, model_cfg=cfg
            )
            if invert:
                w, h = h, w
        except ValueError as e:
            error_msg = f"⚠️ Resolution error: {e}"
            print(error_msg)
            return io.NodeOutput(None, 0, 0, scale_factor, error_msg)

        latent = cls._make_latent(
            w, h, batch_size, cfg.get("channels", 4), scale_factor
        )
        actual_mp = (w * h) / 1e6
        actual_ar = round(w / h, 4)
        details = (
            f"Optimized Resolution: {w}x{h} px\n"
            f"Aspect Ratio: {actual_ar} (requested: {dimensions})\n"
            f"Target MP: {cfg['target_mp']}, Actual MP: {actual_mp:.3f}\n"
            f"VAE Scale Factor: {scale_factor}, Channels: {cfg.get('channels', 4)}\n"
            f"Model: {cfg.get('desc', latent_alignment)}"
        )
        if clamp_warning:
            details = clamp_warning + "\n" + details

        return io.NodeOutput(latent, w, h, scale_factor, details)

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        # Deconstruct kwargs
        dimensions = kwargs["dimensions"]
        invert = kwargs["invert"]
        optimization = kwargs["optimization"]
        latent_alignment = kwargs["latent_alignment"]
        batch_size = kwargs["batch_size"]

        # Get model/custom configuration
        if latent_alignment == "Custom":
            cfg = {
                "vae_scale_factor": kwargs["vae_scale_factor"],
                "target_mp": kwargs["target_mp"],
                "channels": 4,
                "desc": f"Custom (VAE Scale: {kwargs['vae_scale_factor']}, Target: {kwargs['target_mp']}MP)",
            }
        else:
            cfg = cls.MODEL_CONFIG[latent_alignment]

        if not optimization:
            return cls._execute_exact(dimensions, invert, batch_size, cfg)
        else:
            return cls._execute_optimized(
                dimensions, invert, batch_size, cfg, latent_alignment
            )
