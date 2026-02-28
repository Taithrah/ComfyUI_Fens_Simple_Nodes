import math
import os
from typing import Any, Dict

import torch
import yaml
from comfy.model_management import intermediate_device
from comfy_api.latest import io


class OptiEmptyLatent(io.ComfyNode):
    LATENT_CHANNELS: int = 4

    config_path = os.path.join(os.path.dirname(__file__), "model_config.yaml")
    with open(config_path, "r") as f:
        MODEL_CONFIG = yaml.safe_load(f)

    device = intermediate_device()

    @classmethod
    def define_schema(cls) -> io.Schema:
        alignment_options = list(cls.MODEL_CONFIG.keys())
        return io.Schema(
            node_id="OptiEmptyLatent",
            display_name="Optimal Empty Latent",
            category="Fens_Simple_Nodes/Latent",
            search_aliases=[
                "empty",
                "empty latent",
                "new latent",
                "create latent",
                "blank latent",
                "blank",
            ],
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
                    tooltip="TRUE: Automatically calculates best resolution for your aspect ratio. FALSE: Use your own resolution (WxH format).",
                ),
                io.Combo.Input(
                    "latent_alignment",
                    options=alignment_options,
                    default="SDXL (1024px)",
                    tooltip="Optimization preset for model type.",
                ),
                io.Int.Input(
                    "batch_size",
                    default=1,
                    min=1,
                    max=4096,
                    tooltip="Number of latent images in batch (VRAM usage increases with batch size).",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent", tooltip="Latent tensor"),
                io.Int.Output(display_name="width", tooltip="Width"),
                io.Int.Output(display_name="height", tooltip="Height"),
                io.Int.Output(
                    display_name="block_size",
                    tooltip="Block size used for the calculation.",
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
    ) -> tuple[int, int]:
        ideal_px = target_mp * 1024 * 1024
        raw_h = math.sqrt(ideal_px / ar)
        search_range = int(model_cfg.get("search_range", 10))
        align = cls._align

        min_ar = float(model_cfg.get("min_ar", 0.5))
        max_ar = float(model_cfg.get("max_ar", 3.75))

        best_score = float("inf")
        best_w = best_h = 0

        for delta in range(-search_range, search_range + 1):
            h_try = raw_h + delta * block
            w_try = ar * h_try
            w = align(w_try, block)
            h = align(h_try, block)

            if w < block or h < block:
                continue

            candidate_ar = w / h
            if candidate_ar < min_ar or candidate_ar > max_ar:
                continue

            mp = (w * h) / (1024 * 1024)
            mp_err_rel = abs(mp - target_mp) / target_mp
            ar_err_rel = abs(candidate_ar - ar) / ar

            mp_weight = 10.0
            ar_weight = 1.0
            score = (mp_weight * mp_err_rel) + (ar_weight * ar_err_rel)

            if abs(score - best_score) < 1e-9:
                if (w * h) > (best_w * best_h):
                    best_w, best_h = w, h
            elif score < best_score:
                best_score = score
                best_w, best_h = w, h

        if best_w == 0 or best_h == 0:
            raise ValueError(
                f"No valid resolution found for AR~{ar:.3f}, target {target_mp}MP, "
                f"block {block}. Try increasing search_range."
            )

        return best_w, best_h

    @classmethod
    def _make_latent(
        cls, w: int, h: int, bs: int, spacial_downscale_ratio: int
    ) -> dict[str, torch.Tensor]:
        shape = (
            bs,
            cls.LATENT_CHANNELS,
            h // spacial_downscale_ratio,
            w // spacial_downscale_ratio,
        )
        return {"samples": torch.zeros(shape, device=cls.device)}

    @classmethod
    def execute(
        cls,
        dimensions: str,
        invert: bool,
        optimization: bool,
        latent_alignment: str,
        batch_size: int,
    ) -> io.NodeOutput:
        cfg = cls.MODEL_CONFIG[latent_alignment]

        if not optimization:
            try:
                w, h = cls._parse_exact_dimensions(dimensions)
                if invert:
                    w, h = h, w
                w = cls._align(w, cfg["block_size"])
                h = cls._align(h, cfg["block_size"])

                latent = cls._make_latent(
                    w,
                    h,
                    batch_size,
                    cfg["spacial_downscale_ratio"],
                )
                details = (
                    f"Exact Resolution: {w}x{h} px\n"
                    f"Aspect Ratio: {w / h:.4f}\n"
                    f"Block Size: {cfg['block_size']}, VAE Scale: {cfg['spacial_downscale_ratio']}\n"
                    f"Model: {cfg.get('desc', latent_alignment)}"
                )
                return io.NodeOutput(latent, w, h, cfg["block_size"], details)
            except Exception as e:
                return io.NodeOutput(None, 0, 0, cfg["block_size"], f"Error: {e}")
        else:
            try:
                ar = cls.parse_ratio(dimensions)

                # Check aspect ratio bounds
                min_ar = float(cfg.get("min_ar", 0.5))
                max_ar = float(cfg.get("max_ar", 3.75))
                clamp_warning = ""
                if not (min_ar <= ar <= max_ar):
                    clamp_warning = (
                        f"⚠️ Dimensions {ar:.3f} are outside recommended range for {latent_alignment} "
                        f"({min_ar:.2f}-{max_ar:.2f}). Clamping for best results.\n"
                    )
                    ar = max(min_ar, min(ar, max_ar))

                w, h = cls._find_resolution(
                    ar, cfg["target_mp"], cfg["block_size"], cfg
                )
                if invert:
                    w, h = h, w
                latent = cls._make_latent(
                    w,
                    h,
                    batch_size,
                    cfg["spacial_downscale_ratio"],
                )
                details = (
                    f"{clamp_warning}Optimized Resolution: {w}x{h} px\n"
                    f"Aspect Ratio: {w / h:.4f} (requested: {dimensions})\n"
                    f"Target MP: {cfg['target_mp']}, Actual MP: {(w * h) / 1e6:.3f}\n"
                    f"Block Size: {cfg['block_size']}, VAE Scale: {cfg['spacial_downscale_ratio']}\n"
                    f"Model: {cfg.get('desc', latent_alignment)}"
                )
                return io.NodeOutput(latent, w, h, cfg["block_size"], details)
            except Exception as e:
                return io.NodeOutput(None, 0, 0, cfg["block_size"], f"Error: {e}")
