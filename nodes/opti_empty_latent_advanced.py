from __future__ import annotations

import os

import yaml
from comfy_api.latest import io
from typing_extensions import override

from .latent_common import (
    create_latent_for_exact,
    create_latent_for_optimized,
    resolve_cfg,
)


class OptiEmptyLatentAdvanced(io.ComfyNode):
    """
    Node to create an empty latent tensor with optimal or exact resolution for a given aspect ratio or WxH, with advanced customization.
    Integrates tightly with ComfyUI V3 node API and provides UI-friendly output.
    """

    config_path = os.path.join(os.path.dirname(__file__), "model_config.yaml")
    with open(config_path, encoding="utf-8") as f:
        MODEL_CONFIG = yaml.safe_load(f)

    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        preset_names = list(cls.MODEL_CONFIG.keys())
        alignment_options = preset_names + ["Custom"]
        return io.Schema(
            node_id="OptiEmptyLatentAdvanced",
            display_name="Optimal Empty Latent (Advanced)",
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
                    display_name="Dimensions",
                    default="1:1",
                    tooltip="Formats: W:H (e.g. 16:9), WxH (e.g. 1280x720), or decimal (e.g. 1.777). Use WxH when 'Optimization' is FALSE.",
                ),
                io.Combo.Input(
                    "latent_alignment",
                    display_name="Latent Alignment",
                    options=alignment_options,
                    default="SDXL (1024px)",
                    tooltip="Optimization preset for model type. Select 'Custom' to set your own parameters.",
                ),
                io.Boolean.Input(
                    "optimization",
                    display_name="Optimization",
                    default=True,
                    tooltip="TRUE: Automatically calculates best resolution for your aspect ratio. FALSE: Use your own resolution (WxH format).",
                ),
                io.Boolean.Input(
                    "invert",
                    display_name="Invert",
                    default=False,
                    tooltip="Swap width and height (invert aspect ratio, e.g. 16:9 > 9:16).",
                ),
                io.Int.Input(
                    "batch_size",
                    display_name="Batch Size",
                    display_mode=io.NumberDisplay.slider,
                    default=1,
                    min=1,
                    max=4096,
                    tooltip="Number of latent images in batch (VRAM usage increases with batch size).",
                ),
                io.Int.Input(
                    "block_size",
                    display_name="Block Size",
                    display_mode=io.NumberDisplay.slider,
                    default=cls.MODEL_CONFIG["Custom"]["block_size"],
                    min=8,
                    max=64,
                    step=8,
                    advanced=True,
                    tooltip="Pixel dimension alignment constraint. (Only used when 'Custom' is selected.)",
                ),
                io.Int.Input(
                    "spacial_downscale_ratio",
                    display_name="Spacial Downscale Ratio",
                    display_mode=io.NumberDisplay.slider,
                    default=cls.MODEL_CONFIG["Custom"]["spacial_downscale_ratio"],
                    min=8,
                    max=64,
                    step=2,
                    advanced=True,
                    tooltip="The VAE's total downsampling factor. (Only used when 'Custom' is selected.)",
                ),
                io.Float.Input(
                    "target_mp",
                    display_name="Target MP",
                    display_mode=io.NumberDisplay.slider,
                    default=cls.MODEL_CONFIG["Custom"]["target_mp"],
                    min=0.05,
                    max=32.0,
                    step=0.001,
                    advanced=True,
                    tooltip="Target megapixels. (Only used when 'Custom' is selected.)",
                ),
                io.Int.Input(
                    "search_range",
                    display_name="Search Range",
                    display_mode=io.NumberDisplay.slider,
                    default=cls.MODEL_CONFIG["Custom"]["search_range"],
                    min=1,
                    max=100,
                    advanced=True,
                    tooltip="Search range for optimization. Higher values search more possibilities. (Only used when 'Custom' is selected.)",
                ),
                io.Int.Input(
                    "channels",
                    display_name="Channels",
                    display_mode=io.NumberDisplay.slider,
                    default=cls.MODEL_CONFIG["Custom"].get("channels", 4),
                    min=1,
                    max=128,
                    advanced=True,
                    tooltip="Latent channel count (e.g. 4 for SD1/SDXL, 16 for FLUX/Cosmos-Predict2-family). (Only used when 'Custom' is selected.)",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    "latent", display_name="Latent", tooltip="Latent tensor"
                ),
                io.Int.Output("width", display_name="Width", tooltip="Width"),
                io.Int.Output("height", display_name="Height", tooltip="Height"),
                io.Int.Output(
                    "block_size",
                    display_name="Block Size",
                    tooltip="Block size used for the calculation.",
                ),
                io.String.Output(
                    "details",
                    display_name="Details",
                    tooltip="Details about the calculation",
                ),
            ],
            is_experimental=False,
        )

    @classmethod
    @override
    def execute(
        cls,
        dimensions: str,
        invert: bool,
        optimization: bool,
        latent_alignment: str,
        batch_size: int,
        block_size: int,
        spacial_downscale_ratio: int,
        target_mp: float,
        search_range: int,
        channels: int,
    ) -> io.NodeOutput:
        try:
            custom_overrides = None
            if latent_alignment == "Custom":
                custom_overrides = {
                    "block_size": block_size,
                    "spacial_downscale_ratio": spacial_downscale_ratio,
                    "target_mp": target_mp,
                    "search_range": search_range,
                    "channels": channels,
                }
            cfg = resolve_cfg(cls.MODEL_CONFIG, latent_alignment, custom_overrides)
        except ValueError as e:
            msg = f"Error: {e}"
            return io.NodeOutput(None, 0, 0, 0, msg)
        if not optimization:
            try:
                latent, w, h, details = create_latent_for_exact(
                    dimensions, invert, cfg, batch_size
                )
                # Optionally, provide a UI preview for details (uncomment if desired)
                # preview = ui.PreviewText(details)
                return io.NodeOutput(latent, w, h, cfg["block_size"], details)
            except (ValueError, TypeError) as e:
                msg = f"Error: {e}"
                return io.NodeOutput(None, 0, 0, cfg["block_size"], msg)
            except Exception:
                raise
        else:
            try:
                latent, w, h, details = create_latent_for_optimized(
                    dimensions, invert, cfg, batch_size, latent_alignment
                )
                # Optionally, provide a UI preview for details (uncomment if desired)
                # preview = ui.PreviewText(details)
                return io.NodeOutput(latent, w, h, cfg["block_size"], details)
            except (ValueError, TypeError) as e:
                msg = f"Error: {e}"
                return io.NodeOutput(None, 0, 0, cfg["block_size"], msg)
            except Exception:
                raise
