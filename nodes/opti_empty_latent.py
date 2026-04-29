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


class OptiEmptyLatent(io.ComfyNode):
    """
    Node to create an empty latent tensor with optimal or exact resolution for a given aspect ratio or WxH.
    Integrates tightly with ComfyUI V3 node API and provides UI-friendly output.
    """

    config_path = os.path.join(os.path.dirname(__file__), "model_config.yaml")
    with open(config_path, encoding="utf-8") as f:
        MODEL_CONFIG = yaml.safe_load(f)

    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        alignment_options = [k for k in cls.MODEL_CONFIG if k != "Custom"]
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
                    display_name="Dimensions",
                    default="1:1",
                    tooltip="Formats: W:H (e.g. 16:9), WxH (e.g. 1280x720), or decimal (e.g. 1.777). Use WxH when 'Optimization' is FALSE.",
                ),
                io.Combo.Input(
                    "latent_alignment",
                    display_name="Latent Alignment",
                    options=alignment_options,
                    default="SDXL (1024px)",
                    tooltip="Optimization preset for model type.",
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
                    default=1,
                    min=1,
                    max=4096,
                    tooltip="Number of latent images in batch (VRAM usage increases with batch size).",
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
    ) -> io.NodeOutput:
        """
        Create an empty latent tensor with optimal or exact resolution.
        Returns latent, width, height, block size, and details string.
        """
        try:
            cfg = resolve_cfg(cls.MODEL_CONFIG, latent_alignment)
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
                # Unexpected error: re-raise to avoid masking bugs.
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
                # Unexpected error: re-raise to avoid masking bugs.
                raise
