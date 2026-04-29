from __future__ import annotations

from typing import Any

import torch
from comfy.model_management import intermediate_dtype

_SPLIT_INDEX = 2  # Used for splitting dimension strings (W:H or WxH)


def parse_ratio(dimensions: str) -> float:
    """
    Parse a string representing an aspect ratio (W:H, WxH, or decimal) and return the float ratio (W/H).
    Raises ValueError for invalid formats or non-positive values.
    """
    s = dimensions.strip()
    if ":" in s:
        parts = s.split(":", 1)
    elif "x" in s.lower():
        parts = s.lower().split("x", 1)
    else:
        try:
            value = float(s)
        except ValueError as exc:
            raise ValueError(f"Invalid dimensions format: '{dimensions}'") from exc
        if value <= 0:
            raise ValueError(f"Aspect ratio must be positive: '{dimensions}'")
        return value

    if len(parts) != _SPLIT_INDEX:
        raise ValueError(f"Invalid ratio format. Use W:H or WxH, got '{dimensions}'")

    try:
        w, h = map(float, map(str.strip, parts))
    except ValueError as exc:
        raise ValueError(f"Invalid ratio numeric format: '{dimensions}'") from exc

    if w <= 0 or h <= 0:
        raise ValueError(
            f"Width and height must be positive in dimensions '{dimensions}'"
        )
    return w / h


def parse_exact_dimensions(dimensions: str) -> tuple[int, int]:
    """
    Parse a string representing exact dimensions (WxH or W:H) and return (width, height) as integers.
    Raises ValueError for invalid formats or non-positive values.
    """
    s = dimensions.strip().lower()
    if "x" in s:
        parts = s.split("x", 1)
    elif ":" in s:
        parts = s.split(":", 1)
    else:
        raise ValueError(
            f"Use WxH or W:H format for exact resolution, got '{dimensions}'"
        )

    if len(parts) != _SPLIT_INDEX:
        raise ValueError(f"Invalid format. Use WxH or W:H, got '{dimensions}'")

    try:
        w, h = map(int, map(str.strip, parts))
    except ValueError as exc:
        raise ValueError(
            f"Invalid exact dimensions numeric format: '{dimensions}'"
        ) from exc

    if w <= 0 or h <= 0:
        raise ValueError(
            f"Width and height must be positive in dimensions '{dimensions}'"
        )

    return w, h


def align(value: float, block: int) -> int:
    """
    Align a value to the nearest multiple of block size (minimum block).
    Raises ValueError if block is not positive.
    """
    if block <= 0:
        raise ValueError(f"Block size must be positive, got {block}.")
    return max(block, int(round(value / block)) * block)


def make_latent(
    w: int,
    h: int,
    bs: int,
    spacial_downscale_ratio: int,
    device: torch.device,
    dtype=None,
) -> dict[str, Any]:
    """
    Create a latent tensor dict for ComfyUI, with shape (bs, 4, h//downscale, w//downscale).
    Always uses intermediate_dtype for compatibility. Raises ValueError for invalid sizes or ratios.
    """
    if w <= 0 or h <= 0 or bs <= 0:
        raise ValueError(f"Invalid latent size {w}x{h}, batch_size={bs}")

    if spacial_downscale_ratio <= 0:
        raise ValueError(f"Invalid downscale ratio {spacial_downscale_ratio}")

    if w % spacial_downscale_ratio != 0 or h % spacial_downscale_ratio != 0:
        raise ValueError(
            f"Width and height must be divisible by spacial_downscale_ratio ({spacial_downscale_ratio}): {w}x{h}"
        )

    dtype = intermediate_dtype()
    shape = (
        bs,
        4,
        h // spacial_downscale_ratio,
        w // spacial_downscale_ratio,
    )
    # Include the downscale ratio so Comfy's sampler can adjust empty latents for models with different latent formats.
    return {
        "samples": torch.zeros(shape, device=device, dtype=dtype),
        "downscale_ratio_spacial": spacial_downscale_ratio,
    }
