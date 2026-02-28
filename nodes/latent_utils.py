from typing import Tuple

import torch


def parse_ratio(dimensions: str) -> float:
    s = dimensions.strip()
    if ":" in s:
        parts = s.split(":", 1)
    elif "x" in s.lower():
        parts = s.lower().split("x", 1)
    else:
        try:
            return float(s)
        except ValueError as exc:
            raise ValueError(f"Invalid dimensions format: '{dimensions}'") from exc
    w, h = map(float, parts)
    if h == 0:
        raise ValueError(f"Height cannot be zero in dimensions '{dimensions}'")
    return w / h


def parse_exact_dimensions(dimensions: str) -> Tuple[int, int]:
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


def align(value: float, block: int) -> int:
    return max(block, int(round(value / block)) * block)


def make_latent(
    w: int, h: int, bs: int, spacial_downscale_ratio: int, device: torch.device
) -> dict[str, torch.Tensor]:
    shape = (
        bs,
        4,
        h // spacial_downscale_ratio,
        w // spacial_downscale_ratio,
    )
    return {"samples": torch.zeros(shape, device=device)}
