from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from comfy.model_management import intermediate_device, intermediate_dtype

from .latent_utils import align, make_latent, parse_exact_dimensions, parse_ratio

PIXEL_SCALE = 1024 * 1024


def find_resolution(
    ar: float, target_mp: float, block: int, model_cfg: Dict[str, Any]
) -> Tuple[int, int]:
    """Find the optimal resolution for a given aspect ratio and MP target.

    This is the shared implementation extracted from multiple nodes.
    """
    ideal_px = target_mp * PIXEL_SCALE
    raw_h = math.sqrt(ideal_px / ar)
    search_range = int(model_cfg.get("search_range", 5 if block >= 32 else 10))
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
        actual_mp = (w * h) / PIXEL_SCALE
        mp_error = abs(actual_mp - target_mp) / target_mp
        ar_error = abs(candidate_ar - ar) / ar
        mp_weight = 10.0
        ar_weight = 1.0
        score = (mp_weight * mp_error) + (ar_weight * ar_error)
        if abs(score - best_score) < 1e-9:
            if (w * h) > (best_w * best_h):
                best_w, best_h = w, h
        elif score < best_score:
            best_score = score
            best_w, best_h = w, h
    if best_w == 0 or best_h == 0:
        raise ValueError(
            f"No valid resolution found for AR~{ar:.3f}, target {target_mp}MP, "
            f"block {block}. Try broadening search_range or aspect ratio limits."
        )
    return best_w, best_h


def create_latent(w: int, h: int, batch_size: int, spacial_downscale_ratio: int):
    """Wrapper to create a latent with the project's device/dtype helpers."""
    return make_latent(
        w,
        h,
        batch_size,
        spacial_downscale_ratio,
        intermediate_device(),
        dtype=intermediate_dtype(),
    )


def generate_details(
    w: int,
    h: int,
    ar: float,
    cfg: Dict[str, Any],
    latent_alignment: str,
    clamp_warning: str = "",
) -> str:
    details = (
        f"Resolution: {w}x{h} px\n"
        f"Aspect Ratio: {ar:.4f}\n"
        f"Target MP: {cfg.get('target_mp')}, Actual MP: {(w * h) / PIXEL_SCALE:.3f}\n"
        f"Block Size: {cfg['block_size']}, VAE Scale: {cfg['spacial_downscale_ratio']}\n"
        f"Model: {cfg.get('desc', latent_alignment)}"
    )
    if clamp_warning:
        details = clamp_warning + "\n" + details
    return details


def resolve_cfg(
    model_config: Dict[str, Any],
    latent_alignment: str,
    custom_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Resolve the effective configuration for a given alignment, applying custom overrides if requested."""
    if latent_alignment == "Custom":
        cfg = model_config.get("Custom", {}).copy()
        if custom_overrides:
            cfg.update(custom_overrides)
        return cfg
    cfg = model_config.get(latent_alignment)
    if cfg is None:
        raise ValueError(f"Unknown latent_alignment '{latent_alignment}'")
    return cfg


def create_latent_for_exact(
    dimensions: str, invert: bool, cfg: Dict[str, Any], batch_size: int
):
    """Create latent for exact WxH input and return (latent, w, h, details)."""
    w, h = parse_exact_dimensions(dimensions)
    if invert:
        w, h = h, w
    w = align(w, cfg["block_size"])
    h = align(h, cfg["block_size"])
    latent = create_latent(w, h, batch_size, cfg["spacial_downscale_ratio"])
    details = (
        f"Exact Resolution: {w}x{h} px\n"
        f"Aspect Ratio: {w / h:.4f}\n"
        f"Block Size: {cfg['block_size']}, VAE Scale: {cfg['spacial_downscale_ratio']}\n"
        f"Model: {cfg.get('desc', 'Custom')}"
    )
    return latent, w, h, details


def create_latent_for_optimized(
    dimensions: str,
    invert: bool,
    cfg: Dict[str, Any],
    batch_size: int,
    latent_alignment: str,
):
    """Create latent for optimized (aspect-ratio) input and return (latent, w, h, details)."""
    ar = parse_ratio(dimensions)
    min_ar = float(cfg.get("min_ar", 0.5))
    max_ar = float(cfg.get("max_ar", 3.75))
    clamp_warning = ""
    if not (min_ar <= ar <= max_ar):
        clamp_warning = (
            f"⚠️ Dimensions {ar:.3f} are outside recommended range for {latent_alignment} "
            f"({min_ar:.2f}-{max_ar:.2f}). Clamping for best results."
        )
        ar = max(min_ar, min(ar, max_ar))
    w, h = find_resolution(ar, cfg["target_mp"], cfg["block_size"], cfg)
    if invert:
        w, h = h, w
    latent = create_latent(w, h, batch_size, cfg["spacial_downscale_ratio"])
    details = generate_details(w, h, w / h, cfg, latent_alignment, clamp_warning)
    return latent, w, h, details
