from __future__ import annotations

import math
from typing import Any

from comfy.model_management import intermediate_device, intermediate_dtype

from .latent_utils import align, make_latent, parse_exact_dimensions, parse_ratio

PIXEL_SCALE = 1024 * 1024  # Pixels per megapixel (1M reference point)
BLOCK_SIZE_THRESHOLD = 32  # Threshold for adaptive search range
SCORE_TOLERANCE = 1e-7  # Tolerance for score comparison (relaxed for FP precision)


def _within_edge_bounds(
    w: int,
    h: int,
    min_resolution: int | None,
    max_resolution: int | None,
) -> bool:
    """Check whether width/height satisfy optional per-edge resolution bounds."""
    return (
        min_resolution is None or (w >= min_resolution and h >= min_resolution)
    ) and (max_resolution is None or (w <= max_resolution and h <= max_resolution))


def find_resolution(
    ar: float, target_mp: float, block: int, model_cfg: dict[str, Any]
) -> tuple[int, int]:
    """Find the optimal resolution for a given aspect ratio and MP target.

    This is the shared implementation extracted from multiple nodes.

    Algorithm:
      1. Calculate ideal height from target_mp and aspect ratio
      2. Search around ideal height with ±search_range blocks
      3. Score candidates on MP accuracy + AR accuracy
      4. Return best resolution that respects AR and block constraints

    Args:
      ar: Target aspect ratio (width/height)
      target_mp: Target megapixels
      block: Block size alignment constraint
      model_cfg: Model configuration dict with search params

    Returns:
      Tuple of (width, height) in pixels, aligned to block size

    Raises:
      ValueError: If no valid resolution found within constraints
    """
    ideal_px = target_mp * PIXEL_SCALE
    raw_h = math.sqrt(ideal_px / ar)

    # Adaptive search range: smaller blocks → larger search needed
    config_range = int(model_cfg.get("search_range", 10))
    if block < BLOCK_SIZE_THRESHOLD:
        # For fine-grained blocks (16, 32), extend search for precision
        search_range = max(config_range, 20 - (block // 8))
    else:
        search_range = config_range

    min_ar = float(model_cfg.get("min_ar", 0.5))
    max_ar = float(model_cfg.get("max_ar", 4.0))
    # Block alignment pushes candidate ARs slightly outside the configured limits
    # (e.g. targeting AR 3.75 at block=64 rounds to 3.875 — a 3.3% overshoot).
    # Compute the minimum tolerance that block-alignment can impose and widen the
    # filter by that amount so valid candidates are never silently dropped.
    block_ar_overshoot = block / max(raw_h - block, block) if raw_h > block else 0.5
    _effective_min_ar = min_ar * (1.0 - block_ar_overshoot)
    _effective_max_ar = max_ar * (1.0 + block_ar_overshoot)

    # Hard floor/ceiling on edge length (independent of aspect ratio). Only
    # meaningful for models trained on a fixed set of resolution tiers rather
    # than a continuous MP target - e.g. Anima's 512/1024/1536 tiers. None
    # (the default) means unbounded, preserving prior behavior for models
    # that generalize continuously and have no documented hard edge limit.
    min_resolution = model_cfg.get("min_resolution")
    max_resolution = model_cfg.get("max_resolution")

    best_score = float("inf")
    best_w = best_h = 0
    best_pixels = 0

    for delta in range(-search_range, search_range + 1):
        h_try = raw_h + delta * block
        w_try = ar * h_try
        w = align(w_try, block)
        h = align(h_try, block)

        # Constraint: minimum resolution must be at least one block
        if w < block or h < block:
            continue

        # Hard edge-length bounds, independent of aspect ratio - this is
        # what actually keeps a wide/tall search from producing a resolution
        # the model was never trained to handle, regardless of what AR
        # bounds happen to be configured (AR bounds alone don't guarantee
        # this - see the 1776x592 case that motivated adding this check).
        if not _within_edge_bounds(w, h, min_resolution, max_resolution):
            continue

        candidate_ar = w / h
        if candidate_ar < _effective_min_ar or candidate_ar > _effective_max_ar:
            continue

        # Calculate error metrics with proper normalization
        actual_mp = (w * h) / PIXEL_SCALE
        mp_error = abs(actual_mp - target_mp) / target_mp if target_mp > 0 else 0
        ar_error = abs(candidate_ar - ar) / ar if ar > 0 else 0

        # Weighted score: MP accuracy is more critical (10:1)
        score = (10.0 * mp_error) + ar_error

        # Tie-breaking: prefer slightly larger resolutions (better detail)
        pixels = w * h

        if abs(score - best_score) < SCORE_TOLERANCE:
            # Same score: pick the one with more pixels
            if pixels > best_pixels:
                best_w, best_h = w, h
                best_pixels = pixels
        elif score < best_score:
            best_score = score
            best_w, best_h = w, h
            best_pixels = pixels

    if best_w == 0 or best_h == 0:
        bounds_note = ""
        if min_resolution is not None or max_resolution is not None:
            lo = min_resolution if min_resolution is not None else 0
            hi = max_resolution if max_resolution is not None else "∞"
            bounds_note = f", resolution bounds [{lo}, {hi}]px per edge"
        raise ValueError(
            f"No valid resolution found for AR~{ar:.3f}, target {target_mp}MP, "
            f"block {block}{bounds_note}. Try broadening search_range, aspect "
            f"ratio limits, or resolution bounds."
        )

    return best_w, best_h


def create_latent(
    w: int,
    h: int,
    batch_size: int,
    spacial_downscale_ratio: int,
    *,
    channels: int = 4,
):
    """Wrapper to create a latent with the project's device/dtype helpers."""
    return make_latent(
        w,
        h,
        batch_size,
        spacial_downscale_ratio,
        device=intermediate_device(),
        dtype=intermediate_dtype(),
        channels=channels,
    )


def generate_details(
    w: int,
    h: int,
    ar: float,
    cfg: dict[str, Any],
    latent_alignment: str,
    *,
    clamp_warning: str = "",
) -> str:
    """Generate human-readable details about the latent calculation.

    Includes resolution, aspect ratio, MP accuracy, block alignment, and model info.
    """
    actual_mp = (w * h) / PIXEL_SCALE
    target_mp = cfg.get("target_mp", 1.0)
    mp_delta = actual_mp - target_mp
    mp_pct = (mp_delta / target_mp * 100) if target_mp > 0 else 0

    # Show block size, VAE scale, and channel count for clarity
    block_size = cfg["block_size"]
    vae_scale = cfg["spacial_downscale_ratio"]
    channels = cfg.get("channels", 4)
    latent_w = w // vae_scale
    latent_h = h // vae_scale

    details = (
        f"Resolution: {w}×{h} px\n"
        f"Aspect Ratio: {ar:.4f}\n"
        f"Target MP: {target_mp:.6f}, Actual MP: {actual_mp:.6f} ({mp_pct:+.2f}%)\n"
        f"Block Size: {block_size}px, VAE Scale: {vae_scale}× → {latent_w}×{latent_h}×{channels}ch latent\n"
        f"Model: {cfg.get('desc', latent_alignment)}"
    )
    if clamp_warning:
        details = clamp_warning + "\n" + details
    return details


def resolve_cfg(
    model_config: dict[str, Any],
    latent_alignment: str,
    custom_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    dimensions: str, invert: bool, cfg: dict[str, Any], batch_size: int
):
    """Create latent for exact WxH input and return (latent, w, h, details).

    Validates that exact dimensions are properly aligned before creation.

    Args:
      dimensions: String in "WxH" or "W:H" format (e.g., "1024x768")
      invert: If True, swap width and height
      cfg: Model configuration dict
      batch_size: Number of images in batch

    Returns:
      Tuple of (latent_dict, width, height, details_string)

    Raises:
      ValueError: If dimensions invalid or not aligned to block/VAE constraints
      TypeError: If batch_size or config invalid
    """
    w, h = parse_exact_dimensions(dimensions)
    if invert:
        w, h = h, w

    # Validate block alignment
    block = cfg["block_size"]
    vae_scale = cfg["spacial_downscale_ratio"]

    rounding_warning = ""
    if w % block != 0 or h % block != 0:
        orig_w, orig_h = w, h
        w = align(w, block)
        h = align(h, block)
        rounding_warning = (
            f"⚠️ Requested {orig_w}×{orig_h} isn't a multiple of {cfg.get('desc', 'this model')}'s "
            f"{block}px block size - rounded to {w}×{h}."
        )

    latent = create_latent(
        w,
        h,
        batch_size,
        vae_scale,
        channels=cfg.get("channels", 4),
    )
    actual_ar = w / h
    actual_mp = (w * h) / PIXEL_SCALE
    channels = cfg.get("channels", 4)

    bounds_warning = ""
    min_resolution = cfg.get("min_resolution")
    max_resolution = cfg.get("max_resolution")
    out_of_bounds = (
        min_resolution is not None and (w < min_resolution or h < min_resolution)
    ) or (max_resolution is not None and (w > max_resolution or h > max_resolution))
    if out_of_bounds:
        lo = min_resolution if min_resolution is not None else 0
        hi = max_resolution if max_resolution is not None else "∞"
        bounds_warning = (
            f"⚠️ Exact resolution {w}×{h} falls outside {cfg.get('desc', 'this model')}'s "
            f"supported edge range ({lo}-{hi}px per side). Results may be degraded or unreliable."
        )

    details = (
        f"Exact Resolution: {w}×{h} px\n"
        f"Aspect Ratio: {actual_ar:.4f}\n"
        f"Actual MP: {actual_mp:.6f}\n"
        f"Block Size: {block}px, VAE Scale: {vae_scale}× → {w // vae_scale}×{h // vae_scale}×{channels}ch latent\n"
        f"Model: {cfg.get('desc', 'Custom')}"
    )
    warnings = [msg for msg in (rounding_warning, bounds_warning) if msg]
    if warnings:
        details = "\n".join(warnings) + "\n" + details
    return latent, w, h, details


def create_latent_for_optimized(
    dimensions: str,
    invert: bool,
    cfg: dict[str, Any],
    batch_size: int,
    latent_alignment: str,
):
    """Create latent for optimized (aspect-ratio) input and return (latent, w, h, details).

    Uses the find_resolution algorithm to locate best dimensions matching:
    - Target megapixels for the model
    - User-requested aspect ratio
    - Block size constraints
    - VAE downsampling requirements

    Args:
      dimensions: Aspect ratio in formats: "16:9", "16x9", or decimal "1.777"
      invert: If True, swap the aspect ratio (16:9 → 9:16)
      cfg: Model configuration dict with MP target, block size, search range
      batch_size: Number of images in batch
      latent_alignment: Model preset name (for reporting)

    Returns:
      Tuple of (latent_dict, width, height, details_string)

    Raises:
      ValueError: If aspect ratio invalid or outside model constraints
    """
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

    latent = create_latent(
        w,
        h,
        batch_size,
        cfg["spacial_downscale_ratio"],
        channels=cfg.get("channels", 4),
    )
    details = generate_details(
        w,
        h,
        w / h,
        cfg,
        latent_alignment,
        clamp_warning=clamp_warning,
    )
    return latent, w, h, details
