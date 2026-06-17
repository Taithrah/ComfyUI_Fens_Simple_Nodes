from __future__ import annotations

import math
from typing import Any

from comfy.model_management import intermediate_device, intermediate_dtype

from .latent_utils import align, make_latent, parse_exact_dimensions, parse_ratio

PIXEL_SCALE = 1024 * 1024  # Pixels per megapixel (1M reference point)
BLOCK_SIZE_THRESHOLD = 32  # Threshold for adaptive search range
SCORE_TOLERANCE = 1e-7  # Tolerance for score comparison (relaxed for FP precision)


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

        candidate_ar = w / h
        if candidate_ar < _effective_min_ar or candidate_ar > _effective_max_ar:
            continue

        # Calculate error metrics with proper normalization
        actual_mp = (w * h) / PIXEL_SCALE
        mp_error = abs(actual_mp - target_mp) / target_mp if target_mp > 0 else 0
        ar_error = abs(candidate_ar - ar) / ar if ar > 0 else 0

        # Weighted score: MP accuracy is more critical (10:1)
        mp_weight = 10.0
        ar_weight = 1.0
        score = (mp_weight * mp_error) + (ar_weight * ar_error)

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
        raise ValueError(
            f"No valid resolution found for AR~{ar:.3f}, target {target_mp}MP, "
            f"block {block}. Try broadening search_range or aspect ratio limits."
        )

    return best_w, best_h


def create_latent(
    w: int, h: int, batch_size: int, spacial_downscale_ratio: int, channels: int = 4
):
    """Wrapper to create a latent with the project's device/dtype helpers."""
    return make_latent(
        w,
        h,
        batch_size,
        spacial_downscale_ratio,
        intermediate_device(),
        dtype=intermediate_dtype(),
        channels=channels,
    )


def generate_details(
    w: int,
    h: int,
    ar: float,
    cfg: dict[str, Any],
    latent_alignment: str,
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

    if w % block != 0 or h % block != 0:
        w = align(w, block)
        h = align(h, block)
        # Note: alignment is implicit, but should warn user

    latent = create_latent(w, h, batch_size, vae_scale, cfg.get("channels", 4))
    actual_ar = w / h
    actual_mp = (w * h) / PIXEL_SCALE
    channels = cfg.get("channels", 4)

    details = (
        f"Exact Resolution: {w}×{h} px\n"
        f"Aspect Ratio: {actual_ar:.4f}\n"
        f"Actual MP: {actual_mp:.6f}\n"
        f"Block Size: {block}px, VAE Scale: {vae_scale}× → {w // vae_scale}×{h // vae_scale}×{channels}ch latent\n"
        f"Model: {cfg.get('desc', 'Custom')}"
    )
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
        w, h, batch_size, cfg["spacial_downscale_ratio"], cfg.get("channels", 4)
    )
    details = generate_details(w, h, w / h, cfg, latent_alignment, clamp_warning)
    return latent, w, h, details
