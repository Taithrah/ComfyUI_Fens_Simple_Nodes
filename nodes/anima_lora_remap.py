"""
Anima LoRA Remap (model): MODEL in -> MODEL out. Loads nothing.

Put one of these nodes on the MODEL wire BEFORE any LoRA loader/stacker. Every LoRA
added downstream (any loader that goes through ModelPatcher.add_patches) is remapped
from the 28-block anima-base-v1.0 block layout to the larger checkpoint's automatically.

Anima-2.9B and Anima-3.8B are each depth expansions built on the model before them
(LLaMA-Pro style block insertion): Anima-2.9B inserts 12 blocks into the 28-block
anima-base-v1.0 to reach 40; Anima-3.8B inserts 12 more into the 40-block Anima-2.9B to
reach 52 (per the checkpoint's own metadata: expansion_source Anima-2.9B,
old_block_count 40, new_block_count 52, insertion_positions
[3,7,11,15,19,23,27,31,35,39,43,47]). LoRAs, though, are trained on the original
28-block base -- so a LoRA applied to either larger checkpoint has its block indices
misaligned unless remapped all the way from the 28-block layout, which is what both
nodes below do (the 3.8B one composes both expansion steps). The misalignment is
silent either way, since the inserted blocks' indices are also valid indices in the
larger model, so no "lora key not loaded" warning is ever emitted.

How: every standard loader ends in ModelPatcher.add_patches(patches, strength), with
patch keys already resolved against the larger model. We intercept there and shift the
block index, then translate the applied keys back to their original names so
ComfyUI's own "lora key not loaded" integrity check still compares correctly instead
of flagging every renamed key as a false mismatch. clone() rebuilds patchers with
self.__class__, so the behaviour follows the model through the rest of the graph; no
global state is touched.

Requires the model to actually load with all of its target block count (e.g. the
40-block Anima-2.9B needs the blocksPatch / original ComfyUI-Anima-2.9B, or a ComfyUI
build with PR #15555, before this node can do anything).
"""

import logging
import re

import comfy.lora
from comfy_api.latest import io

LOG_PREFIX = "[Anima LoRA Remap]"

# Anchored so diffusion_model.llm_adapter.blocks.N.* never matches.
MODEL_BLOCK_RE = re.compile(r"^(diffusion_model\.blocks\.)(\d+)(\..*)$")


def _model_blocks(key_map):
    out = set()
    for v in key_map.values():
        m = MODEL_BLOCK_RE.match(v) if isinstance(v, str) else None
        if m:
            out.add(int(m.group(2)))
    return out


def _is_anima(key_map):
    return any(
        isinstance(v, str) and v.startswith("diffusion_model.llm_adapter.")
        for v in key_map.values()
    )


def _patch_key_name(key):
    # patch keys are either "diffusion_model.x.weight" or ("diffusion_model.x.weight", slice_info)
    return key[0] if isinstance(key, tuple) else key


def _patch_blocks(patches):
    out = set()
    for k in patches.keys():
        name = _patch_key_name(k)
        m = MODEL_BLOCK_RE.match(name) if isinstance(name, str) else None
        if m:
            out.add(int(m.group(2)))
    return out


def _remap_single_key(key, base_to_target):
    name = _patch_key_name(key)
    m = MODEL_BLOCK_RE.match(name) if isinstance(name, str) else None
    if m is None:
        return key
    dst = base_to_target[int(m.group(2))]
    new_name = f"{m.group(1)}{dst}{m.group(3)}"
    return (new_name,) + tuple(key[1:]) if isinstance(key, tuple) else new_name


class _AnimaExpansion:
    """A depth expansion: a LoRA trained on the smaller (base) block count, remapped onto the larger (target) one."""

    def __init__(self, label, inserted_blocks, num_target_blocks):
        self.label = label
        self.num_target_blocks = num_target_blocks
        kept = [i for i in range(num_target_blocks) if i not in inserted_blocks]
        self.num_base_blocks = len(kept)
        assert self.num_base_blocks == num_target_blocks - len(inserted_blocks)
        self.base_to_target = dict(
            enumerate(kept)
        )  # base block idx -> target block idx

    @classmethod
    def compose(cls, first, second, label):
        """Chain base->mid (first) with mid->target (second) into one base->target expansion.

        Both of our nodes are meant to remap LoRAs trained on the original 28-block
        anima-base-v1.0, regardless of which larger checkpoint they're applied to. Anima-3.8B
        is two expansion steps away from that base (28->40->52), so its node needs the
        composed 28->52 mapping, not the 40->52 mapping between 2.9B and 3.8B alone.
        """
        if first.num_target_blocks != second.num_base_blocks:
            raise ValueError(
                f"cannot compose {first.label!r} (target {first.num_target_blocks} blocks) with "
                f"{second.label!r} (base {second.num_base_blocks} blocks): block counts don't match"
            )
        obj = cls.__new__(cls)
        obj.label = label
        obj.num_base_blocks = first.num_base_blocks
        obj.num_target_blocks = second.num_target_blocks
        obj.base_to_target = {
            j: second.base_to_target[mid] for j, mid in first.base_to_target.items()
        }
        return obj


# anima-base-v1.0 (28 blocks) -> Anima-2.9B (40 blocks): 12 blocks inserted at these indices.
EXPANSION_29B = _AnimaExpansion(
    "2.9B",
    inserted_blocks=(2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36),
    num_target_blocks=40,
)

# Anima-2.9B (40 blocks) -> Anima-3.8B (52 blocks): 12 more blocks inserted, per the
# 3.8B checkpoint's own metadata (insertion_positions in the 52-block numbering). This is
# an intermediate stage only -- LoRAs are trained on the 28-block base, not on 2.9B, so
# this gets composed with EXPANSION_29B below rather than used directly by any node.
_EXPANSION_29B_TO_38B = _AnimaExpansion(
    "2.9B->3.8B",
    inserted_blocks=(3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47),
    num_target_blocks=52,
)

# anima-base-v1.0 (28 blocks) -> Anima-3.8B (52 blocks) directly: a 28-block LoRA remapped
# straight onto 3.8B, matching the same base checkpoint the 2.9B node remaps LoRAs from.
EXPANSION_38B = _AnimaExpansion.compose(
    EXPANSION_29B, _EXPANSION_29B_TO_38B, label="3.8B"
)


def _make_remap_mixin(expansion):
    class _AnimaRemapMixin:
        _expansion = expansion

        def add_patches(self, patches, *args, **kwargs):
            exp = self._expansion
            # Only a patch set covering exactly the base block range is remapped. An
            # already-remapped set (or a native target-sized one) includes blocks beyond
            # that range, so it never gets double-mapped.
            if _patch_blocks(patches) != set(range(exp.num_base_blocks)):
                return super().add_patches(patches, *args, **kwargs)

            remap = {k: _remap_single_key(k, exp.base_to_target) for k in patches}
            remapped_patches = {remap[k]: v for k, v in patches.items()}
            shifted = sum(1 for k, v in remap.items() if v != k)
            logging.info(
                f"{LOG_PREFIX} {exp.num_base_blocks}-block LoRA on {exp.num_target_blocks}-block "
                f"Anima-{exp.label}: remapped {shifted} target keys."
            )

            applied = super().add_patches(remapped_patches, *args, **kwargs)
            # Translate applied (destination-named) keys back to the caller's original names, so
            # ComfyUI's own "NOT LOADED" integrity check (which compares against the keys it
            # handed us) matches correctly instead of flagging every renamed key as a mismatch.
            applied_set = set(applied)
            return [k for k in patches if remap[k] in applied_set]

    return _AnimaRemapMixin


_MIXINS = {}
_PATCHED_CLASSES = {}


def _mixin_for(expansion):
    if expansion not in _MIXINS:
        _MIXINS[expansion] = _make_remap_mixin(expansion)
    return _MIXINS[expansion]


def _remap_class_for(base, expansion):
    mixin = _mixin_for(expansion)
    if issubclass(base, mixin):
        return base
    key = (base, expansion)
    if key not in _PATCHED_CLASSES:
        name = f"AnimaRemap{expansion.label.replace('.', '')}{base.__name__}"
        _PATCHED_CLASSES[key] = type(name, (mixin, base), {})
    return _PATCHED_CLASSES[key]


class _AnimaModelRemapNode(io.ComfyNode):
    _expansion = None  # set by subclass

    @classmethod
    def execute(cls, model):
        exp = cls._expansion
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        if not (
            _is_anima(key_map)
            and _model_blocks(key_map) == set(range(exp.num_target_blocks))
        ):
            logging.warning(
                f"{LOG_PREFIX} model is not a {exp.num_target_blocks}-block Anima-{exp.label}; "
                f"passing it through unchanged."
            )
            return io.NodeOutput(model)

        new_model = (
            model.clone()
        )  # never alter the upstream patcher other branches may share
        new_model.__class__ = _remap_class_for(type(new_model), exp)
        return io.NodeOutput(new_model)


class AnimaModelRemap29B(_AnimaModelRemapNode):
    _expansion = EXPANSION_29B

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FensAnimaModelRemap29B",
            display_name="Anima 2.9B LoRA Remap (model)",
            category="loaders",
            description="Place before any LoRA loader/stacker. 28-block Anima LoRAs added downstream are remapped to the "
            "40-block Anima-2.9B layout. Other LoRAs and other models are unaffected.",
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output()],
        )


class AnimaModelRemap38B(_AnimaModelRemapNode):
    _expansion = EXPANSION_38B

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FensAnimaModelRemap38B",
            display_name="Anima 3.8B LoRA Remap (model)",
            category="loaders",
            description="Place before any LoRA loader/stacker. 28-block Anima LoRAs added downstream are remapped to the "
            "52-block Anima-3.8B layout. Other LoRAs and other models are unaffected.",
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output()],
        )
