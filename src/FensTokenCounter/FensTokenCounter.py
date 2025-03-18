from typing import List
from transformers import AutoTokenizer
from .encoder_mapping import ENCODER_MODEL_MAPPING


class FensTokenCounter:
    NAME = "Fens Token Counter"

    @classmethod
    def INPUT_TYPES(cls):
        encoder_keys = list(ENCODER_MODEL_MAPPING.keys())
        default_encoder = "CLIP-ViT-bigG-14-laion2B-39B-b160k"
        if default_encoder not in encoder_keys:
            default_encoder = encoder_keys[0]

        return {
            "required": {
                "encoder": (
                    encoder_keys,
                    {
                        "default": default_encoder,
                        "tooltip": "Select the encoder to use.",
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The prompt to count.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("Total Tokens",)
    OUTPUT_TOOLTIPS = ("The token count using the selected encoder",)
    CATEGORY = "Fens_Simple_Nodes"
    FUNCTION = "count_tokens"
    DESCRIPTION = "Get the token count of a prompt using the selected encoder. "

    def count_tokens(self, text: str, encoder: List[str]) -> tuple:
        if isinstance(encoder, str):
            encoder = [encoder]

        tokenizer_cache = {}
        total_tokens = 0

        for encoder in encoder:
            model_name = ENCODER_MODEL_MAPPING.get(encoder)
            if not model_name:
                continue

            try:
                if model_name not in tokenizer_cache:
                    # Universal application of legacy=False
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        # legacy=False if "t5" in model_name.lower() else None  # Only set for T5
                        legacy=False,  # Apply to all tokenizers
                    )
                    tokenizer_cache[model_name] = tokenizer

                tokenizer = tokenizer_cache[model_name]
                total_tokens += len(tokenizer.tokenize(text))
            except Exception:
                continue

        return (total_tokens,)
