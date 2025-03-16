from typing import List
from transformers import AutoTokenizer
from .encoder_mapping import ENCODER_MODEL_MAPPING


class FensTokenCounter:
    NAME = "Fens Token Counter"

    @classmethod
    def INPUT_TYPES(cls):
        encoders_list = list(ENCODER_MODEL_MAPPING.keys())
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The prompt to be counted.",
                    },
                ),
                "encoders": (
                    encoders_list,
                    {"default": "CLIP-ViT-bigG-14-laion2B-39B-b160k"},
                ),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("total_token_count",)
    OUTPUT_TOOLTIPS = ("The token count using the selected encoder",)
    CATEGORY = "Fens_Simple_Nodes"
    FUNCTION = "count_tokens"
    DESCRIPTION = (
        "Count tokens in a prompt using the selected encoder to return a total count."
    )

    def count_tokens(self, prompt: str, encoders: List[str]) -> tuple:
        if isinstance(encoders, str):
            encoders = [encoders]

        tokenizer_cache = {}
        total_tokens = 0

        for encoder in encoders:
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
                total_tokens += len(tokenizer.tokenize(prompt))
            except Exception:
                continue

        return (total_tokens,)
