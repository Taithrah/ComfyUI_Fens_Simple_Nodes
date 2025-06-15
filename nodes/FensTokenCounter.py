import warnings

from transformers import CLIPTokenizer, T5Tokenizer

from .encoder_mapping import ENCODER_MODEL_MAPPING


class FensTokenCounter:
    """
    Node to count tokens in a prompt using a selected encoder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        encoder_keys = list(ENCODER_MODEL_MAPPING.keys())
        default_encoder = "CLIP BigG-14 (LAION, Patch 14)"
        if default_encoder not in encoder_keys:
            default_encoder = encoder_keys[0] if encoder_keys else ""
        return {
            "required": {
                "primary_encoder": (
                    encoder_keys,
                    {
                        "default": default_encoder,
                        "tooltip": "Select the primary encoder to use.",
                    },
                )
            },
            "optional": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The prompt to count.",
                    },
                ),
            },
        }

    CATEGORY = "Fens_Simple_Nodes/Utility"
    DESCRIPTION = "Get the token count of a prompt using the selected encoders."
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("total_tokens",)
    OUTPUT_NODE = True
    OUTPUT_TOOLTIPS = ("The token count using the selected encoders.",)
    FUNCTION = "count_tokens"

    # Class-level tokenizer cache
    _tokenizer_cache = {}

    @staticmethod
    def _get_tokenizer(model_name: str):
        """
        Returns a tokenizer instance for the given model name.
        """
        if "t5" in model_name.lower():
            return T5Tokenizer.from_pretrained(model_name, legacy=True)
        else:
            return CLIPTokenizer.from_pretrained(model_name)

    def count_tokens(self, primary_encoder: str, text: str = "") -> tuple:
        """
        Counts the number of tokens in the given text using the selected encoder.
        """
        if not text or not text.strip():
            return (0,)

        model_name = ENCODER_MODEL_MAPPING.get(primary_encoder)
        if not model_name:
            warnings.warn(f"Unknown encoder: {primary_encoder}")
            return (0,)

        try:
            if model_name not in self._tokenizer_cache:
                self._tokenizer_cache[model_name] = self._get_tokenizer(model_name)
            tokenizer = self._tokenizer_cache[model_name]
            token_count = len(tokenizer.tokenize(text))
            return (token_count,)
        except Exception as e:
            warnings.warn(f"Error processing encoder '{primary_encoder}': {e}")
            return (0,)
