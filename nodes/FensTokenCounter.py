from transformers import CLIPTokenizer, T5Tokenizer

from .encoder_mapping import ENCODER_MODEL_MAPPING


class FensTokenCounter:
    @classmethod
    def INPUT_TYPES(cls):
        encoder_keys = list(ENCODER_MODEL_MAPPING.keys())
        default_encoder = "CLIP BigG-14 (LAION, Patch 14)"
        if default_encoder not in encoder_keys:
            default_encoder = encoder_keys[0]

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

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("total_tokens",)
    OUTPUT_NODE = True
    OUTPUT_TOOLTIPS = ("The token count using the selected encoders.",)
    CATEGORY = "Fens_Simple_Nodes/Utility"
    FUNCTION = "count_tokens"
    DESCRIPTION = "Get the token count of a prompt using the selected encoders."

    def count_tokens(self, primary_encoder: str, text: str = "") -> tuple:
        # If text is empty, return 0 tokens
        if not text.strip():
            return (0,)

        tokenizer_cache = {}
        total_tokens = 0

        def get_tokenizer(model_name: str):
            # Check if the model name contains "t5" to use T5Tokenizer
            if "t5" in model_name.lower():
                return T5Tokenizer.from_pretrained(model_name, legacy=True)
            else:
                return CLIPTokenizer.from_pretrained(model_name)

        # Process primary encoder
        model_name = ENCODER_MODEL_MAPPING.get(primary_encoder)
        if model_name:
            try:
                if model_name not in tokenizer_cache:
                    tokenizer_cache[model_name] = get_tokenizer(model_name)

                tokenizer = tokenizer_cache[model_name]
                total_tokens += len(tokenizer.tokenize(text))
            except Exception as e:
                print(f"Error processing primary encoder '{primary_encoder}': {e}")

        return (total_tokens,)
