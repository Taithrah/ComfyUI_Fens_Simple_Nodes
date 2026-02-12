import logging
from typing import Optional, Union

from comfy_api.latest import io
from transformers import CLIPTokenizer, T5Tokenizer

from .encoder_mapping import ENCODER_MODEL_MAPPING


class FensTokenCounter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        encoder_keys = list(ENCODER_MODEL_MAPPING.keys())
        default_encoder = "CLIP BigG-14 (LAION, Patch 14)"
        if default_encoder not in encoder_keys:
            default_encoder = encoder_keys[0] if encoder_keys else ""
        return io.Schema(
            node_id="FensTokenCounter",
            display_name="Fens Token Counter",
            category="Fens_Simple_Nodes/Utility",
            search_aliases=["token", "tokens", "token count", "count tokens"],
            description="Get the token count of a prompt using the selected encoders.",
            inputs=[
                io.Combo.Input(
                    "primary_encoder",
                    options=encoder_keys,
                    default=default_encoder,
                    tooltip="Select the primary encoder to use.",
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="The text to be encoded or counted.",
                    optional=True,
                ),
            ],
            outputs=[
                io.Int.Output(
                    display_name="total_tokens",
                    tooltip="The token count using the selected encoders.",
                ),
                io.String.Output(
                    display_name="text",
                    tooltip="The input prompt (multiline string).",
                ),
            ],
        )

    # Class-level tokenizer cache
    _tokenizer_cache: dict[str, Union[CLIPTokenizer, T5Tokenizer]] = {}

    @classmethod
    def _get_tokenizer(cls, model_name: str) -> Union[CLIPTokenizer, T5Tokenizer]:
        """Load tokenizer with automatic model downloading from HuggingFace."""
        try:
            if "t5" in model_name.lower():
                return T5Tokenizer.from_pretrained(model_name, legacy=True)
            return CLIPTokenizer.from_pretrained(model_name)
        except Exception as e:
            logging.error(
                f"FensTokenCounter: Failed to load tokenizer {model_name}. Error: {e}"
            )
            raise

    @classmethod
    def execute(cls, primary_encoder: str, text: Optional[str] = None) -> io.NodeOutput:
        if not primary_encoder:
            logging.warning("FensTokenCounter: No primary_encoder provided.")
            return io.NodeOutput(0, text or "")

        if not text or not text.strip():
            return io.NodeOutput(0, text or "")

        model_name = ENCODER_MODEL_MAPPING.get(primary_encoder)
        if not model_name:
            logging.warning(
                f"FensTokenCounter: Encoder '{primary_encoder}' not found in mapping."
            )
            return io.NodeOutput(0, text or "")

        try:
            if model_name not in cls._tokenizer_cache:
                logging.info(f"FensTokenCounter: Loading tokenizer for {model_name}...")
                cls._tokenizer_cache[model_name] = cls._get_tokenizer(model_name)

            tokenizer = cls._tokenizer_cache[model_name]
            token_count = len(tokenizer.tokenize(text))
            return io.NodeOutput(token_count, text)
        except Exception as e:
            logging.error(
                f"FensTokenCounter: Failed to tokenize text with {primary_encoder}. Error: {e}"
            )
            return io.NodeOutput(0, text or "")
