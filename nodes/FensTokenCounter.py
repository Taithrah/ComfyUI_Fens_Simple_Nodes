from comfy_api.latest import io
from transformers import CLIPTokenizer, T5Tokenizer

from .encoder_mapping import ENCODER_MODEL_MAPPING


class FensTokenCounter(io.ComfyNode):
    """
    Node to count tokens in a prompt using a selected encoder.
    """

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
                    tooltip="The prompt to count.",
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
    _tokenizer_cache = {}

    @classmethod
    def _get_tokenizer(cls, model_name: str):
        if "t5" in model_name.lower():
            return T5Tokenizer.from_pretrained(model_name, legacy=True)
        else:
            return CLIPTokenizer.from_pretrained(model_name)

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        primary_encoder = kwargs.get("primary_encoder", "")
        text = kwargs.get("text", "")
        if not text or not text.strip():
            return io.NodeOutput(0, text)
        model_name = ENCODER_MODEL_MAPPING.get(primary_encoder)
        if not model_name:
            return io.NodeOutput(0, text)
        try:
            if model_name not in cls._tokenizer_cache:
                cls._tokenizer_cache[model_name] = cls._get_tokenizer(model_name)
            tokenizer = cls._tokenizer_cache[model_name]
            token_count = len(tokenizer.tokenize(text))
            return io.NodeOutput(token_count, text)
        except Exception:
            return io.NodeOutput(0, text)
