import logging
from typing import Optional

from comfy_api.latest import io


class FensTokenCounter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FensTokenCounter",
            display_name="Fens Token Counter",
            category="Fens_Simple_Nodes/Utility",
            search_aliases=["token", "tokens", "token count", "count tokens"],
            description="Count typed prompt tokens and show the active tokenizer context window usage.",
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="ComfyUI CLIP object (text encoder stack) from the current workflow.",
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="The text to be encoded or counted.",
                    optional=True,
                ),
                io.Combo.Input(
                    "count_strategy",
                    options=["max_stream", "sum_streams"],
                    default="max_stream",
                    advanced=True,
                    tooltip="How to aggregate counts across tokenizer branches (e.g. l/g/t5xxl): max_stream = largest branch count, sum_streams = sum of all branches.",
                ),
            ],
            outputs=[
                io.Int.Output(
                    display_name="total_tokens",
                    tooltip="Typed token count (excluding padding and most special tokens).",
                ),
                io.Int.Output(
                    display_name="context_limit_tokens",
                    tooltip="Total available tokens in the active context window tier (e.g. 77, 154, 231).",
                ),
                io.Int.Output(
                    display_name="chunk_count",
                    tooltip="Number of tokenizer chunks/windows used for this prompt.",
                ),
                io.String.Output(
                    display_name="details",
                    tooltip="Human-readable summary of typed tokens and context usage.",
                ),
                io.String.Output(
                    display_name="text",
                    tooltip="The input prompt (multiline string).",
                ),
            ],
        )

    @classmethod
    def _count_stream_prompt_tokens(cls, stream_batches: list[list[tuple]]) -> int:
        total = 0
        for batch in stream_batches:
            for token_item in batch:
                # With return_word_ids=True, Comfy tokenizers generally return
                # token-weight tuples with word_id metadata:
                # (token, weight, word_id). Special tokens typically use word_id=0.
                if isinstance(token_item, (tuple, list)) and len(token_item) >= 3:
                    if token_item[2] > 0:
                        total += 1
                else:
                    total += 1
        return total

    @classmethod
    def _stream_context_limit_tokens(cls, stream_batches: list[list[tuple]]) -> int:
        return sum(len(batch) for batch in stream_batches)

    @classmethod
    def execute(
        cls,
        clip,
        text: Optional[str] = None,
        count_strategy: str = "max_stream",
    ) -> io.NodeOutput:
        if clip is None:
            logging.warning("FensTokenCounter: clip input is None.")
            return io.NodeOutput(0, 0, 0, "No CLIP input connected.", text or "")

        if not text or not text.strip():
            return io.NodeOutput(0, 0, 0, "No prompt text provided.", text or "")

        try:
            # token_streams is a dict keyed by tokenizer branch names
            # (for example l/g/t5xxl).
            token_streams = clip.tokenize(text, return_word_ids=True)
            if not isinstance(token_streams, dict) or not token_streams:
                return io.NodeOutput(
                    0, 0, 0, "Tokenizer returned no token streams.", text
                )

            prompt_counts = [
                cls._count_stream_prompt_tokens(stream_batches)
                for stream_batches in token_streams.values()
            ]
            context_limits = [
                cls._stream_context_limit_tokens(stream_batches)
                for stream_batches in token_streams.values()
            ]
            chunk_counts = [
                len(stream_batches) for stream_batches in token_streams.values()
            ]

            if count_strategy == "sum_streams":
                token_count = sum(prompt_counts)
                context_limit_tokens = sum(context_limits)
                chunk_count = sum(chunk_counts)
            else:
                token_count = max(prompt_counts)
                context_limit_tokens = max(context_limits)
                chunk_count = max(chunk_counts)

            details = (
                f"Prompt tokens: {token_count} | "
                f"Context limit: {context_limit_tokens} | "
                f"Chunks: {chunk_count} | "
                f"Strategy: {count_strategy}"
            )

            return io.NodeOutput(
                token_count,
                context_limit_tokens,
                chunk_count,
                details,
                text,
            )
        except Exception as e:
            logging.error("FensTokenCounter: Failed to tokenize text. Error: %s", e)
            return io.NodeOutput(0, 0, 0, f"Error: {e}", text or "")
