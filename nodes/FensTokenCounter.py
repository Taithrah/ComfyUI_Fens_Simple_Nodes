from __future__ import annotations

import logging
from typing import Any, Optional

from comfy_api.latest import io
from typing_extensions import override


class FensTokenCounter(io.ComfyNode):
    """
    Counts prompt tokens using the provided CLIP object and shows context window usage.
    Integrates tightly with ComfyUI V3 node API and provides UI-friendly output.
    """

    @classmethod
    @override
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
                    display_name="CLIP",
                    tooltip="ComfyUI CLIP object (text encoder stack) from the current workflow.",
                ),
                io.String.Input(
                    "text",
                    display_name="Prompt Text",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="The text to be encoded or counted.",
                    optional=True,
                ),
                io.Combo.Input(
                    "count_strategy",
                    display_name="Count Strategy",
                    options=["max_stream", "sum_streams"],
                    default="max_stream",
                    advanced=True,
                    tooltip="How to aggregate counts across tokenizer branches (e.g. l/g/t5xxl): max_stream = largest branch count, sum_streams = sum of all branches.",
                ),
            ],
            outputs=[
                io.Int.Output(
                    "total_tokens",
                    display_name="Total Tokens",
                    tooltip="Typed token count (excluding padding and most special tokens).",
                ),
                io.Int.Output(
                    "context_limit_tokens",
                    display_name="Context Limit Tokens",
                    tooltip="Total available tokens in the active context window tier (e.g. 77, 154, 231).",
                ),
                io.Int.Output(
                    "chunk_count",
                    display_name="Chunk Count",
                    tooltip="Number of tokenizer chunks/windows used for this prompt.",
                ),
                io.String.Output(
                    "details",
                    display_name="Details",
                    tooltip="Human-readable summary of typed tokens and context usage.",
                ),
                io.String.Output(
                    "text",
                    display_name="Prompt Echo",
                    tooltip="The input prompt (multiline string).",
                ),
            ],
            is_experimental=False,
        )

    @classmethod
    def _count_stream_prompt_tokens(cls, stream_batches: list[list[Any]]) -> int:
        """Count non-special tokens in a stream batch."""
        total = 0
        for batch in stream_batches:
            for token_item in batch:
                if isinstance(token_item, (tuple, list)) and len(token_item) >= 3:
                    word_id = token_item[2]
                    if isinstance(word_id, int) and word_id > 0:
                        total += 1
                elif isinstance(token_item, int):
                    total += 1
                else:
                    total += 1
        return total

    @classmethod
    def _stream_context_limit_tokens(cls, stream_batches: list[list[Any]]) -> int:
        """Count total tokens (including padding/special) in all batches."""
        return sum(len(batch) for batch in stream_batches)

    @classmethod
    @override
    def execute(
        cls,
        clip: Any,
        text: Optional[str] = None,
        count_strategy: str = "max_stream",
    ) -> io.NodeOutput:
        """
        Count prompt tokens and context window usage for a given text and CLIP object.
        Returns token count, context limit, chunk count, details, and prompt echo.
        """
        if clip is None:
            msg = "No CLIP input connected."
            logging.warning(f"FensTokenCounter: {msg}")
            return io.NodeOutput(0, 0, 0, msg, text or "")

        if not text or not text.strip():
            msg = "No prompt text provided."
            return io.NodeOutput(0, 0, 0, msg, text or "")

        try:
            # token_streams is a dict keyed by tokenizer branch names (e.g. l/g/t5xxl)
            token_streams = clip.tokenize(text, return_word_ids=True)
            if not isinstance(token_streams, dict) or not token_streams:
                msg = "Tokenizer returned no token streams."
                return io.NodeOutput(0, 0, 0, msg, text)

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
            elif count_strategy == "max_stream":
                token_count = max(prompt_counts)
                context_limit_tokens = max(context_limits)
                chunk_count = max(chunk_counts)
            else:
                logging.warning(
                    f"FensTokenCounter: Unknown count_strategy '{count_strategy}', using max_stream."
                )
                token_count = max(prompt_counts)
                context_limit_tokens = max(context_limits)
                chunk_count = max(chunk_counts)

            details = (
                f"Prompt tokens: {token_count} | "
                f"Context limit: {context_limit_tokens} | "
                f"Chunks: {chunk_count} | "
                f"Strategy: {count_strategy}"
            )

            # Optionally, provide a UI preview for details (uncomment if desired)
            # preview = ui.PreviewText(details)

            return io.NodeOutput(
                token_count,
                context_limit_tokens,
                chunk_count,
                details,
                text,
            )
        except Exception as e:
            msg = f"Error: {e}"
            logging.error(f"FensTokenCounter: Failed to tokenize text. {msg}")
            return io.NodeOutput(0, 0, 0, msg, text or "")
