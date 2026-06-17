from __future__ import annotations

import logging
import re
from typing import Any

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

    EXPECTED_TOKEN_COUNT = 3
    MIN_WEIGHT_SEGMENT_LEN = 2  # Minimum length for weight syntax: "(x)"

    @classmethod
    def _escape_important(cls, text: str) -> str:
        """
        Escape special characters that would be interpreted as weight syntax.
        Used to protect literal parentheses from weight parsing.
        Converts \\( to marker \x00\x02 and \\) to marker \x00\x01
        """
        text = text.replace("\\)", "\x00\x01")
        text = text.replace("\\(", "\x00\x02")
        return text

    @classmethod
    def _unescape_important(cls, text: str) -> str:
        """Restore escaped parentheses from markers back to literal characters."""
        text = text.replace("\x00\x01", ")")
        text = text.replace("\x00\x02", "(")
        return text

    @classmethod
    def _parse_parentheses(cls, string: str) -> list[str]:
        """
        Parse a string into segments, respecting nested parentheses.
        Used to extract weight-syntax segments from the prompt.
        Returns list of segments like ["text", "(weighted:1.5)", "more text"]
        """
        result = []
        current_item = ""
        nesting_level = 0
        for char in string:
            if char == "(":
                if nesting_level == 0:
                    if current_item:
                        result.append(current_item)
                        current_item = "("
                    else:
                        current_item = "("
                else:
                    current_item += char
                nesting_level += 1
            elif char == ")":
                nesting_level -= 1
                if nesting_level == 0:
                    result.append(current_item + ")")
                    current_item = ""
                else:
                    current_item += char
            else:
                current_item += char
        if current_item:
            result.append(current_item)
        return result

    @classmethod
    def _token_weights(
        cls, string: str, current_weight: float = 1.0
    ) -> list[tuple[str, float]]:
        """
        Parse weight syntax from prompt text.
        (text:weight) syntax applies a multiplier to the text tokens.
        Returns list of (text, weight) tuples where weight is the final multiplier.
        Note: Weights don't add tokens, they modify embedding strength.
        """
        parsed = cls._parse_parentheses(string)
        out = []
        for segment in parsed:
            weight = current_weight
            if (
                len(segment) >= cls.MIN_WEIGHT_SEGMENT_LEN
                and segment[0] == "("
                and segment[-1] == ")"
            ):
                # Check for weight syntax like (text:1.5)
                inner = segment[1:-1]
                colon_idx = inner.rfind(":")
                if colon_idx > 0:
                    try:
                        weight = float(inner[colon_idx + 1 :])
                        text = inner[:colon_idx]
                        out.append((text, weight))
                    except ValueError:
                        # Malformed weight, treat whole thing as text
                        out.append((segment, current_weight))
                else:
                    # Just parentheses, no weight
                    out.append((inner, current_weight))
            else:
                out.append((segment, current_weight))
        return out

    @classmethod
    def _preprocess_prompt(cls, text: str) -> tuple[str, dict[str, Any]]:
        """
        Preprocess prompt to extract special syntax and information.

        Returns:
            tuple: (cleaned_text, analysis_dict) where analysis_dict contains:
                - break_count: Number of BREAK operations
                - has_escaped_parens: Whether escaped parens are present
                - special_functions: List of detected special functions
        """
        analysis = {
            "break_count": 0,
            "has_escaped_parens": False,
            "special_functions": [],
        }

        # Check for escaped parentheses
        if "\\(" in text or "\\)" in text:
            analysis["has_escaped_parens"] = True

        # Count BREAK operations - must be BREAK on its own (not "breaking" or "rebreak")
        # Uses word boundaries and checks context
        # Use lookbehind to avoid consuming whitespace and allow overlapping matches
        break_pattern = r"(?:^|(?<=\s))BREAK(?=\s|$|[,;:])"
        break_matches = re.findall(break_pattern, text, re.IGNORECASE | re.MULTILINE)
        analysis["break_count"] = len(break_matches)

        # Detect special functions
        # Functions can have optional parentheses: "TE()" or just "TE" alone as keyword
        special_functions = [
            "TE",
            "CAT",
            "AVG",
            "AND",
            "STYLE",
            "SDXL",
            "SHUFFLE",
            "SHIFT",
            "CUT",
        ]
        for func in special_functions:
            # Match function name followed by either ( or whitespace/punctuation/end
            pattern = rf"(?:^|\s|[,;(]){func}(?:\s*\(|(?:\s|$|[,;)]))"
            if re.search(pattern, text, re.IGNORECASE):
                analysis["special_functions"].append(func)

        # First escape important characters
        cleaned = cls._escape_important(text)

        # Then unescape for normal processing (we just needed to mark them)
        cleaned = cls._unescape_important(cleaned)

        return cleaned, analysis

    @classmethod
    def _count_stream_prompt_tokens(cls, stream_batches: list[list[Any]]) -> int:
        """
        Count non-special tokens in a stream batch.

        Each token in a batch is typically a tuple: (token_id, weight, word_id)
        We count entries with positive word_id to filter out special tokens
        like start/end/padding tokens (which have word_id <= 0).
        """
        total = 0
        for batch in stream_batches:
            for token_item in batch:
                if (
                    isinstance(token_item, (tuple, list))
                    and len(token_item) >= cls.EXPECTED_TOKEN_COUNT
                ):
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
    def _split_on_break(cls, text: str) -> list[str]:
        """
        Split text on BREAK operations and remove BREAK from segments.
        BREAK creates chunk boundaries but should not be counted as tokens.

        Returns:
            List of text segments split at BREAK boundaries.
        """
        # Split on BREAK with word boundaries
        break_pattern = r"(?:^|\s)BREAK(?=\s|$|[,;:])"
        segments = re.split(break_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        # Filter out empty segments
        return [seg.strip() for seg in segments if seg.strip()]

    @classmethod
    def _tokenize_break_segments(
        cls, clip: Any, segments: list[str]
    ) -> dict[str, list[list[Any]]]:
        """
        Tokenize each BREAK-separated segment independently and merge the
        resulting batches per stream, mirroring how BREAK is actually
        processed by ComfyUI's conditioning pipeline (each segment is
        tokenized and padded/chunked on its own, then concatenated).

        Tokenizing per-segment (rather than tokenizing the whole BREAK-joined
        text once and guessing at padding) gives correct results regardless
        of tokenizer family: fixed-window tokenizers (CLIP-style) get padded
        per segment exactly as the real encoder will pad them, and unbounded
        tokenizers (Qwen3/T5/Llama-style encoders with no fixed context
        window) simply contribute their real token count with no padding,
        since that's what they actually produce - no per-architecture
        special-casing required.

        Returns:
            Merged dict of {stream_name: [batch, batch, ...]} across all segments.
        """
        merged: dict[str, list[list[Any]]] = {}
        for segment in segments:
            if not segment:
                continue
            segment_streams = clip.tokenize(segment, return_word_ids=True)
            if not isinstance(segment_streams, dict):
                continue
            for stream_name, batches in segment_streams.items():
                merged.setdefault(stream_name, []).extend(batches)
        return merged

    @classmethod
    def _process_token_counts(
        cls,
        token_streams: dict[str, list[list[Any]]],
        count_strategy: str,
    ) -> tuple[int, int, int]:
        """
        Process token streams to get counts and chunks.

        Returns:
            Tuple of (token_count, context_limit_tokens, chunk_count)
        """
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
            if count_strategy != "max_stream":
                logging.warning(
                    "FensTokenCounter: Unknown count_strategy %s, using max_stream.",
                    count_strategy,
                )
            token_count = max(prompt_counts)
            context_limit_tokens = max(context_limits)
            chunk_count = max(chunk_counts)

        return token_count, context_limit_tokens, chunk_count

    @classmethod
    @override
    def execute(
        cls,
        clip: Any,
        text: str | None = None,
        count_strategy: str = "max_stream",
    ) -> io.NodeOutput:
        """
        Count prompt tokens and context window usage for a given text and CLIP object.

        Features:
        - Handles escape sequences (\\( and \\) for literal parentheses)
        - Accounts for weight syntax ((text:weight) - weights don't add tokens)
        - On BREAK, tokenizes each segment independently so padding/chunking
          matches what the active tokenizer actually does per segment
          (correct for both fixed-window encoders like CLIP and unbounded
          encoders like Qwen3/T5/Llama-style text encoders)
        - Supports multi-encoder models (SD1, SDXL, Flux, Anima, etc.)
        - Shows chunk count and context window usage

        Returns:
            tuple: (total_tokens, context_limit, chunk_count, details, text_echo)
        """
        if clip is None:
            msg = "No CLIP input connected."
            logging.warning("FensTokenCounter: %s", msg)
            return io.NodeOutput(0, 0, 0, msg, text or "")

        if not text or not text.strip():
            msg = "No prompt text provided."
            return io.NodeOutput(0, 0, 0, msg, text or "")

        try:
            # Preprocess to detect special syntax
            cleaned_text, analysis = cls._preprocess_prompt(text)
            break_count = analysis["break_count"]

            if break_count > 0:
                # Tokenize each BREAK-separated segment independently so
                # chunking/padding reflects what the tokenizer actually does
                # per segment, rather than guessing at a fixed-window size.
                segments = cls._split_on_break(cleaned_text)
                token_streams = cls._tokenize_break_segments(clip, segments)
            else:
                token_streams = clip.tokenize(cleaned_text, return_word_ids=True)

            if not isinstance(token_streams, dict) or not token_streams:
                msg = "Tokenizer returned no token streams."
                return io.NodeOutput(0, 0, 0, msg, text)

            # Get token counts and chunk information
            final_token_count, context_limit_tokens, chunk_count = (
                cls._process_token_counts(token_streams, count_strategy)
            )

            # Build output details
            details_parts = [
                f"Prompt tokens: {final_token_count}",
                f"Context limit: {context_limit_tokens}",
                f"Chunks: {chunk_count}",
                f"Strategy: {count_strategy}",
            ]

            if break_count > 0:
                details_parts.append(f"BREAK ops: {break_count}")
            if analysis["has_escaped_parens"]:
                details_parts.append("Has escaped parens: Yes")
            if analysis["special_functions"]:
                func_str = ", ".join(analysis["special_functions"])
                details_parts.append(f"Functions: {func_str}")

            details = " | ".join(details_parts)

            return io.NodeOutput(
                final_token_count,
                context_limit_tokens,
                chunk_count,
                details,
                text,
            )
        except (ValueError, TypeError) as e:
            msg = f"Error: {e}"
            logging.error("FensTokenCounter: Failed to tokenize text. %s", msg)
            return io.NodeOutput(0, 0, 0, msg, text or "")
        except Exception:
            raise
