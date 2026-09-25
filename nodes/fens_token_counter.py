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
                io.Boolean.Input(
                    "show_token_breakdown",
                    display_name="Show Token Breakdown",
                    default=False,
                    advanced=True,
                    tooltip="Append a per-token breakdown (id, decoded text, weight, word id) for each stream to the Details output.",
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
                    tooltip="Total padded slots in the active context window/floor (e.g. 77/154/231 for CLIP-style encoders). For unbounded encoders like T5XXL/Qwen3-family, this is just their minimum padding floor, not a hard ceiling. Only counts streams whose encoder is actually loaded - tokenizer-only streams with no backing model (e.g. Anima's unused t5xxl) are excluded automatically.",
                ),
                io.Int.Output(
                    "chunk_count",
                    display_name="Chunk Count",
                    tooltip="Number of tokenizer chunks/windows used for this prompt.",
                ),
                io.String.Output(
                    "details",
                    display_name="Details",
                    tooltip="Human-readable summary of typed tokens and context usage. Includes a per-token breakdown when Show Token Breakdown is enabled.",
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
    MIN_TOKEN_WEIGHT_TUPLE_LEN = 2  # Minimum length for a (token_id, weight) tuple
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
            # Match function name followed by either ( or whitespace/punctuation/end.
            # NOTE: intentionally case-sensitive (no re.IGNORECASE) - several of
            # these keywords (AND, CUT, SHIFT, STYLE) are also ordinary English
            # words, so case-insensitive matching false-positives on normal prose
            # (e.g. "her hair and uniform" was matching the AND function). Prompt
            # syntax functions are conventionally typed in caps, same as BREAK.
            pattern = rf"(?:^|\s|[,;(]){func}(?:\s*\(|(?:\s|$|[,;)]))"
            if re.search(pattern, text):
                analysis["special_functions"].append(func)

        # First escape important characters
        cleaned = cls._escape_important(text)

        # Then unescape for normal processing (we just needed to mark them)
        cleaned = cls._unescape_important(cleaned)

        return cleaned, analysis

    @classmethod
    def _resolve_special_token_ids(cls, sub_tokenizer: Any) -> set[int]:
        """
        Best-effort resolution of special/padding token ids for a given
        sub-tokenizer. Used as a fallback for streams that don't carry
        reliable per-token word_id metadata (e.g. some unbounded LLM-style
        encoders such as Qwen3/T5 may hand back (token_id, weight) pairs
        rather than the (token_id, weight, word_id) triples CLIP-style
        tokenizers use), so we still have some way to exclude BOS/EOS/pad
        tokens instead of silently counting everything as real content.

        ComfyUI's SDTokenizer wrapper (which Qwen3-/T5-style tokenizers are
        built on, e.g. Anima's qwen3_06b/t5xxl) stores these as
        `pad_token`/`start_token`/`end_token` - no "_id" suffix, and no
        separate bos/eos naming. We check those first, then also check the
        underlying HuggingFace tokenizer (`sub_tokenizer.tokenizer`) for its
        standard `*_token_id` attributes, in case a future/unfamiliar
        tokenizer wrapper doesn't follow the SDTokenizer shape.
        """
        ids: set[int] = set()
        if sub_tokenizer is None:
            return ids
        for attr in ("pad_token", "start_token", "end_token"):
            value = getattr(sub_tokenizer, attr, None)
            if isinstance(value, int):
                ids.add(value)
        hf_tokenizer = getattr(sub_tokenizer, "tokenizer", None)
        for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
            value = getattr(hf_tokenizer, attr, None)
            if isinstance(value, int):
                ids.add(value)
        return ids

    @classmethod
    def _count_stream_prompt_tokens(
        cls,
        stream_batches: list[list[Any]],
        special_ids: set[int] | None = None,
    ) -> int:
        """
        Count non-special tokens in a stream batch.

        Each token in a batch is typically a tuple: (token_id, weight, word_id)
        We count entries with positive word_id to filter out special tokens
        like start/end/padding tokens (which have word_id <= 0).

        Some tokenizer streams (e.g. Qwen3/T5-style unbounded encoders) may
        not populate word_id at all - in that case we can't distinguish real
        tokens from padding/special tokens by word_id, so we fall back to
        excluding only ids we can positively identify as special via
        `special_ids` (resolved from the tokenizer's pad/bos/eos attributes).
        This avoids the previous behavior of unconditionally counting every
        non-triple token entry as real content, which silently included
        padding/special tokens whenever a stream's tuples didn't match the
        3-element (token_id, weight, word_id) shape.
        """
        special_ids = special_ids or set()
        total = 0
        for batch in stream_batches:
            for token_item in batch:
                if (
                    isinstance(token_item, (tuple, list))
                    and len(token_item) >= cls.EXPECTED_TOKEN_COUNT
                ):
                    token_id, word_id = token_item[0], token_item[2]
                    if isinstance(word_id, int) and word_id > 0:
                        total += 1
                    elif (
                        word_id is None
                        and isinstance(token_id, int)
                        and token_id not in special_ids
                    ):
                        # No word_id metadata for this stream - fall back to
                        # identifying specials by id instead of counting blind.
                        total += 1
                elif (
                    isinstance(token_item, (tuple, list))
                    and len(token_item) >= cls.MIN_TOKEN_WEIGHT_TUPLE_LEN
                ):
                    token_id = token_item[0]
                    if isinstance(token_id, int) and token_id not in special_ids:
                        total += 1
                elif isinstance(token_item, int):
                    if token_item not in special_ids:
                        total += 1
        return total

    @classmethod
    def _stream_context_limit_tokens(cls, stream_batches: list[list[Any]]) -> int:
        """Count total tokens (including padding/special) in all batches."""
        return sum(len(batch) for batch in stream_batches)

    @classmethod
    def _resolve_sub_tokenizer(cls, clip: Any, stream_name: str) -> Any | None:
        """
        Find the underlying per-stream tokenizer object for a given stream
        name (e.g. "l", "g", "t5xxl", "qwen3_06b"), so its vocabulary can be
        used to decode token ids back to text.

        ComfyUI's tokenizer wrapper classes store each sub-tokenizer as an
        attribute, but the attribute naming convention differs by class:
        some wrappers (e.g. SD1/SDXL/Flux's CLIP streams) prefix it with
        "clip_" (stream "l" -> attribute "clip_l"), while others (e.g.
        Anima's qwen3_06b/t5xxl) use the stream name directly with no
        prefix. Both are real, currently-used ComfyUI conventions, so we
        just try both rather than guessing a single one.
        """
        tokenizer = getattr(clip, "tokenizer", None)
        if tokenizer is None:
            return None
        for attr_name in (stream_name, f"clip_{stream_name}"):
            sub_tokenizer = getattr(tokenizer, attr_name, None)
            if sub_tokenizer is not None:
                return sub_tokenizer
        return None

    @classmethod
    def _resolve_sub_encoder_model(
        cls, clip: Any, stream_name: str
    ) -> tuple[Any | None, bool]:
        """
        Find the underlying per-stream *encoder model* submodule for a given
        stream name (as opposed to _resolve_sub_tokenizer, which finds the
        tokenizer). This is used to tell whether a stream is actually backed
        by loaded weights.

        ComfyUI's tokenizer wrapper classes are cheap and typically
        instantiate every sub-tokenizer they know about unconditionally
        (e.g. Anima's tokenizer wrapper always builds both a qwen3_06b and
        a t5xxl tokenizer). The encoder *model* classes, however, only
        build a submodule for an encoder if the checkpoint actually
        contains weights for it (this is how multi-encoder architectures
        like SD3/Flux work: which of clip_l/clip_g/t5xxl get built depends
        on what's present in the loaded state dict). So checking the model
        side rather than the tokenizer side tells us which streams are
        real vs. tokenizer-only scaffolding with nothing backing them.

        Returns:
            (sub_model_or_None, model_lookup_succeeded) - the second value
            is False only when we couldn't find any cond_stage_model object
            to inspect at all (unfamiliar CLIP wrapper shape), so callers
            can distinguish "confirmed absent" from "couldn't check."
        """
        cond_stage_model = getattr(clip, "cond_stage_model", None)
        if cond_stage_model is None:
            patcher = getattr(clip, "patcher", None)
            cond_stage_model = getattr(patcher, "model", None) if patcher else None
        if cond_stage_model is None:
            return None, False
        for attr_name in (stream_name, f"clip_{stream_name}"):
            sub_model = getattr(cond_stage_model, attr_name, None)
            if sub_model is not None:
                return sub_model, True
        return None, True

    @classmethod
    def _filter_active_streams(
        cls, clip: Any, token_streams: dict[str, list[list[Any]]]
    ) -> tuple[dict[str, list[list[Any]]], list[str]]:
        """
        Drop streams whose encoder model isn't actually loaded, so
        tokenizer-only streams (e.g. Anima's t5xxl, which is tokenized but
        never has a corresponding model loaded since Anima checkpoints only
        ship qwen3_06b weights) don't get counted as if they mattered.

        If we can't find a cond_stage_model to inspect at all (unfamiliar
        CLIP wrapper shape), we leave every stream in place rather than
        guessing - it's safer to include an extra stream than to silently
        drop a real one.

        Returns:
            (filtered_token_streams, excluded_stream_names)
        """
        active_streams: dict[str, list[list[Any]]] = {}
        excluded: list[str] = []
        for stream_name, stream_batches in token_streams.items():
            sub_model, model_lookup_succeeded = cls._resolve_sub_encoder_model(
                clip, stream_name
            )
            if sub_model is not None or not model_lookup_succeeded:
                # Real encoder found, or we couldn't inspect the model at
                # all - keep the stream rather than risk dropping a real one.
                active_streams[stream_name] = stream_batches
            else:
                excluded.append(stream_name)

        if not active_streams:
            # Filtering removed everything (shouldn't normally happen) -
            # fall back to the original unfiltered streams rather than
            # returning an empty result.
            return token_streams, []

        return active_streams, excluded

    @classmethod
    def _decode_token_id(cls, sub_tokenizer: Any, token_id: Any) -> str | None:
        """
        Decode a single token id back to its text using the sub-tokenizer's
        vocabulary, if available. Returns None if it can't be resolved
        (e.g. unsupported tokenizer object, or the id isn't an int - it
        could be a raw embedding tensor for custom/textual-inversion
        embeddings, which has no vocab entry).
        """
        decoded_text: str | None = None
        if sub_tokenizer is not None and isinstance(token_id, int):
            inv_vocab = getattr(sub_tokenizer, "inv_vocab", None)
            if inv_vocab is not None:
                token_str = inv_vocab.get(token_id)
                if isinstance(token_str, str):
                    decoded_text = token_str
            if decoded_text is None:
                decode_fn = getattr(sub_tokenizer, "decode", None)
                if callable(decode_fn):
                    try:
                        decoded = decode_fn([token_id])
                    except Exception:
                        decoded = None
                    if isinstance(decoded, str):
                        decoded_text = decoded
                    elif (
                        isinstance(decoded, list)
                        and decoded
                        and isinstance(decoded[0], str)
                    ):
                        decoded_text = decoded[0]
        return decoded_text

    @classmethod
    def _build_token_breakdown(
        cls,
        clip: Any,
        token_streams: dict[str, list[list[Any]]],
        excluded_streams: list[str] | None = None,
    ) -> str:
        """
        Build a human-readable per-token breakdown for every stream: each
        real (non-padding) token's id, decoded text, weight, and word id.
        Streams in excluded_streams are labeled as not contributing to the
        totals (no loaded encoder backing them), so it's clear why a
        stream shown here doesn't affect Prompt tokens/Context limit.
        """
        excluded_streams = excluded_streams or []
        lines = []
        for stream_name, stream_batches in token_streams.items():
            sub_tokenizer = cls._resolve_sub_tokenizer(clip, stream_name)
            label = (
                f"[{stream_name}] (excluded - no loaded encoder)"
                if stream_name in excluded_streams
                else f"[{stream_name}]"
            )
            lines.append(label)
            position = 0
            for batch in stream_batches:
                for token_item in batch:
                    if (
                        isinstance(token_item, (tuple, list))
                        and len(token_item) >= cls.EXPECTED_TOKEN_COUNT
                    ):
                        token_id, weight, word_id = (
                            token_item[0],
                            token_item[1],
                            token_item[2],
                        )
                    elif (
                        isinstance(token_item, (tuple, list))
                        and len(token_item) >= cls.MIN_TOKEN_WEIGHT_TUPLE_LEN
                    ):
                        token_id, weight, word_id = token_item[0], token_item[1], None
                    else:
                        token_id, weight, word_id = token_item, 1.0, None

                    if isinstance(word_id, int) and word_id <= 0:
                        position += 1
                        continue  # skip special/padding tokens in the breakdown

                    decoded = cls._decode_token_id(sub_tokenizer, token_id)
                    decoded_str = repr(decoded) if decoded is not None else "?"
                    lines.append(
                        f"  [{position}] id={token_id} text={decoded_str} weight={weight} word_id={word_id}"
                    )
                    position += 1
        return "\n".join(lines)

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
        clip: Any,
        token_streams: dict[str, list[list[Any]]],
    ) -> tuple[int, int, int]:
        """
        Process token streams to get counts and chunks.

        Aggregates via max across streams: parallel multi-encoder
        architectures (e.g. SD3/Flux's clip_l/clip_g/t5xxl) each encode the
        SAME full prompt text independently, so summing would just count
        identical content multiple times - max reflects the actual binding
        constraint (the stream that fills up first). Streams with no
        backing encoder model (tokenizer-only scaffolding, e.g. Anima's
        unused t5xxl) are filtered out before this runs.

        Returns:
            Tuple of (token_count, context_limit_tokens, chunk_count)
        """
        prompt_counts = [
            cls._count_stream_prompt_tokens(
                stream_batches,
                cls._resolve_special_token_ids(
                    cls._resolve_sub_tokenizer(clip, stream_name)
                ),
            )
            for stream_name, stream_batches in token_streams.items()
        ]
        context_limits = [
            cls._stream_context_limit_tokens(stream_batches)
            for stream_batches in token_streams.values()
        ]
        chunk_counts = [
            len(stream_batches) for stream_batches in token_streams.values()
        ]

        token_count = max(prompt_counts)
        context_limit_tokens = max(context_limits)
        chunk_count = max(chunk_counts)

        return token_count, context_limit_tokens, chunk_count

    @classmethod
    def _build_details_summary(
        cls,
        final_token_count: int,
        context_limit_tokens: int,
        chunk_count: int,
        analysis: dict[str, Any],
        excluded_streams: list[str],
    ) -> str:
        """Build the human-readable details summary line."""
        details_parts = [
            f"Prompt tokens: {final_token_count}",
            f"Context limit: {context_limit_tokens}",
            f"Chunks: {chunk_count}",
        ]
        if excluded_streams:
            details_parts.append(
                f"Excluded (no loaded encoder): {', '.join(excluded_streams)}"
            )

        break_count = analysis["break_count"]
        if break_count > 0:
            details_parts.append(f"BREAK ops: {break_count}")
        if analysis["has_escaped_parens"]:
            details_parts.append("Has escaped parens: Yes")

        special_functions = analysis["special_functions"]
        if special_functions:
            func_str = ", ".join(special_functions)
            details_parts.append(f"Functions: {func_str}")

        return " | ".join(details_parts)

    @classmethod
    @override
    def execute(
        cls,
        clip: Any,
        text: str | None = None,
        show_token_breakdown: bool = False,
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
        - Optionally appends a per-token breakdown (id, decoded text,
          weight, word id) per stream to the Details output

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

            # Drop streams with no backing encoder model (tokenizer-only
            # scaffolding, e.g. Anima's unused t5xxl) so they don't get
            # counted as if they contributed to conditioning.
            active_streams, excluded_streams = cls._filter_active_streams(
                clip, token_streams
            )

            # Get token counts and chunk information
            final_token_count, context_limit_tokens, chunk_count = (
                cls._process_token_counts(clip, active_streams)
            )

            details = cls._build_details_summary(
                final_token_count,
                context_limit_tokens,
                chunk_count,
                analysis,
                excluded_streams,
            )

            if show_token_breakdown:
                breakdown = cls._build_token_breakdown(
                    clip, token_streams, excluded_streams
                )
                if breakdown:
                    details = f"{details}\n\nToken breakdown:\n{breakdown}"

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
