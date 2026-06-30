# FensTokenCounter

The **FensTokenCounter** node counts typed prompt tokens and also shows the active tokenizer context window tier from the connected **ComfyUI CLIP** object.

## Parameters

- **CLIP**
  - Connect `CLIP` output (for example from a checkpoint/model loader).

- **Prompt Text**
  - The text prompt to count tokens for. Supports multiline input and dynamic prompts.

- **Count Strategy** *(Advanced)*
  - `max_stream`: Uses the largest tokenizer-branch count (recommended for multi-branch text encoders like SDXL/SD3).
  - `sum_streams`: Sums token counts across all tokenizer branches.

- **Show Token Breakdown** *(Advanced)*
  - When enabled, appends a per-token breakdown (token id, decoded text, weight, and word id) for each tokenizer branch to the **Details** output. Off by default since it can get long for longer prompts.

## Usage

1. Connect `CLIP` text encoder to the node.
2. Enter your prompt in the text field.
3. (Optional) Adjust advanced counting options.
4. The node outputs:

- Typed token count for the prompt.
- Current context-limit tier (for example 77 or 154).
- Number of tokenizer chunks/windows in use.
- The input prompt text (for reference).

## Output

- **Token Count**
  - The number of tokens in the prompt, as counted by the connected `CLIP` tokenizer(s).

- **Context Limit**
  - The active context window capacity tier in tokens (for example `77`, `154`, `231`).

- **Chunk Count**
  - How many tokenizer chunks/windows were needed for the prompt.

- **Details**
  - A readable summary including prompt tokens, context limit, chunk count, and strategy. Includes a per-token breakdown when Show Token Breakdown is enabled.

- **Prompt Text**
  - The input prompt text (multiline string).

## Notes

- Different text-encoder tokenizer branches may tokenize the same text differently, resulting in different counts.
- If no text is provided, the output will be 0.
- If the CLIP input is missing/invalid, numeric outputs are 0 and details explain why.
