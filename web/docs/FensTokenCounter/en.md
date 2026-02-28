# FensTokenCounter

The **FensTokenCounter** node counts the number of tokens in a text prompt using a selected encoder model. It supports multiple encoder types and provides the token count for any prompt you enter.

## Parameters

- **Primary Encoder**
  - Select the encoder to use for tokenization. The dropdown options are populated from your local configuration (`encoder_mapping.py`).

- **Prompt Text**
  - The text prompt to count tokens for. Supports multiline input and dynamic prompts.

## Usage

1. Choose the desired encoder from the dropdown.
2. Enter your prompt in the text field.
3. The node outputs:
  - The total token count for the prompt using the selected encoder.
  - The input prompt text (for reference).

## Output

- **Token Count**
  - The number of tokens in the prompt, as counted by the selected encoder.

- **Prompt Text**
  - The input prompt text (multiline string).

## Notes

- Different encoders may tokenize the same text differently, resulting in different token counts.
- If no text is provided, the output will be 0.
- If an unknown encoder is selected, the output will be 0.
- The encoder list is based on your local configuration (`encoder_mapping.py`). Make sure your models are available and correctly mapped.
- The node caches loaded tokenizers for performance. If a tokenizer fails to load, an error will be logged and the output will be 0.
