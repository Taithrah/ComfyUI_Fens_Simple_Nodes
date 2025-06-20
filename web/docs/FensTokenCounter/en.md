# FensTokenCounter

This node counts the number of tokens in a text prompt using a selected encoder model.

## Parameters

- **primary_encoder**:  
  Select the encoder to use for tokenization.
- **text**:  
  The text prompt to count tokens for. Supports multiline input and dynamic prompts.

## Usage

1. Choose the desired encoder from the dropdown.
2. Enter your prompt in the text field.
3. The node outputs the total token count for the prompt using the selected encoder.

## Output

- **total_tokens**:  
  The number of tokens in the prompt, as counted by the selected encoder.

## Notes

- Different encoders may tokenize the same text differently, resulting in different token counts.
- If no text is provided, the output will be 0.
- If an unknown encoder is selected, a warning is issued and the output will be 0.
