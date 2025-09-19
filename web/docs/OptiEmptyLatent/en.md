# OptiEmptyLatent

The **OptiEmptyLatent** node calculates and generates an empty latent tensor with an optimal resolution for a given aspect ratio and megapixel (MP) target, tailored for different model types in ComfyUI. It supports both automatic (optimized) and manual (exact) resolution modes.

## Parameters

- **Dimensions**  
  Specify the aspect ratio or resolution.
  - Formats:
    - `W:H` (e.g. `16:9`)
    - `WxH` (e.g. `1280x720`)
    - Decimal (e.g. `1.777`)
  - Use `WxH` when using exact resolution mode.

- **Invert**  
  If enabled, swaps width and height (inverts aspect ratio, e.g. `16:9` → `9:16`).

- **Optimization**  
  - `TRUE`: Automatically calculates the best resolution for your aspect ratio and model.
  - `FALSE`: Uses your provided resolution (must be in `WxH` or `W:H` format).

- **Latent Alignment**  
  Model preset. Selects the optimization preset for the model type (e.g. SDXL, SD1.5, Custom).
  - Each preset has its own block size, channel count, and recommended aspect ratio range.

- **Batch Size**  
  Number of latent images in the batch (higher values increase VRAM usage).

- **Block Size**  
  The minimum multiple for width/height. Only used when 'Custom' is selected.

- **Target MP**  
  Target megapixels. Only used when 'Custom' is selected.

## Usage

1. **Optimized Mode (default):**
   - Enter your desired aspect ratio (e.g. `16:9`, `1.33`, etc.).
   - Select the model preset (`Latent Alignment`).
   - The node will output a latent tensor with the optimal resolution for your settings.

2. **Exact Resolution Mode:**
   - Set **Optimization** to `FALSE`.
   - Enter your desired resolution in `WxH` or `W:H` format (e.g. `1024x768`).
   - The node will output a latent tensor with your specified resolution, rounded to the nearest multiple of the model's block size.

3. **Invert Aspect Ratio:**
   - Enable **Invert** to swap width and height.

## Output

- **Latent**  
  The generated empty latent tensor (shape: `(batch_size, channels, height//8, width//8)`).

- **Width**  
  The width (in pixels) of the generated latent.

- **Height**  
  The height (in pixels) of the generated latent.

- **Block Size**  
  The block size used for the calculation.

- **Details**  
  Information about the chosen resolution, aspect ratio, and model.

## Notes

- The node automatically clamps aspect ratios to the recommended range for the selected model.
- When using exact resolution mode (optimization = FALSE), the width and height you enter will be rounded to the nearest multiple of the model's block size. This ensures compatibility with the model, so the final resolution may differ slightly from what you entered.
- Details about the chosen resolution, aspect ratio, and model are shown in the UI output.
- For best results, use optimized mode unless you need a specific resolution.

## Example

| Parameter        | Value           |
| ---------------- | --------------- |
| Dimensions       | `16:9`          |
| Invert           | `FALSE`         |
| Optimization     | `TRUE`          |
| Latent Alignment | `SDXL (1024px)` |
| Batch Size       | `1`             |

_Output:_

- Optimized Resolution: `1216x704 px`
- Aspect Ratio: `1.727`
- Target MP: `0.86`, Actual MP: `0.857`
- Block Size: `64`, Channels: `4`
- Model: `SDXL (1024px)`
