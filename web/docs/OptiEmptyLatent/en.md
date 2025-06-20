# OptiEmptyLatent Node

The **OptiEmptyLatent** node calculates and generates an empty latent tensor with an optimal resolution for a given aspect ratio and megapixel (MP) target, tailored for different model types in ComfyUI. It supports both automatic (optimized) and manual resolution modes.

## Parameters

- **dimensions**:  
  Input aspect ratio or resolution.

  - Formats:
    - `W:H` (e.g. `16:9`)
    - `WxH` (e.g. `1280x720`)
    - Decimal (e.g. `1.777`)
  - Use `WxH` when using exact resolution mode.

- **invert**:  
  Boolean. If enabled, swaps width and height (inverts aspect ratio, e.g. `16:9` → `9:16`).

- **optimization**:  
  Boolean.

  - `TRUE`: Automatically calculates the best resolution for your aspect ratio and model.
  - `FALSE`: Uses your provided resolution (must be in `WxH` or `W:H` format).

- **latent_alignment**:  
  Model preset. Selects the optimization preset for the model type (e.g. SDXL, SD1.5).

  - Each preset has its own block size, channel count, and recommended aspect ratio range.

- **batch_size**:  
  Integer. Number of latent images in the batch (higher values increase VRAM usage).

## Usage

1. **Optimized Mode (default):**

   - Enter your desired aspect ratio (e.g. `16:9`, `1.33`, etc.).
   - Select the model preset (`latent_alignment`).
   - The node will output a latent tensor with the optimal resolution for your settings.

2. **Exact Resolution Mode:**

   - Set **optimization** to `FALSE`.
   - Enter your desired resolution in `WxH` or `W:H` format (e.g. `1024x768`).
   - The node will output a latent tensor with your specified resolution, **rounded to the nearest multiple of the model's block size** (see Notes).

3. **Invert Aspect Ratio:**
   - Enable **invert** to swap width and height.

## Output

- **latent**:  
  The generated empty latent tensor (shape: `(batch_size, channels, height//8, width//8)`).

- **width**:  
  The width (in pixels) of the generated latent.

- **height**:  
  The height (in pixels) of the generated latent.

## Notes

- The node automatically clamps aspect ratios to the recommended range for the selected model.
- When using exact resolution mode (optimization = FALSE), the width and height you enter will be rounded to the nearest multiple of the model's block size. This ensures compatibility with the model, so the final resolution may differ slightly from what you entered.
- Details about the chosen resolution, aspect ratio, and model are shown in the UI output.
- For best results, use optimized mode unless you need a specific resolution.

## Example

| Parameter        | Value           |
| ---------------- | --------------- |
| dimensions       | `16:9`          |
| invert           | `FALSE`         |
| optimization     | `TRUE`          |
| latent_alignment | `SDXL (1024px)` |
| batch_size       | `1`             |

_Output:_

- Optimized Resolution: `1216x704 px`
- Aspect Ratio: `1.727`
- Target MP: `0.86`, Actual MP: `0.857`
- Block Size: `64`, Channels: `4`
- Model: `SDXL (1024px)`
