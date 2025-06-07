<div align="center">
  
[![English](https://img.shields.io/badge/Languages-English-blue)](README.md)
[![License](https://img.shields.io/badge/License-GPL3.0-lightgreen)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![submit issue](https://img.shields.io/badge/Submit-issue-pink)](https://github.com/Taithrah/ComfyUI_Fens_Simple_Nodes/issues)

</div>

---

The aim of this repository is to offer a suite of custom nodes for ComfyUI that focus on simplicity and utility.

## Quickstart

1. Look for `Fens-Simple-Nodes` in ComfyUI-Manager. If you are installing manually, clone this repository under `ComfyUI/custom_nodes`. 
1. Restart ComfyUI.

# Features
- **Token Counter**
  - Simple to use with the ability to change encoders.
  - On par with webui a1111, forge, etc.

- **Optimal Empty Latent**
  - The tool will compute the closest compatible resolution to your target resolution or ratio, ensuring you get a output with minimal deformity (varies by checkpoint).
  - Specify your target aspect ratio in one of three formats: 
    - Pixel dimensions: `WxH` (e.g. `1920x1080`)
    - Ratio: `W:H` (e.g. `16:9`)
    - Decimal: (e.g. `1.7778`)

# Examples

### Token Counter

![TokenCount](https://raw.githubusercontent.com/Taithrah/ComfyUI_Fens_Simple_Nodes/refs/heads/main/examples/TokenCount.webp)

### WebUI-Forge Example for comparison
![ForgeCount](https://raw.githubusercontent.com/Taithrah/ComfyUI_Fens_Simple_Nodes/refs/heads/main/examples/ForgeCount.webp)

### Optimal Empty Latent
![OptimalEmptyLatent](https://raw.githubusercontent.com/Taithrah/ComfyUI_Fens_Simple_Nodes/refs/heads/main/examples/OptimalEmptyLatent.webp)
