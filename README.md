<div align="center">

[![English](https://img.shields.io/badge/Languages-English-blue)](README.md) [![License](https://img.shields.io/badge/License-GPL3.0-lightgreen)](https://www.gnu.org/licenses/gpl-3.0.en.html) [![Submit issue](https://img.shields.io/badge/Submit-issue-pink)](https://github.com/Taithrah/ComfyUI_Fens_Simple_Nodes/issues)

</div>

---

# Fens Simple Nodes for ComfyUI

**A suite of custom nodes designed for simplicity, flexibility, and power.**

## How to Use

1. **Install:**  
   - Find `Fens-Simple-Nodes` in ComfyUI-Manager, **or**  
   - Clone this repo into your `ComfyUI/custom_nodes` folder.

2. **Restart ComfyUI.**

3. **Drag the nodes into your workflow and enjoy!**

## What’s Included

- **Token Counter:**
  Count typed prompt tokens and see your current context window tier using your connected text encoder.

- **Optimal Empty Latent:**
  Quickly get the perfect image size for your model and aspect ratio.  
  - Enter aspect ratio as `16:9`, `1920x1080`, or even `1.7778`
  - Let the node pick the best resolution for your model, or set your own.
  - Easily swap between portrait and landscape.
  - Presets for SD1, SD2, SDXL, and more.
  - Supports batch generation.

- **Anima LoRA Remap:**
  Anima 2.9B and 3.8B are bigger, expanded versions of the base Anima 2B model. A LoRA trained on the smaller model doesn't line up correctly on the bigger one by default.
  It still loads and runs, but with no warning that it's landing on the wrong layers, usually giving weaker, off or no result at all. These node fix that automatically.
  - Add the node right after your model loader, and **before** any LoRA loader or LoRA stack node so it just passes the model through, ready for LoRAs to be applied correctly downstream.
  - Two versions, both for LoRAs trained on **Anima 2B** — one to use with the 2.9B model, and one to use with the 3.8B model.
  - Works with whatever LoRA loader or stacker you already use, so need to switch nodes.

## Screenshots

**Token Counter Example:**  
![TokenCount](https://raw.githubusercontent.com/Taithrah/ComfyUI_Fens_Simple_Nodes/refs/heads/main/examples/TokenCount.webp)

**WebUI-Forge Token count for comparison:**  
![ForgeCount](https://raw.githubusercontent.com/Taithrah/ComfyUI_Fens_Simple_Nodes/refs/heads/main/examples/ForgeCount.webp)

**Optimal Empty Latent Example:**  
![OptimalEmptyLatent](https://raw.githubusercontent.com/Taithrah/ComfyUI_Fens_Simple_Nodes/refs/heads/main/examples/OptimalEmptyLatent.webp)
