# DDPM Face Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **From pure noise to realistic faces. A PyTorch implementation of Denoising Diffusion Probabilistic Model (DDPM) trained on CelebA-HQ dataset.**

<div align="center">
  <img src="assets/diffusion_progression.gif" alt="Diffusion Progression" width="600"/>
  <br>
  <em>Reverse diffusion process: Pure noise → Face (200 steps)</em>
</div>

---

## 📖 Overview

This repository contains a complete implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** from scratch. DDPMs are the foundation behind modern generative AI systems like DALL-E, Stable Diffusion, and Midjourney. The model learns to reverse a gradual noising process, transforming random Gaussian noise into high-quality face images.

**Key highlights:**
- 🔥 Custom U-Net architecture with time embeddings and attention
- 🎨 Cosine noise scheduling for improved sample quality
- 💨 Mixed precision training for faster computation
- 📊 Quantitative evaluation (PSNR & SSIM)
- 🎮 Interactive Gradio web app for real-time generation

---

## 🏗️ Architecture

<div align="center">
  <img src="assets/unet_architecture.png" alt="U-Net Architecture" width="700"/>
</div>

The model uses a **U-Net** with:
- **Downsampling blocks** with residual connections
- **Attention mechanisms** at lower resolutions
- **Time step embeddings** to condition the model on noise level
- **Upsampling blocks** to reconstruct the image

---

## 📊 Results

### Progressive Denoising
The model starts with pure noise and gradually removes noise over 200 timesteps:

| Step 200 (Noise) | Step 150 | Step 100 | Step 50 | Step 0 (Output) |
|:----------------:|:--------:|:--------:|:-------:|:---------------:|
| ![noise](assets/noise.png) | ![step150](assets/step150.png) | ![step100](assets/step100.png) | ![step50](assets/step50.png) | ![output](assets/output.png) |

### Sample Generated Faces

<div align="center">
  <img src="assets/generated_samples.png" alt="Generated Samples" width="600"/>
</div>

### Quantitative Metrics

| Metric | Score |
|--------|-------|
| PSNR | 22.47 dB |
| SSIM | 0.784 |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory

### Setup

```bash
# Clone the repository
git clone https://github.com/minahil0018/ddpm-face-generation.git
cd ddpm-face-generation

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
