# DDPM Face Generation

> From pure noise to realistic faces. A PyTorch implementation of Denoising Diffusion Probabilistic Model (DDPM) trained on CelebA-HQ.

---

## 📖 Overview

This repository contains a complete implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** from scratch. The model learns to reverse a gradual noising process, transforming random Gaussian noise into high-quality face images.

**Key Features:**
- Custom U-Net architecture with time embeddings and attention
- Cosine noise scheduling for improved sample quality  
- Mixed precision training for faster computation
- Quantitative evaluation (PSNR & SSIM)
- Interactive Gradio web app for real-time generation

---

## 📊 Results

### Progressive Denoising Process

Starting from pure noise, the model gradually removes noise over 200 timesteps to generate realistic faces. The forward process adds noise to real images, and the reverse process learns to denoise.

### Sample Generated Faces

The model generates diverse, realistic faces with proper facial structure, skin textures, and variations in hair and features.

### Quantitative Metrics

| Metric | Score |
|--------|-------|
| PSNR | 22.47 dB |
| SSIM | 0.784 |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/minahil0018/ddpm-face-generation.git
cd ddpm-face-generation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
