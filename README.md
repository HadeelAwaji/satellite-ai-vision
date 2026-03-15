# 🛰️ SatelliteAI Vision

> AI-powered satellite and aerial image analysis — live on HuggingFace Spaces

⚠️ This project is currently a work in progress and will continue to be improved with additional features and optimizations.
[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-HuggingFace-yellow)](https://huggingface.co/spaces/SatelliteAI-Vision/satellite-ai-vision)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-6.8-orange)](https://gradio.app)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)](https://ultralytics.com)

---

## 🌍 Live Demo

**Try it now (no account needed):**
👉 https://huggingface.co/spaces/SatelliteAI-Vision/satellite-ai-vision

---

## 📌 About

SatelliteAI Vision is a personal AI project for analyzing satellite and aerial imagery. It began as a desktop Python/Tkinter prototype and was later rebuilt into a deployable web application with three AI-powered features accessible through a browser.

---

## ✨ Features

### 🔬 Super Resolution
Enhances low-resolution satellite images using the **EDSR (Enhanced Deep Super-Resolution)** neural network. Supports 2×, 3×, and 4× upscaling with automatic sharpening post-processing. Falls back to OpenCV bicubic interpolation if the AI model is unavailable.

### 📦 Object Detection
Detects objects in aerial imagery using **YOLOv8n**. Identifies vehicles (cars, trucks, buses), aircraft, boats, trains, and people with labeled bounding boxes and confidence scores. Color-coded by object class.

### 🗺️ AI Change Detection
Bi-temporal change analysis between two satellite images of the same area:
- **SSIM (Structural Similarity Index)** — measures perceptual similarity
- **Histogram-matched pixel differencing** — normalizes lighting/seasonal differences
- **Semantic region labeling** — classifies changes as vegetation, water, urban/built, bare soil, or general
- Outputs: change overlay, heatmap, and binary mask

### 🚀 Full Pipeline
Runs all three operations sequentially on a pair of images with a single click.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Web UI | Gradio 6.8 |
| Super-Resolution | EDSR via `super-image` |
| Object Detection | YOLOv8n via `ultralytics` |
| Change Detection | OpenCV + scikit-image SSIM |
| Image Processing | PIL, NumPy, OpenCV |
| Deployment | HuggingFace Spaces (free CPU) |

---

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/satellite-ai-vision
cd satellite-ai-vision

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open `http://localhost:7860` in your browser.

---

## 📁 Project Structure

```
satellite-ai-vision/
├── app.py              # Main Gradio app — all features and UI
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 💡 Design Decisions

- **Auto-resize** — any image uploaded (even 30MB+ satellite tiles) is automatically resized to 2000px max before processing to prevent memory crashes on free CPU
- **Progress tracking** — every operation shows live step-by-step progress messages
- **Graceful fallback** — if the EDSR AI model fails to load, the app falls back to OpenCV bicubic upscaling automatically
- **No login required** — fully public, anyone with the link can use it

---

## 📈 Project Journey

| Stage | Description |
|---|---|
| v1 | Desktop Tkinter app with basic contrast enhancement and SIFT-based alignment |
| v2 | Added AI super-resolution (EDSR), YOLOv8 object detection, SSIM change detection |
| v3 | Rebuilt as Gradio web app, deployed to HuggingFace Spaces |
| v3.1 | Added auto-resize, progress bars, friendly UX messages |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---
🌍 Potential Applications

Environmental monitoring

Urban expansion analysis

Disaster damage assessment

Infrastructure monitoring

Satellite imagery enhancement

---

Note: This project is a personal work and is not affiliated with any organization or institution.

---
*Built with ❤️ using Python, Gradio, and open-source AI models.*
