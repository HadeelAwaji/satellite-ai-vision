import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageDraw
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim
import functools

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_PIXELS = 2000   # Auto-resize longest side to this on any upload

# ─── Model Loading (cached) ───────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def get_yolo_model():
    from ultralytics import YOLO
    return YOLO("yolov8m.pt")  # medium → أفضل دقة للكائنات الصغيرة مثل النباتات والمباني

@functools.lru_cache(maxsize=2)
def get_sr_model(scale=2):
    try:
        from super_image import EdsrModel
        model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
        model.eval()
        return model
    except Exception:
        return None

# ─── Helper Utilities ─────────────────────────────────────────────────────────

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv2_to_pil(arr):
    if arr.ndim == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def resize_for_display(img, max_size=800):
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return img

def auto_resize(img, label="Image"):
    """Resize image so longest side <= MAX_PIXELS. Returns (img, note_string)."""
    w, h = img.size
    if max(w, h) > MAX_PIXELS:
        ratio = MAX_PIXELS / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        note = (
            f"📏 {label} was {w}×{h}px — auto-resized to "
            f"{new_w}×{new_h}px so it runs smoothly on free CPU.\n"
        )
        return img, note
    return img, ""

# ─── 1. SUPER RESOLUTION ──────────────────────────────────────────────────────

def super_resolve(input_image, scale=2, progress=gr.Progress()):
    if input_image is None:
        return None, "❌ Please upload an image first."

    progress(0.0, desc="🔍 Checking image size...")
    input_image, resize_note = auto_resize(input_image, "Input")
    w, h = input_image.size
    info = []
    if resize_note:
        info.append(resize_note)
    info.append(f"📐 Processing size: {w}×{h}px")
    info.append("⏳ This runs on free CPU — please wait 20–60 seconds...\n")

    progress(0.2, desc="🤖 Loading AI super-resolution model...")
    model = get_sr_model(scale)

    progress(0.5, desc="✨ Upscaling image...")
    if model is not None:
        try:
            from super_image import ImageLoader
            import torch
            inputs = ImageLoader.load_image(input_image)
            with torch.no_grad():
                preds = model(inputs)
            ImageLoader.save_image(preds, "temp_sr.png")
            result = Image.open("temp_sr.png").convert("RGB")
            method = "🤖 EDSR AI Super-Resolution"
        except Exception:
            result = _cv2_upscale(input_image, scale)
            method = "🔧 OpenCV Bicubic (AI model unavailable)"
    else:
        result = _cv2_upscale(input_image, scale)
        method = "🔧 OpenCV Bicubic Upscaling"

    progress(0.85, desc="🔆 Sharpening output...")
    sharpened = _sharpen(result)
    nw, nh = sharpened.size

    progress(1.0, desc="✅ Done!")
    info += [
        f"✅ Output size: {nw}×{nh}px",
        f"🔍 Scale factor: {scale}×",
        f"⚙️  Method: {method}",
        "✨ Sharpening: Applied",
        "",
        "💡 Tip: 2× scale gives the fastest results on free CPU.",
    ]
    return sharpened, "\n".join(info)

def _cv2_upscale(img, scale):
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    up = cv2.resize(arr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(up)

def _sharpen(img):
    arr = np.array(img)
    k = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    return Image.fromarray(np.clip(cv2.filter2D(arr, -1, k), 0, 255).astype(np.uint8))

# ─── 2. OBJECT DETECTION ──────────────────────────────────────────────────────

LABEL_COLORS = {
    "car": (0, 200, 100), "truck": (0, 150, 255), "bus": (255, 150, 0),
    "boat": (0, 100, 255), "airplane": (200, 0, 200), "person": (255, 50, 50),
    "motorcycle": (50, 200, 200), "bicycle": (150, 255, 50), "train": (200, 100, 0),
}
DEFAULT_COLOR = (255, 255, 0)

def detect_objects(input_image, confidence=0.25, progress=gr.Progress()):
    if input_image is None:
        return None, "❌ Please upload an image first.", ""

    progress(0.0, desc="🔍 Preparing image...")
    input_image, resize_note = auto_resize(input_image, "Input")
    notes = []
    if resize_note:
        notes.append(resize_note)
    notes.append("⏳ Running YOLOv8 on free CPU — please wait 15–40 seconds...\n")

    progress(0.3, desc="📦 Loading YOLOv8 model...")
    model = get_yolo_model()

    progress(0.6, desc="🔎 Scanning for objects...")
    results   = model(np.array(input_image.convert("RGB")), conf=confidence)
    annotated = input_image.copy().convert("RGB")
    draw      = ImageDraw.Draw(annotated)
    counts, detections = {}, []

    for result in results:
        for box in result.boxes:
            cls_id     = int(box.cls[0])
            label      = model.names[cls_id]
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = LABEL_COLORS.get(label, DEFAULT_COLOR)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            tag = f"{label} {conf_score:.0%}"
            draw.rectangle([x1, y1 - 18, x1 + len(tag) * 7, y1], fill=color)
            draw.text((x1 + 3, y1 - 16), tag, fill=(0, 0, 0))
            counts[label] = counts.get(label, 0) + 1
            detections.append(f"• {label.capitalize()} — {conf_score:.1%} @ [{x1},{y1},{x2},{y2}]")

    progress(1.0, desc="✅ Done!")
    total = sum(counts.values())
    summary = notes + [f"🔍 Total objects detected: {total}", ""]
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        summary.append(f"  {cls.capitalize()}: {cnt}")
    summary += ["", "── Detailed Detections ──", ""] + (detections or ["No objects found."])
    summary.append("\n💡 Tip: Lower the confidence slider to detect more objects.")

    counts_str = "\n".join([f"{k}: {v}" for k, v in counts.items()]) or "None detected"
    return resize_for_display(annotated), "\n".join(summary), counts_str

# ─── 3. AI CHANGE DETECTION ───────────────────────────────────────────────────

CHANGE_CATEGORIES = {
    "vegetation":  ((34, 139, 34), 40),
    "water":       ((30, 100, 200), 40),
    "urban/built": ((180, 50, 50), 50),
    "bare soil":   ((150, 100, 50), 45),
    "general":     ((255, 200, 0), 30),
}

def detect_changes_ai(ref_image, tgt_image, threshold=30, show_overlay=True, progress=gr.Progress()):
    if ref_image is None or tgt_image is None:
        return None, None, None, "❌ Please upload both a Reference and a Target image."

    progress(0.0, desc="🔍 Preparing images...")
    notes = []
    ref_image, n1 = auto_resize(ref_image, "Reference")
    tgt_image, n2 = auto_resize(tgt_image, "Target")
    if n1: notes.append(n1)
    if n2: notes.append(n2)
    notes.append("⏳ Running change detection on free CPU — please wait 20–60 seconds...\n")

    progress(0.2, desc="🎨 Normalising colours...")
    ref_cv = pil_to_cv2(ref_image)
    tgt_cv = pil_to_cv2(tgt_image)
    if ref_cv.shape != tgt_cv.shape:
        tgt_cv = cv2.resize(tgt_cv, (ref_cv.shape[1], ref_cv.shape[0]), interpolation=cv2.INTER_CUBIC)
    tgt_matched = np.clip(
        match_histograms(tgt_cv.astype(float), ref_cv.astype(float), channel_axis=-1), 0, 255
    ).astype(np.uint8)

    progress(0.4, desc="📊 Computing difference map...")
    ref_gray = cv2.cvtColor(ref_cv,      cv2.COLOR_BGR2GRAY).astype(np.float32)
    tgt_gray = cv2.cvtColor(tgt_matched, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff_norm = cv2.normalize(cv2.absdiff(ref_gray, tgt_gray), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    progress(0.55, desc="🧠 Calculating SSIM similarity...")
    ssim_score, ssim_map = ssim(ref_gray, tgt_gray, full=True, data_range=255.0)
    ssim_diff = ((1 - ssim_map) * 127).clip(0, 255).astype(np.uint8)

    progress(0.65, desc="🗺️ Building change mask...")
    combined = cv2.addWeighted(diff_norm, 0.6, ssim_diff, 0.4, 0)
    _, binary_mask = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel)

    progress(0.78, desc="🏷️ Labelling change regions...")
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heat_img = Image.fromarray(cv2.applyColorMap(combined, cv2.COLORMAP_JET)[:, :, ::-1]).convert("RGB")
    overlay  = cv2_to_pil(ref_cv).convert("RGBA") if show_overlay else None
    annotated_mask  = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    category_counts = {k: 0 for k in CHANGE_CATEGORIES}

    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cat = _classify_change(ref_cv[y:y+h, x:x+w], tgt_matched[y:y+h, x:x+w])
        category_counts[cat] += 1
        color, _ = CHANGE_CATEGORIES[cat]
        cv2.drawContours(annotated_mask, [cnt], -1, color[::-1], 2)
        cv2.putText(annotated_mask, cat, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color[::-1], 1)

    if show_overlay and overlay is not None:
        mask_rgba = Image.fromarray(annotated_mask).convert("RGBA")
        mask_rgba.putalpha(160)
        overlay   = Image.alpha_composite(overlay, mask_rgba).convert("RGB")

    progress(0.95, desc="📋 Generating report...")
    changed_px = int(np.sum(binary_mask > 0))
    total_px   = binary_mask.size
    change_pct = changed_px / total_px * 100
    valid_contours = [c for c in contours if cv2.contourArea(c) >= 100]

    summary = notes + [
        f"📊 SSIM Score:     {ssim_score:.4f}  (1.0 = identical images)",
        f"🗺️  Changed Area:  {change_pct:.2f}% of image",
        f"🔢 Changed Pixels: {changed_px:,} / {total_px:,}",
        f"📦 Change Regions: {len(valid_contours)}",
        "",
        "── Change Categories ──",
    ]
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
        if cnt > 0:
            summary.append(f"  {cat.capitalize()}: {cnt} region(s)")

    if change_pct < 5:
        summary.append("\n✅ Low change — likely minor seasonal or lighting variation.")
    elif change_pct < 20:
        summary.append("\n⚠️  Moderate change — notable land cover differences detected.")
    else:
        summary.append("\n🚨 High change — significant transformation detected.")

    summary.append("\n💡 Tip: Adjust the sensitivity slider if the mask looks too noisy or sparse.")

    progress(1.0, desc="✅ Done!")
    return (
        resize_for_display(overlay or cv2_to_pil(annotated_mask)),
        resize_for_display(heat_img),
        resize_for_display(Image.fromarray(binary_mask)),
        "\n".join(summary),
    )

def _classify_change(roi_ref, roi_tgt):
    if roi_ref.size == 0 or roi_tgt.size == 0:
        return "general"
    def mean_bgr(a):
        return a.mean(axis=(0, 1)) if a.ndim == 3 else np.zeros(3)
    b, g, r = mean_bgr(roi_tgt)
    ndvi = (g - r) / (g + r + 1e-6)
    if ndvi > 0.1:                             return "vegetation"
    if b > r and b > g and b > 80:             return "water"
    if r > 120 and g > 100 and b > 80:         return "urban/built"
    if r > g and r > b and g > b:              return "bare soil"
    return "general"

# ─── 4. FULL PIPELINE ─────────────────────────────────────────────────────────

def full_pipeline(ref_image, tgt_image, sr_scale, conf, progress=gr.Progress()):
    if ref_image is None or tgt_image is None:
        return None, None, None, None, None, "❌ Please upload both a Reference and a Target image."

    logs = ["🚀 Starting Full Pipeline — 3 steps total...\n"]

    progress(0.05, desc="✨ Step 1/3 — Super-Resolving images...")
    ref_sr, ref_info = super_resolve(ref_image, sr_scale)
    tgt_sr, tgt_info = super_resolve(tgt_image, sr_scale)
    logs += ["── Step 1: Super-Resolution ──", ref_info, tgt_info, "✅ Done.\n"]

    progress(0.45, desc="🗺️ Step 2/3 — Detecting changes...")
    overlay, heatmap, mask, change_summary = detect_changes_ai(ref_sr, tgt_sr)
    logs += ["── Step 2: Change Detection ──", change_summary, "✅ Done.\n"]

    progress(0.78, desc="📦 Step 3/3 — Detecting objects...")
    detected, det_summary, _ = detect_objects(tgt_sr, conf)
    logs += ["── Step 3: Object Detection ──", det_summary, "✅ Done.\n"]

    progress(1.0, desc="🎉 Pipeline complete!")
    logs.append("🎉 Full pipeline finished successfully!")
    return ref_sr, tgt_sr, overlay, heatmap, detected, "\n".join(logs)

# ─── GRADIO UI ────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
body, .gradio-container {
    background: #0d0f14 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #e2e8f0 !important;
}
.gradio-container { max-width: 1200px !important; margin: auto; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
.gr-button-primary {
    background: linear-gradient(135deg, #00d4aa, #0099cc) !important;
    border: none !important; color: #0d0f14 !important;
    font-weight: 700 !important; font-family: 'Space Mono', monospace !important;
    letter-spacing: 0.05em !important; transition: all 0.2s ease !important;
}
.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 212, 170, 0.35) !important;
}
.tab-nav button { font-family: 'Space Mono', monospace !important; font-size: 0.85rem !important; color: #94a3b8 !important; }
.tab-nav button.selected { color: #00d4aa !important; border-bottom: 2px solid #00d4aa !important; }
footer { display: none !important; }
"""

NOTICE = """
> 💡 **Before you start:**
> - **Any image size accepted** — large files are automatically resized to 2000px max before processing
> - Supported formats: **JPG · PNG · WEBP**
> - Running on **free CPU** — each operation takes **20–60 seconds**, please be patient!
> - For the fastest experience keep images under **5MB** when possible
"""

def build_app():
    with gr.Blocks(css=CUSTOM_CSS, title="🛰️ SatelliteAI Vision") as demo:

        gr.Markdown("""
# 🛰️ SatelliteAI Vision
### AI-Powered Satellite & Aerial Image Analysis
*Super-Resolution · Change Detection · Object Detection*
---""")
        gr.Markdown(NOTICE)

        with gr.Tabs():

            # Tab 1 — Super Resolution
            with gr.Tab("🔬 Super Resolution"):
                gr.Markdown("### Enhance image clarity with AI upscaling (EDSR model)")
                with gr.Row():
                    with gr.Column():
                        sr_input = gr.Image(type="pil", label="📁 Upload Image  (any size · JPG/PNG/WEBP)")
                        sr_scale = gr.Radio([2, 3, 4], value=2, label="Upscale Factor")
                        sr_btn   = gr.Button("✨ Enhance with AI", variant="primary")
                    with gr.Column():
                        sr_output = gr.Image(label="🖼️ Enhanced Output")
                        sr_info   = gr.Textbox(label="📊 Processing Info", lines=8, interactive=False)
                sr_btn.click(super_resolve, [sr_input, sr_scale], [sr_output, sr_info])

            # Tab 2 — Object Detection
            with gr.Tab("📦 Object Detection"):
                gr.Markdown("### Detect vehicles, aircraft, boats & more using YOLOv8")
                with gr.Row():
                    with gr.Column():
                        od_input = gr.Image(type="pil", label="📁 Upload Satellite Image  (any size)")
                        od_conf  = gr.Slider(0.1, 0.9, value=0.25, step=0.05,
                                             label="Confidence Threshold  (lower = detect more objects)")
                        od_btn   = gr.Button("🔍 Detect Objects", variant="primary")
                    with gr.Column():
                        od_output  = gr.Image(label="🖼️ Annotated Result")
                        od_summary = gr.Textbox(label="📊 Detection Summary", lines=12, interactive=False)
                od_btn.click(detect_objects, [od_input, od_conf],
                             [od_output, od_summary, gr.Textbox(visible=False)])

            # Tab 3 — Change Detection
            with gr.Tab("🗺️ Change Detection"):
                gr.Markdown("### AI bi-temporal change analysis — upload two images of the same area at different times")
                with gr.Row():
                    cd_ref = gr.Image(type="pil", label="📁 Reference Image  (earlier / before)")
                    cd_tgt = gr.Image(type="pil", label="📁 Target Image  (later / after)")
                with gr.Row():
                    cd_thresh  = gr.Slider(10, 80, value=30, step=5,
                                           label="Change Sensitivity  (lower = detect more changes)")
                    cd_overlay = gr.Checkbox(value=True, label="Show change overlay on reference")
                    cd_btn     = gr.Button("🔎 Detect Changes", variant="primary")
                with gr.Row():
                    cd_overlay_out = gr.Image(label="🗺️ Change Overlay")
                    cd_heat        = gr.Image(label="🌡️ Change Heatmap")
                    cd_mask        = gr.Image(label="⬜ Binary Mask")
                cd_summary = gr.Textbox(label="📊 Analysis Report", lines=14, interactive=False)
                cd_btn.click(detect_changes_ai, [cd_ref, cd_tgt, cd_thresh, cd_overlay],
                             [cd_overlay_out, cd_heat, cd_mask, cd_summary])

            # Tab 4 — Full Pipeline
            with gr.Tab("🚀 Full Pipeline"):
                gr.Markdown("### Run all 3 steps at once — Super-Resolution → Change Detection → Object Detection")
                gr.Markdown("⚠️ Expect **2–5 minutes** on free CPU. Great for demos!")
                with gr.Row():
                    fp_ref = gr.Image(type="pil", label="📁 Reference Image  (before)")
                    fp_tgt = gr.Image(type="pil", label="📁 Target Image  (after)")
                with gr.Row():
                    fp_scale = gr.Radio([2, 3], value=2, label="SR Scale")
                    fp_conf  = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Detection Confidence")
                    fp_btn   = gr.Button("⚡ Run Full Pipeline", variant="primary")
                with gr.Row():
                    fp_ref_out = gr.Image(label="✨ SR Reference")
                    fp_tgt_out = gr.Image(label="✨ SR Target")
                    fp_overlay = gr.Image(label="🗺️ Change Overlay")
                    fp_heat    = gr.Image(label="🌡️ Heatmap")
                    fp_detect  = gr.Image(label="📦 Detections")
                fp_log = gr.Textbox(label="📋 Pipeline Log", lines=18, interactive=False)
                fp_btn.click(full_pipeline, [fp_ref, fp_tgt, fp_scale, fp_conf],
                             [fp_ref_out, fp_tgt_out, fp_overlay, fp_heat, fp_detect, fp_log])

        gr.Markdown("""
---
**SatelliteAI Vision** · EDSR · YOLOv8 · OpenCV · scikit-image · Free HuggingFace CPU
        """)
    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch()
