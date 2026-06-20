import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import base64
import numpy as np
import cv2

st.set_page_config(page_title="CropGuard AI", page_icon="🌿", layout="wide")

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64_image("bg.jpg")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*, *::before, *::after {{ font-family: 'Inter', sans-serif; box-sizing: border-box; }}
.stApp {{
    background-image: url("data:image/jpg;base64,{bg}");
    background-size: cover; background-position: center; background-attachment: fixed;
}}
.stApp::before {{
    content: ''; position: fixed; inset: 0;
    background: linear-gradient(160deg, rgba(240,248,240,0.82) 0%, rgba(230,245,235,0.78) 40%, rgba(220,238,248,0.80) 100%);
    z-index: 0; backdrop-filter: blur(1px);
}}
[data-testid="stAppViewContainer"] > .main > .block-container {{
    position: relative; z-index: 1; max-width: 1120px; padding-top: 0 !important;
}}
section[data-testid="stSidebar"] {{ display: none; }}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stHorizontalBlock"] {{
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 0 !important; min-height: 0 !important;
}}
[data-testid="stHorizontalBlock"] > div {{
    background: transparent !important; border: none !important;
    box-shadow: none !important; min-height: 0 !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploader"] > div > div:nth-child(2) {{ display: none !important; }}
.hero {{ text-align: center; padding: 1.8rem 0 0.8rem; }}
.hero-badge {{
    display: inline-block; background: rgba(80,140,100,0.12);
    border: 1px solid rgba(80,140,100,0.30); color: #3a7a52;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em;
    padding: 0.25rem 1rem; border-radius: 999px; margin-bottom: 0.7rem; text-transform: uppercase;
}}
.hero h1 {{ font-size: 2.8rem; font-weight: 700; color: #1a3a28; margin: 0.2rem 0 0.3rem; letter-spacing: -0.03em; text-shadow: 0 2px 20px rgba(255,255,255,0.8); }}
.hero h1 .g {{ color: #2e7d52; }}
.hero p {{ font-size: 0.95rem; color: #4a6858; margin-top: 0.2rem; }}
.stat-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 0.65rem; margin: 0.8rem 0; }}
.stat-card {{
    background: rgba(185,225,195,0.97); border: 1.5px solid rgba(60,120,80,0.45);
    border-top: 2px solid rgba(60,130,80,0.65); border-radius: 14px; padding: 0.85rem;
    text-align: center; box-shadow: 0 4px 16px rgba(0,80,40,0.10);
}}
.stat-number {{ font-size: 1.7rem; font-weight: 700; color: #1a4a28; }}
.stat-label {{ color: #2a4a30; font-size: 0.65rem; margin-top: 0.3rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }}
.divider {{ border: none; border-top: 1px solid rgba(80,140,100,0.18); margin: 0.7rem 0; }}
.glass-panel {{
    background: rgba(185,225,195,0.97); border: 1.5px solid rgba(60,120,80,0.55);
    border-top: 2px solid rgba(60,130,80,0.75); border-radius: 18px;
    padding: 1.3rem; box-shadow: 0 8px 32px rgba(0,80,40,0.18); margin-bottom: 1rem;
}}
.panel-title {{ font-size: 0.85rem; font-weight: 700; color: #0e3018; margin-bottom: 0.8rem; padding-bottom: 0.5rem; border-bottom: 2px solid rgba(40,100,60,0.35); }}
.metric-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin: 0.6rem 0; }}
.metric-card {{ background: rgba(140,195,120,0.55); border: 1.5px solid rgba(60,120,50,0.50); border-top: 2px solid rgba(60,120,50,0.70); border-radius: 12px; padding: 0.75rem 1rem; }}
.metric-label {{ font-size: 0.68rem; font-weight: 700; color: #1a3a10; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.25rem; }}
.metric-value {{ font-size: 1.3rem; font-weight: 800; color: #0d2008; }}
.disease-banner {{ background: rgba(220,80,80,0.08); border: 1px solid rgba(200,80,80,0.22); border-left: 3px solid #d47070; border-radius: 0 10px 10px 0; padding: 0.8rem 1rem; margin-bottom: 0.8rem; }}
.healthy-banner {{ background: rgba(60,160,100,0.12); border: 1px solid rgba(60,160,100,0.28); border-left: 3px solid #4aaa78; border-radius: 0 10px 10px 0; padding: 0.8rem 1rem; margin-bottom: 0.8rem; }}
.banner-title {{ font-size: 0.98rem; font-weight: 700; color: #1a3228; }}
.banner-sub {{ font-size: 0.76rem; color: #2a4a20; margin-top: 0.2rem; font-weight: 600; }}
.treatment-box {{ background: rgba(160,200,120,0.35); border: 1px solid rgba(100,140,60,0.32); border-left: 3px solid #6aaa40; border-radius: 0 10px 10px 0; padding: 0.85rem 1rem; margin-top: 0.8rem; }}
.treatment-title {{ font-size: 0.65rem; color: #1a3010; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }}
.treatment-text {{ font-size: 0.82rem; color: #1a3010; line-height: 1.65; font-weight: 600; }}
.pred-label {{ font-size: 0.82rem; font-weight: 700; color: #1e3a10; margin-bottom: 0.15rem; display: block; }}
.placeholder-box {{ background: rgba(185,225,195,0.60); border: 1.5px dashed rgba(60,120,80,0.40); border-radius: 14px; padding: 2.5rem 2rem; text-align: center; margin-top: 1rem; }}
.placeholder-text {{ font-size: 0.80rem; font-weight: 600; color: #1e4028; }}
[data-testid="stFileUploader"] label, [data-testid="stFileUploader"] label p {{ color: #1a3a10 !important; font-weight: 700 !important; font-size: 0.94rem !important; }}
.steps-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 0.65rem; margin: 0.5rem 0 1rem; }}
.step-card {{ background: rgba(185,225,195,0.97); border: 1.5px solid rgba(60,120,80,0.35); border-top: 2px solid rgba(60,120,80,0.55); border-radius: 14px; padding: 0.9rem; box-shadow: 0 4px 14px rgba(0,80,40,0.10); }}
.step-number {{ font-size: 1.3rem; font-weight: 700; color: rgba(46,100,60,0.35); line-height: 1; }}
.step-title {{ font-size: 0.82rem; font-weight: 700; color: #1e4228; margin: 0.3rem 0 0.2rem; }}
.step-desc {{ font-size: 0.73rem; color: #1e4028; line-height: 1.55; font-weight: 500; }}
.crops-grid {{ display: grid; grid-template-columns: repeat(5,1fr); gap: 0.5rem; margin: 0.5rem 0; }}
.crop-pill {{ background: rgba(185,225,195,0.97); border: 1.5px solid rgba(60,120,80,0.30); border-radius: 10px; padding: 0.55rem 0.3rem; text-align: center; font-size: 0.73rem; color: #1e4228; font-weight: 600; box-shadow: 0 2px 8px rgba(0,80,40,0.08); }}
.crop-emoji {{ font-size: 1.1rem; display: block; margin-bottom: 0.18rem; }}
.section-label {{ font-size: 0.65rem; font-weight: 700; color: #1e3a18; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.6rem; }}
.gradcam-note {{ font-size: 0.72rem; color: #2a4a20; font-weight: 600; margin-top: 0.5rem; text-align: center; font-style: italic; }}
.footer {{ text-align: center; color: #1e3a18; padding: 1.2rem; font-size: 0.72rem; line-height: 1.9; border-top: 1px solid rgba(80,140,100,0.18); margin-top: 1rem; font-weight: 500; }}
</style>
""", unsafe_allow_html=True)

treatments = {
    "Apple___Apple_scab": "Apply fungicides containing captan or myclobutanil. Remove and destroy infected leaves.",
    "Apple___Black_rot": "Prune infected branches. Apply copper-based fungicide every 7-10 days.",
    "Apple___Cedar_apple_rust": "Apply fungicide at pink stage. Remove nearby cedar trees if possible.",
    "Apple___healthy": "No treatment needed. Continue regular care.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply strobilurin fungicides. Rotate crops annually.",
    "Corn_(maize)___Common_rust_": "Apply fungicides early. Use resistant varieties next season.",
    "Corn_(maize)___Northern_Leaf_Blight": "Apply fungicide at first sign. Ensure good air circulation.",
    "Corn_(maize)___healthy": "No treatment needed. Continue regular care.",
    "Grape___Black_rot": "Apply mancozeb or myclobutanil. Remove mummified berries.",
    "Grape___Esca_(Black_Measles)": "Prune infected wood. Apply wound sealant after pruning.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply copper fungicide. Improve vineyard air circulation.",
    "Grape___healthy": "No treatment needed. Continue regular care.",
    "Potato___Early_blight": "Apply chlorothalonil or mancozeb. Remove infected leaves immediately.",
    "Potato___Late_blight": "Apply metalaxyl fungicide urgently. Destroy infected plants.",
    "Potato___healthy": "No treatment needed. Continue regular care.",
    "Tomato___Bacterial_spot": "Apply copper-based bactericide. Avoid overhead irrigation.",
    "Tomato___Early_blight": "Apply fungicide containing chlorothalonil. Remove lower infected leaves.",
    "Tomato___Late_blight": "Apply metalaxyl immediately. Remove and destroy infected plants.",
    "Tomato___Leaf_Mold": "Improve ventilation. Apply fungicide containing chlorothalonil.",
    "Tomato___Septoria_leaf_spot": "Apply mancozeb or chlorothalonil. Remove infected leaves.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Apply miticide or neem oil. Increase humidity around plants.",
    "Tomato___Target_Spot": "Apply fungicide. Ensure proper plant spacing for airflow.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Remove infected plants. Control whitefly population with insecticide.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants. Disinfect tools. Control aphids.",
    "Tomato___healthy": "No treatment needed. Continue regular care.",
    "Strawberry___Leaf_scorch": "Remove infected leaves. Apply fungicide containing captan. Avoid overhead watering.",
    "Strawberry___healthy": "No treatment needed. Continue regular care.",
    "Peach___Bacterial_spot": "Apply copper-based bactericide in early spring. Prune infected branches.",
    "Peach___healthy": "No treatment needed. Continue regular care.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur-based fungicide. Ensure good air circulation.",
    "Cherry_(including_sour)___healthy": "No treatment needed. Continue regular care.",
    "Blueberry___healthy": "No treatment needed. Continue regular care.",
    "Pepper,_bell___Bacterial_spot": "Apply copper bactericide. Avoid overhead irrigation.",
    "Pepper,_bell___healthy": "No treatment needed. Continue regular care.",
    "Soybean___healthy": "No treatment needed. Continue regular care.",
    "Squash___Powdery_mildew": "Apply potassium bicarbonate or neem oil. Improve air circulation.",
    "Raspberry___healthy": "No treatment needed. Continue regular care.",
    "Orange___Haunglongbing_(Citrus_greening)": "No cure available. Remove infected trees. Control psyllid insects.",
}

def get_treatment(cls):
    return treatments.get(cls, "Consult your local Krishi Vigyan Kendra for specific treatment advice.")

def pbar(pct, h=8):
    return (
        '<div style="background:rgba(60,120,80,0.18);border-radius:999px;'
        f'height:{h}px;overflow:hidden;margin:0.25rem 0 0.6rem;">'
        f'<div style="width:{pct:.1f}%;background:#2e7d52;height:100%;border-radius:999px;"></div>'
        '</div>'
    )

@st.cache_resource
def load_model():
    with open('class_names.json') as f:
        class_names = json.load(f)
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 38)
    )
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    return model, class_names

# ── Grad-CAM ────────────────────────────────────────────────────────────────
def generate_gradcam(model, tensor, class_idx):
    """
    Generates a Grad-CAM heatmap for the given image tensor and predicted class.
    Returns a PIL Image of the heatmap overlaid on the original image.
    """
    gradients = []
    activations = []

    # Hook into the last conv layer (layer4) to capture gradients & activations
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_f = model.layer4.register_forward_hook(forward_hook)
    handle_b = model.layer4.register_full_backward_hook(backward_hook)

    # Forward pass WITH gradients enabled
    output = model(tensor)
    model.zero_grad()
    # Backprop only for the predicted class
    output[0, class_idx].backward()

    handle_f.remove()
    handle_b.remove()

    # Pool gradients across channels
    grads = gradients[0].detach()          # (1, C, H, W)
    acts  = activations[0].detach()        # (1, C, H, W)
    weights = grads.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

    # Weighted combination of activation maps
    cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
    cam = torch.relu(cam)
    cam = cam.squeeze().numpy()

    # Normalise to 0-255
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = (cam * 255).astype(np.uint8)

    # Resize to match input image size (224x224)
    cam_resized = cv2.resize(cam, (224, 224))

    # Apply colour map and overlay on original image
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Convert original tensor back to numpy image
    orig = tensor.squeeze().permute(1, 2, 0).numpy()
    orig = orig * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    orig = np.clip(orig * 255, 0, 255).astype(np.uint8)

    # Blend heatmap with original image
    overlay = cv2.addWeighted(orig, 0.55, heatmap, 0.45, 0)
    return Image.fromarray(overlay)
# ────────────────────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

model, class_names = load_model()

st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered · ResNet-50 · PlantVillage</div>
    <h1>Crop<span class="g">Guard</span> AI</h1>
    <p>Instant crop disease detection for Indian farmers — snap, upload, protect.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-grid">
    <div class="stat-card"><div class="stat-number">99.57%</div><div class="stat-label">Accuracy</div></div>
    <div class="stat-card"><div class="stat-number">38</div><div class="stat-label">Disease Classes</div></div>
    <div class="stat-card"><div class="stat-number">87K</div><div class="stat-label">Training Images</div></div>
    <div class="stat-card"><div class="stat-number">14</div><div class="stat-label">Crop Types</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload a clear leaf photo (JPG or PNG)", type=["jpg","jpeg","png"])

image = None
if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
    except Exception as e:
        st.error(f"Could not read image: {e}")

if not image:
    st.markdown('<div class="placeholder-box"><div style="font-size:1.6rem;margin-bottom:0.4rem;">🌿</div><div class="placeholder-text">Upload a leaf photo above — your preview and analysis will appear here.</div></div>', unsafe_allow_html=True)
else:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="glass-panel"><div class="panel-title">📸 Leaf Preview</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        with st.spinner("🔍 Analysing your crop..."):
            tensor = transform(image).unsqueeze(0)

            # Prediction (no_grad for speed)
            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, 1)[0]
                top3 = torch.topk(probs, 3)

            pred_idx = top3.indices[0].item()
            pred  = class_names[pred_idx]
            conf  = top3.values[0].item() * 100
            parts = pred.split('___')
            crop  = parts[0].replace('_', ' ')
            disease = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
            healthy = 'healthy' in pred.lower()

            # Generate Grad-CAM (needs gradients, so run separately)
            gradcam_img = generate_gradcam(model, tensor, pred_idx)

        if healthy:
            banner = (
                '<div class="healthy-banner">'
                '<div class="banner-title">✅ ' + crop + ' is Healthy</div>'
                '<div class="banner-sub">No disease detected. Keep up the good care!</div>'
                '</div>'
            )
        else:
            banner = (
                '<div class="disease-banner">'
                '<div class="banner-title">⚠️ ' + disease + '</div>'
                '<div class="banner-sub">Detected in ' + crop + ' plant</div>'
                '</div>'
            )

        metrics = (
            '<div class="metric-row">'
            '<div class="metric-card"><div class="metric-label">🌱 Crop</div>'
            '<div class="metric-value">' + crop + '</div></div>'
            '<div class="metric-card"><div class="metric-label">🎯 Confidence</div>'
            '<div class="metric-value">' + f'{conf:.1f}%' + '</div></div>'
            '</div>'
        )

        conf_bar = pbar(conf, 9)

        treatment = (
            '<div class="treatment-box">'
            '<div class="treatment-title">💊 Treatment Advice</div>'
            '<div class="treatment-text">' + get_treatment(pred) + '</div>'
            '</div>'
        )

        top3_rows = ''
        for i in range(3):
            cls  = class_names[top3.indices[i].item()]
            prob = top3.values[i].item() * 100
            p2   = cls.split('___')
            lbl  = p2[0].replace('_', ' ') + ' — ' + (p2[1].replace('_', ' ') if len(p2) > 1 else '')
            top3_rows += '<span class="pred-label">' + lbl + ': ' + f'{prob:.1f}%' + '</span>' + pbar(prob, 7)

        top3_block = (
            '<details style="margin-top:0.8rem;">'
            '<summary style="cursor:pointer;font-size:0.82rem;font-weight:700;color:#1e3a08;'
            'padding:0.5rem;background:rgba(140,190,100,0.30);border-radius:8px;list-style:none;">'
            '📊 Top 3 Predictions</summary>'
            '<div style="padding:0.6rem 0.2rem 0.2rem;">' + top3_rows + '</div>'
            '</details>'
        )

        html = (
            '<div class="glass-panel">'
            '<div class="panel-title">🔬 Analysis Results</div>'
            + banner + metrics + conf_bar + treatment + top3_block +
            '</div>'
        )

        st.markdown(html, unsafe_allow_html=True)

    # Grad-CAM section below both columns
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel"><div class="panel-title">🔥 Grad-CAM — Where the AI Looked</div>', unsafe_allow_html=True)
    st.image(gradcam_img, use_container_width=True)
    st.markdown('<div class="gradcam-note">🔴 Red/warm areas = parts of the leaf the AI focused on most to make its prediction.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)
st.markdown("""
<div class="steps-grid">
    <div class="step-card"><div class="step-number">01</div><div class="step-title">📸 Take a clear photo</div><div class="step-desc">Close-up of a single leaf in good natural lighting. Avoid blurry or dark images.</div></div>
    <div class="step-card"><div class="step-number">02</div><div class="step-title">⬆️ Upload the image</div><div class="step-desc">Upload your JPG or PNG above. Our ResNet-50 model processes it instantly.</div></div>
    <div class="step-card"><div class="step-number">03</div><div class="step-title">💊 Get treatment advice</div><div class="step-desc">Receive instant diagnosis with confidence score and actionable treatment steps.</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Supported Crops</div>', unsafe_allow_html=True)
st.markdown("""
<div class="crops-grid">
    <div class="crop-pill"><span class="crop-emoji">🍎</span>Apple</div>
    <div class="crop-pill"><span class="crop-emoji">🌽</span>Corn</div>
    <div class="crop-pill"><span class="crop-emoji">🍇</span>Grape</div>
    <div class="crop-pill"><span class="crop-emoji">🥔</span>Potato</div>
    <div class="crop-pill"><span class="crop-emoji">🍅</span>Tomato</div>
    <div class="crop-pill"><span class="crop-emoji">🍑</span>Peach</div>
    <div class="crop-pill"><span class="crop-emoji">🍒</span>Cherry</div>
    <div class="crop-pill"><span class="crop-emoji">🫐</span>Blueberry</div>
    <div class="crop-pill"><span class="crop-emoji">🌶️</span>Pepper</div>
    <div class="crop-pill"><span class="crop-emoji">🍓</span>Strawberry</div>
    <div class="crop-pill"><span class="crop-emoji">🌿</span>Soybean</div>
    <div class="crop-pill"><span class="crop-emoji">🎃</span>Squash</div>
    <div class="crop-pill"><span class="crop-emoji">🍊</span>Orange</div>
    <div class="crop-pill"><span class="crop-emoji">🍇</span>Raspberry</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Built with ❤️ using ResNet-50 + PyTorch · Trained on PlantVillage Dataset (87K images)<br>
    For farmers across India 🇮🇳 · Kisan Call Centre: 1800-180-1551 (Free, 24/7)
</div>
""", unsafe_allow_html=True)
