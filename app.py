import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json

# ---- Page config ----
st.set_page_config(
    page_title="CropGuard AI",
    page_icon="🌿",
    layout="wide"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .hero { text-align: center; padding: 2rem 0; }
    .hero h1 { font-size: 3rem; color: #2ecc71; margin-bottom: 0; }
    .hero p { font-size: 1.2rem; color: #888; }
    .result-card {
        background: #1e2530;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2ecc71;
    }
    .disease-card {
        background: #1e2530;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #e74c3c;
    }
    .stat-box {
        background: #1e2530;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-number { font-size: 2rem; font-weight: bold; color: #2ecc71; }
    .stat-label { color: #888; font-size: 0.9rem; }
    .upload-section {
        border: 2px dashed #2ecc71;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .footer { text-align: center; color: #444; padding: 2rem; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ---- Treatment database ----
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
}

def get_treatment(class_name):
    return treatments.get(class_name, "Consult your local Krishi Vigyan Kendra for specific treatment advice.")

# ---- Load model ----
@st.cache_resource
def load_model():
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512), nn.ReLU(),
        nn.Dropout(0.4), nn.Linear(512, 38)
    )
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    return model, class_names

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---- Header ----
st.markdown("""
<div class="hero">
    <h1>🌿 CropGuard AI</h1>
    <p>AI-powered crop disease detection for Indian farmers | 99.57% Accuracy</p>
</div>
""", unsafe_allow_html=True)

# ---- Stats row ----
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-box"><div class="stat-number">99.57%</div><div class="stat-label">Accuracy</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-box"><div class="stat-number">38</div><div class="stat-label">Disease Classes</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-box"><div class="stat-number">87K</div><div class="stat-label">Training Images</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="stat-box"><div class="stat-number">14</div><div class="stat-label">Crop Types</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ---- Main layout ----
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 📸 Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a clear photo of a single leaf",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

with right:
    st.markdown("### 🔬 Analysis Results")
    if not uploaded_file:
        st.info("👈 Upload a leaf image to get instant disease detection results.")
        st.markdown("**Supported crops:**")
        st.markdown("🍎 Apple · 🌽 Corn · 🍇 Grape · 🥔 Potato · 🍅 Tomato · and more")
    else:
        model, class_names = load_model()
        with st.spinner("🔍 Analysing your crop..."):
            tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, 1)[0]
                top3 = torch.topk(probs, 3)

        predicted_class = class_names[top3.indices[0].item()]
        confidence = top3.values[0].item() * 100
        parts = predicted_class.split('___')
        crop = parts[0].replace('_', ' ')
        disease = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
        is_healthy = 'healthy' in predicted_class.lower()

        # Result
        if is_healthy:
            st.success(f"✅ **{crop}** plant is **Healthy!**")
        else:
            st.error(f"⚠️ **{disease}** detected in **{crop}**!")

        # Metrics
        m1, m2 = st.columns(2)
        m1.metric("🌱 Crop", crop)
        m2.metric("🦠 Status", "Healthy ✅" if is_healthy else "Diseased ⚠️")

        # Confidence
        st.markdown(f"**Confidence: {confidence:.1f}%**")
        st.progress(confidence / 100)

        # Treatment
        st.markdown("### 💊 Treatment Advice")
        treatment = get_treatment(predicted_class)
        if is_healthy:
            st.success(f"🌱 {treatment}")
        else:
            st.warning(f"💡 {treatment}")

        # Top 3
        with st.expander("📊 See Top 3 Predictions"):
            for i in range(3):
                cls = class_names[top3.indices[i].item()]
                prob = top3.values[i].item() * 100
                parts2 = cls.split('___')
                label = f"{parts2[0].replace('_',' ')} — {parts2[1].replace('_',' ') if len(parts2)>1 else ''}"
                st.progress(prob / 100, text=f"{label}: {prob:.1f}%")

# ---- Footer ----
st.markdown("---")
st.markdown("""
<div class="footer">
    Built with ❤️ using ResNet-50 + PyTorch | Trained on PlantVillage Dataset<br>
    For farmers across India 🇮🇳 | Krishi Vigyan Kendra Partner Ready
</div>
""", unsafe_allow_html=True)