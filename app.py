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
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

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
.hero {{ text-align: center; padding: 1rem 0 0.8rem; }}
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

/* Olive Green Styling for Language Dropdown Label */
.lang-container div[data-testid="stWidgetLabel"] p {{
    color: #2e7d52 !important;
    font-weight: 700 !important;
}}
</style>
""", unsafe_allow_html=True)

# Complete Multilingual Translations Mapping
I18N = {
    "English": {
        "title_pre": "Crop", "title_post": "Guard AI",
        "badge_text": "AI-Powered · ResNet-50 · PlantVillage",
        "hero_sub": "Instant crop disease detection for Indian farmers — snap, upload, protect.",
        "num_acc": "99.57%", "num_cls": "38", "num_imgs": "87K", "num_crops": "14",
        "stat_acc": "Accuracy", "stat_cls": "Disease Classes", "stat_imgs": "Training Images", "stat_crops": "Crop Types",
        "uploader_lbl": "📂 Upload a clear leaf photo (JPG or PNG)",
        "placeholder": "Upload a leaf photo above — your preview and analysis will appear here.",
        "preview_title": "📸 Leaf Preview", "analysis_title": "🔬 Analysis Results",
        "crop_lbl": "🌱 Crop", "conf_lbl": "🎯 Confidence", "treatment_lbl": "💊 Treatment Advice",
        "top3_lbl": "📊 Top 3 Predictions", "gradcam_title": "🔥 Grad-CAM — Where the AI Looked",
        "gradcam_desc": "🔴 Red/warm areas = parts of the leaf the AI focused on most to make its prediction.",
        "how_works": "How it works", "supported_crops": "Supported Crops",
        "step1_num": "01", "step1_t": "📸 Take a clear photo", "step1_d": "Close-up of a single leaf in good natural lighting. Avoid blurry or dark images.",
        "step2_num": "02", "step2_t": "⬆️ Upload the image", "step2_d": "Upload your JPG or PNG above. Our ResNet-50 model processes it instantly.",
        "step3_num": "03", "step3_t": "💊 Get treatment advice", "step3_d": "Receive instant diagnosis with confidence score and actionable treatment steps.",
        "healthy_sub": "No disease detected. Keep up the good care!", "healthy_title": "Healthy",
        "footer_text": "Built with ❤️ using ResNet-50 + PyTorch · Trained on PlantVillage Dataset (87K images)<br>For farmers across India 🇮🇳 · Kisan Call Centre: 1800-180-1551 (Free, 24/7)",
        "fallback_advice": "Consult your local Krishi Vigyan Kendra for specific treatment advice."
    },
    "हिन्दी": {
        "title_pre": "क्रॉप", "title_post": "गार्ड AI",
        "badge_text": "एआई-संचालित · ResNet-50 · प्लांटविलेज",
        "hero_sub": "भारतीय किसानों के लिए तत्काल फसल रोग की पहचान - फोटो लें, अपलोड करें, सुरक्षित करें।",
        "num_acc": "९९.५७%", "num_cls": "३८", "num_imgs": "८७ हजार", "num_crops": "१४",
        "stat_acc": "सटीकता", "stat_cls": "रोग श्रेणियां", "stat_imgs": "प्रशिक्षण छवियां", "stat_crops": "फसलों के प्रकार",
        "uploader_lbl": "📂 पत्ती की एक साफ फोटो अपलोड करें (JPG या PNG)",
        "placeholder": "ऊपर पत्ती की फोटो अपलोड करें — आपका पूर्वावलोकन और विश्लेषण यहां दिखाई देगा।",
        "preview_title": "📸 पत्ती का पूर्वावलोकन", "analysis_title": "🔬 विश्लेषण के परिणाम",
        "crop_lbl": "🌱 फसल", "conf_lbl": "🎯 विश्वास स्कोर", "treatment_lbl": "💊 उपचार सलाह",
        "top3_lbl": "📊 शीर्ष 3 संभावित अनुमान", "gradcam_title": "🔥 Grad-CAM — एआई ने कहां देखा",
        "gradcam_desc": "🔴 लाल/गर्म क्षेत्र = पत्ती के वे हिस्से जिन पर एआई ने सटीक भविष्यवाणी करने के लिए सबसे अधिक ध्यान केंद्रित किया।",
        "how_works": "यह कैसे काम करता है", "supported_crops": "समर्थित फसलें",
        "step1_num": "०१", "step1_t": "📸 साफ फोटो लें", "step1_d": "अच्छी प्राकृतिक रोशनी में एक पत्ती की पास से तस्वीर लें। धुंधली या अंधेरी छवियों से बचें।",
        "step2_num": "०२", "step2_t": "⬆️ फोटो अपलोड करें", "step2_d": "अपनी JPG या PNG फाइल ऊपर अपलोड करें। हमारा ResNet-50 मॉडल तुरंत प्रोसेस करेगा।",
        "step3_num": "०३", "step3_t": "💊 उपचार सलाह प्राप्त करें", "step3_d": "सटीकता स्कोर और व्यावहारिक उपचार चरणों के साथ तत्काल निदान प्राप्त करें।",
        "healthy_sub": "कोई बीमारी नहीं मिली। अपनी फसलों की अच्छी देखभाल जारी रखें!", "healthy_title": "स्वस्थ है",
        "footer_text": "ResNet-50 + PyTorch के साथ ❤️ से निर्मित · प्लांटविलेज डेटासेट (87K इमेज) पर प्रशिक्षित<br>भारतीय किसानों के लिए 🇮🇳 · किसान कॉल सेंटर: 1800-180-1551 (निःशुल्क, 24/7)",
        "fallback_advice": "विशिष्ट उपचार सलाह के लिए अपने स्थानीय कृषि विज्ञान केंद्र से संपर्क करें।"
    },
    "मराठी": {
        "title_pre": "क्रॉप", "title_post": "गार्ड एआई",
        "badge_text": "एआय-चालित · ResNet-50 · प्लांटव्हिलेज",
        "hero_sub": "भारतीय शेतकऱ्यांसाठी त्वरित पीक रोग शोधक यंत्रणा — फोटो काढा, अपलोड करा, संरक्षण करा.",
        "num_acc": "९९.५७%", "num_cls": "३८", "num_imgs": "८७ हजार", "num_crops": "१४",
        "stat_acc": "अचूकता", "stat_cls": "रोगांचे प्रकार", "stat_imgs": "एकूण चित्रे", "stat_crops": "पिकांचे प्रकार",
        "uploader_lbl": "📂 पानाचा स्पष्ट फोटो अपलोड करा (JPG किंवा PNG)",
        "placeholder": "वर पानाचा फोटो अपलोड करा — तुमचा प्रिव्ह्यू आणि विश्लेषण येथे दिसेल.",
        "preview_title": "📸 पानाचा प्रिव्ह्यू", "analysis_title": "🔬 विश्लेषणाचा निकाल",
        "crop_lbl": "🌱 पीक", "conf_lbl": "🎯 आत्मविश्वास", "treatment_lbl": "💊 उपचाराचा सल्ला",
        "top3_lbl": "📊 टॉप ३ संभाव्य अंदाज", "gradcam_title": "🔥 Grad-CAM — एआई (AI) ने कुठे पाहिले",
        "gradcam_desc": "🔴 लाल/उबदार भाग = पानाचे ते भाग ज्यावर AI ने अंदाज लावण्यासाठी सर्वात जास्त लक्ष केंद्रित केले.",
        "how_works": "हे कसे कार्य करते", "supported_crops": "समर्थित पिके",
        "step1_num": "०१", "step1_t": "📸 स्पष्ट फोटो काढा", "step1_d": "चांगल्या नैसर्गिक प्रकाशात एकाच पानाचा जवळून फोटो घ्या. अस्पष्ट किंवा गडद फोटो टाळा.",
        "step2_num": "०२", "step2_t": "⬆️ इमेज अपलोड करा", "step2_d": "तुमचा JPG किंवा PNG फोटो वर अपलोड करा. आमचे ResNet-50 मॉडेल त्वरित प्रक्रिया करेल.",
        "step3_num": "०३", "step3_t": "💊 उपचाराचा सल्ला मिळवा", "step3_d": "अचूकता टक्केवारी आणि आवश्यक उपचारांच्या चरणांसह त्वरित निदान मिळवा.",
        "healthy_sub": "कोणताही रोग आढळला नाही. पिकाची अशीच उत्तम काळजी घेत राहा!", "healthy_title": "निरोगी आहे",
        "footer_text": "ResNet-50 + PyTorch वापरून ❤️ ने बनवलेले · प्लांटविलेज डेटासेटवर (87K चित्रे) प्रशिक्षित<br>भारतातील शेतकऱ्यांसाठी 🇮🇳 · किसान कॉल सेंटर: 1800-180-1551 (मोफत, २४/७)",
        "fallback_advice": "विशिष्ट उपचारांच्या सल्ल्यासाठी तुमच्या स्थानिक कृषी विज्ञान केंद्राशी संपर्क साधा।"
    }
}

# Crop and Disease Text Maps
CROP_TRANSLATIONS = {
    "Apple": {"English": "Apple", "हिन्दी": "सेब", "मराठी": "सफरचंद"},
    "Corn (maize)": {"English": "Corn (Maize)", "हिन्दी": "मक्का", "मराठी": "मका"},
    "Grape": {"English": "Grape", "हिन्दी": "अंगूर", "मराठी": "द्राक्षे"},
    "Potato": {"English": "Potato", "हिन्दी": "आलू", "मराठी": "बटाटा"},
    "Tomato": {"English": "Tomato", "हिन्दी": "टमाटर", "मराठी": "टोमॅटो"},
    "Strawberry": {"English": "Strawberry", "हिन्दी": "स्ट्रॉबेरी", "मराठी": "स्ट्रॉबेरी"},
    "Peach": {"English": "Peach", "हिन्दी": "आड़ू", "मराठी": "पीड (Peach)"},
    "Cherry (including sour)": {"English": "Cherry", "हिन्दी": "चेरी", "मराठी": "चेरी"},
    "Blueberry": {"English": "Blueberry", "हिन्दी": "ब्लूबेरी", "मराठी": "ब्लूबेरी"},
    "Pepper, bell": {"English": "Bell Pepper", "हिन्दी": "शिमला मिर्च", "मराठी": "ढोबळी मिरची"},
    "Soybean": {"English": "Soybean", "हिन्दी": "सोयाबीन", "मराठी": "सोयाबीन"},
    "Squash": {"English": "Squash", "हिन्दी": "स्कॉश (कद्दू वर्ग)", "मराठी": "स्क्वॅश"},
    "Orange": {"English": "Orange", "हिन्दी": "संतरा", "मराठी": "संत्री"},
    "Raspberry": {"English": "Raspberry", "हिन्दी": "रस्पबेरी", "मराठी": "रास्पबेरी"}
}

DISEASE_TRANSLATIONS = {
    "Apple scab": {"English": "Apple scab", "हिन्दी": "सेब का पपड़ी रोग", "मराठी": "सफरचंदावरील खवले रोग"},
    "Black rot": {"English": "Black rot", "हिन्दी": "ब्लैक रॉट (काला सड़न)", "मराठी": "काळा सडणे (Black Rot)"},
    "Cedar apple rust": {"English": "Cedar apple rust", "हिन्दी": "देवदार सेब जंग रोग", "मराठी": "तांबेरा रोग (Cedar Rust)"},
    "Cercospora leaf spot Gray leaf spot": {"English": "Cercospora / Gray leaf spot", "हिन्दी": "सार्कोस्पोरा पत्ती धब्बा रोग", "मराठी": "पर्णावरील करपा रोग"},
    "Common rust": {"English": "Common rust", "हिन्दी": "सामान्य जंग रोग", "मराठी": "तांबेरा रोग (Common Rust)"},
    "Northern Leaf Blight": {"English": "Northern Leaf Blight", "हिन्दी": "उत्तरी पत्ती झुलसा रोग", "मराठी": "उत्तरी पानावरील करपा"},
    "Esca (Black Measles)": {"English": "Esca (Black Measles)", "हिन्दी": "एस्का (काला खसरा)", "मराठी": "एस्का रोग (Black Measles)"},
    "Leaf blight (Isariopsis Leaf Spot)": {"English": "Leaf blight", "हिन्दी": "पत्ती झुलसा रोग", "मराठी": "पानावरील करपा रोग"},
    "Early blight": {"English": "Early blight", "हिन्दी": "अगेती झुलसा रोग", "मराठी": "लवकर येणारा करपा रोग"},
    "Late blight": {"English": "Late blight", "हिन्दी": "पछेती झुलसा रोग", "मराठी": "उशिरा येणारा करपा रोग"},
    "Bacterial spot": {"English": "Bacterial spot", "हिन्दी": "जीवाणु जनित धब्बा रोग", "मराठी": "जिवाणूजन्य ठिपके"},
    "Leaf Mold": {"English": "Leaf Mold", "हिन्दी": "पत्ती मोल्ड रोग", "मराठी": "पानावरील बुरशी (Leaf Mold)"},
    "Septoria leaf spot": {"English": "Septoria leaf spot", "हिन्दी": "सेप्टोरिया पत्ती धब्बा रोग", "मराठी": "सेप्टोरिया पानावरील ठिपके"},
    "Spider mites Two-spotted spider mite": {"English": "Two-spotted spider mite", "हिन्दी": "दो-धब्बों वाले मकड़ी के कीड़े", "मराठी": "दोन ठिपक्यांची कोळी कीड"},
    "Target Spot": {"English": "Target Spot", "हिन्दी": "टार्गेट स्पॉट रोग", "मराठी": "टार्गेट स्पॉट रोग"},
    "Tomato Yellow Leaf Curl Virus": {"English": "Tomato Yellow Leaf Curl Virus", "हिन्दी": "टमाटर पीला पत्ती कुंचन वायरस", "मराठी": "टोमॅटो यलो लीफ कर्ल व्हायरस"},
    "Tomato mosaic virus": {"English": "Tomato mosaic virus", "हिन्दी": "टमाटर मोज़ेक वायरस", "मराठी": "टोमॅटो मोझॅक व्हायरस"},
    "Leaf scorch": {"English": "Leaf scorch", "हिन्दी": "पत्ती का झुलसना", "मराठी": "पाने जळणे रोग (Leaf Scorch)"},
    "Powdery mildew": {"English": "Powdery mildew", "हिन्दी": "पाउडरी मिल्ड्यू (चूर्णिल आसिता)", "मराठी": "भुरी रोग (Powdery Mildew)"},
    "Haunglongbing (Citrus greening)": {"English": "Citrus greening (HLB)", "हिन्दी": "सिट्रस ग्रीनिंग रोग", "मराठी": "सायट्रस ग्रीनिंग रोग"},
    "healthy": {"English": "Healthy", "हिन्दी": "स्वस्थ", "मराठी": "निरोगी"}
}

treatments = {
    "Apple___Apple_scab": {
        "English": "Apply fungicides containing captan or myclobutanil. Remove and destroy infected leaves.",
        "हिन्दी": "कैप्टान या माइक्लोबुटानिल युक्त कवकनाशी लगाएं। संक्रमित पत्तियों को हटाकर नष्ट कर दें।",
        "मराठी": "कॅप्टन किंवा मायक्लोब्युटानिल असलेली बुरशीनाशके वापरा. संसर्ग झालेली पाने काढून नष्ट करा."
    },
    "Apple___Black_rot": {
        "English": "Prune infected branches. Apply copper-based fungicide every 7-10 days.",
        "हिन्दी": "संक्रमित शाखाओं की छंटाई करें। हर 7-10 दिनों में तांबा आधारित कवकनाशी लगाएं।",
        "मराठी": "संसर्ग झालेल्या फांद्या छाटा. दर ७-१० दिवसांनी तांबे-आधारित बुरशीनाशक फवारा."
    },
    "Apple___Cedar_apple_rust": {
        "English": "Apply fungicide at pink stage. Remove nearby cedar trees if possible.",
        "हिन्दी": "गुलाबी कली की अवस्था में कवकनाशी लगाएं। यदि संभव हो तो पास के देवदार (cedar) के पेड़ों को हटा दें।",
        "मराठी": "गुलाबी कळीच्या अवस्थेत बुरशीनाशक वापरा. शक्य असल्यास जवळची सिडारची झाडे काढून टाका."
    },
    "Apple___healthy": {
        "English": "No treatment needed. Continue regular care.",
        "हिन्दी": "किसी उपचार की आवश्यकता नहीं है। नियमित देखभाल जारी रखें।",
        "मराठी": "कोणत्याही उपचारांची गरज नाही. नियमित काळजी घेणे सुरू ठेवा."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "English": "Apply strobilurin fungicides. Rotate crops annually.",
        "हिन्दी": "स्ट्रोबिल्यूरिन कवकनाशी लगाएं। प्रतिवर्ष फसलों का चक्रण (crop rotation) करें।",
        "मराठी": "स्ट्रोबिल्युरिन बुरशीनाशके वापरा. दरवर्षी पिकांची फेरपालट करा."
    },
    "Corn_(maize)___Common_rust_": {
        "English": "Apply fungicides early. Use resistant varieties next season.",
        "हिन्दी": "शुरुआत में ही कवकनाशी लगाएं। अगले सीजन में रोग-प्रतिरोधी किस्मों के बीजों का उपयोग करें।",
        "मराठी": "सुरुवातीच्या काळातच बुरशीनाशके फवारा. पुढील हंगामात रोगप्रतिकारक जातींचे बियाणे वापरा."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "English": "Apply fungicide at first sign. Ensure good air circulation.",
        "हिन्दी": "पहला लक्षण दिखते ही कवकनाशी लगाएं। खेतों में हवा का अच्छा संचार सुनिश्चित करें।",
        "मराठी": "पहिले लक्षण दिसताच बुरशीनाशक वापरा. शेतात हवा खेळती राहील याची काळजी घ्या."
    },
    "Corn_(maize)___healthy": {
        "English": "No treatment needed. Continue regular care.",
        "हिन्दी": "किसी उपचार की आवश्यकता नहीं है। नियमित देखभाल जारी रखें।",
        "मराठी": "कोणत्याही उपचारांची गरज नाही. नियमित काळजी घेणे सुरू ठेवा."
    },
    "Grape___Black_rot": {
        "English": "Apply mancozeb or myclobutanil. Remove mummified berries.",
        "हिन्दी": "मैनकोजेब या माइक्लोबुटानिल लगाएं। सूखे और सड़े हुए अंगूरों को बेल से हटा दें।",
        "मराठी": "मॅन्कोझेब किंवा मायक्लोब्युटानिल फवारा. सुकलेली आणि सडलेली द्राक्षे वेलीवरून काढून टाका."
    },
    "Grape___Esca_(Black_Measles)": {
        "English": "Prune infected wood. Apply wound sealant after pruning.",
        "हिन्दी": "संक्रमित लकड़ी की छंटाई करें। छंटाई के बाद कटे हुए हिस्सों पर बोर्डो पेस्ट लगाएं।",
        "मराठी": "बाधित झालेली लाकडे छाटून टाका. छाटणी केल्यानंतर झाडाच्या जखमेवर बोर्डो पेस्ट किंवा सीलंट लावा."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "English": "Apply copper fungicide. Improve vineyard air circulation.",
        "हिन्दी": "कॉपर (तांबा) कवकनाशी लगाएं। अंगूर के बाग में हवा का प्रवाह बेहतर करें।",
        "मराठी": "तांबे-युक्त बुरशीनाशक लावा. द्राक्षाच्या बागेत हवा खेळती राहू द्या."
    },
    "Grape___healthy": {
        "English": "No treatment needed. Continue regular care.",
        "हिन्दी": "किसी उपचार की आवश्यकता नहीं है। नियमित देखभाल जारी रखें।",
        "मराठी": "कोणत्याही उपचारांची गरज नाही. नियमित काळजी घेणे सुरू ठेवा."
    },
    "Potato___Early_blight": {
        "English": "Apply chlorothalonil or mancozeb. Remove infected leaves immediately.",
        "हिन्दी": "क्लोरोथैलोनिल या मैनकोजेब का छिड़काव करें। संक्रमित पत्तियों को तुरंत हटा दें।",
        "मराठी": "क्लोरोथॅलोनिल किंवा मॅन्कोझेब फवारा. संसर्ग झालेली पाने त्वरित काढून टाका."
    },
    "Potato___Late_blight": {
        "English": "Apply metalaxyl fungicide urgently. Destroy infected plants.",
        "हिन्दी": "तुरंत मेटलैक्सिल कवकनाशी का प्रयोग करें। अत्यधिक संक्रमित पौधों को उखाड़कर नष्ट कर दें।",
        "मराठी": "त्वरित मेटलॅक्सिल बुरशीनाशकाचा वापर करा. रोगट वनस्पती मुळासकट उपटून नष्ट करा."
    },
    "Potato___healthy": {
        "English": "No treatment needed. Continue regular care.",
        "हिन्दी": "किसी उपचार की आवश्यकता नहीं है। नियमित देखभाल जारी रखें।",
        "मराठी": "कोणत्याही उपचारांची गरज नाही. नियमित काळजी घेणे सुरू ठेवा."
    },
    "Tomato___Bacterial_spot": {
        "English": "Apply copper-based bactericide. Avoid overhead irrigation.",
        "हिन्दी": "तांबा आधारित जीवाणुनाशक लगाएं। पौधों के ऊपर से पानी देने (ओवरहेड सिंचाई) से बचें।",
        "मराठी": "तांबे-आधारित जिवाणूनाशक वापरा. पानावरील तुषार सिंचन टाळा."
    },
    "Tomato___Early_blight": {
        "English": "Apply fungicide containing chlorothalonil. Remove lower infected leaves.",
        "हिन्दी": "क्लोरोथैलोनिल युक्त कवकनाशी लगाएं। नीचे की संक्रमित पत्तियों को हटा दें।",
        "मराठी": "क्लोरोथॅलोनिलयुक्त बुरशीनाशक वापरा. खालची संसर्ग झालेली पाने काढून टाका."
    },
    "Tomato___Late_blight": {
        "English": "Apply metalaxyl immediately. Remove and destroy infected plants.",
        "हिन्दी": "तुरंत मेटलैक्सिल कवकनाशी का छिड़काव करें। संक्रमित पौधों को खेतों से हटाकर नष्ट कर दें।",
        "मराठी": "त्वरित मेटलॅक्सिल फवारा. संसर्ग केलेले झाड काढून नष्ट करा."
    },
    "Tomato___Leaf_Mold": {
        "English": "Improve ventilation. Apply fungicide containing chlorothalonil.",
        "हिन्दी": "खेत या ग्रीनहाउस में वेंटिलेशन (हवा का आना-जाना) सुधारें। कवकनाशी लगाएं।",
        "मराठी": "हवेची हालचाल (व्हेंटिलेशन) सुधारा. क्लोरोथॅलोनिल असलेले बुरशीनाशक वापरा."
    },
    "Tomato___Septoria_leaf_spot": {
        "English": "Apply mancozeb or chlorothalonil. Remove infected leaves.",
        "हिन्दी": "मैनकोजेब या क्लोरोथैलोनिल लगाएं। रोगग्रस्त पत्तियों को तोड़कर नष्ट करें।",
        "मराठी": "मॅन्कोझेब किंवा क्लोरोथॅलोनिल वापरा. संसर्ग झालेली पाने काढून टाका."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "English": "Apply miticide or neem oil. Increase humidity around plants.",
        "हिन्दी": "माइटिसाइड या नीम के तेल का प्रयोग करें। पौधों के आसपास नमी बढ़ाएं।",
        "मराठी": "मायटीसाइड किंवा कडुनिंबाचे तेल वापरा. झाडांच्या आजूबाजूला आर्द्रता वाढवा."
    },
    "Tomato___Target_Spot": {
        "English": "Apply fungicide. Ensure proper plant spacing for airflow.",
        "हिन्दी": "उपयुक्त कवकनाशी लगाएं। हवा के अच्छे प्रवाह के लिए पौधों के बीच उचित दूरी सुनिश्चित करें।",
        "मराठी": "योग्य बुरशीनाशक लावा. हवेच्या प्रवाहासाठी झाडांमध्ये योग्य अंतर ठेवा."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "English": "Remove infected plants. Control whitefly population with insecticide.",
        "हिन्दी": "संक्रमित पौधों को तुरंत उखाड़ें। कीटनाशक की मदद से सफेद मक्खी को नियंत्रित करें।",
        "मराठी": "बाधित झाडे काढून टाका. कीटकनाशकाचा वापर करून पांढऱ्या माशीचे नियंत्रण करा."
    },
    "Tomato___Tomato_mosaic_virus": {
        "English": "Remove infected plants. Disinfect tools. Control aphids.",
        "हिन्दी": "संक्रमित पौधों को नष्ट करें। कृषि उपकरणों को साफ और कीटाणुरहित करें।",
        "मराठी": "बाधित झाडे नष्ट करा. शेतीची अवजारे जंतूमुक्त करा. मावा कीड नियंत्रित करा."
    },
    "Tomato___healthy": {
        "English": "No treatment needed. Continue regular care.",
        "हिन्दी": "किसी उपचार की आवश्यकता नहीं है। नियमित देखभाल जारी रखें।",
        "मराठी": "कोणत्याही उपचारांची गरज नाही. नियमित काळजी घेणे सुरू ठेवा."
    },
    "Strawberry___Leaf_scorch": {
        "English": "Remove infected leaves. Apply fungicide containing captan. Avoid overhead watering.",
        "हिन्दी": "संक्रमित पत्तियां हटाएं। कैप्टान युक्त कवकनाशी का प्रयोग करें। ऊपर से पानी छिड़कने से बचें।",
        "मराठी": "बाधित पाने काढून टाका. कॅप्टन असलेले बुरशीनाशक वापरा. पानावरील थेट पाणी देणे टाळा."
    },
    "Strawberry___healthy": {
        "English": "No treatment needed. Continue regular care.",
        "हिन्दी": "किसी उपचार की आवश्यकता नहीं है। नियमित देखभाल जारी रखें।",
        "मराठी": "कोणत्याही उपचारांची गरज नाही. नियमित काळजी घेणे सुरू ठेवा."
    },
    "Peach___Bacterial_spot": {
        "English": "Apply copper-based bactericide in early spring. Prune infected branches.",
        "हिन्दी": "शुरुआती वसंत में तांबा आधारित जीवाणुनाशक लगाएं। संक्रमित शाखाओं की छंटाई करें।",
        "मराठी": "वसंत ऋतूच्या सुरुवातीला तांबे-आधारित जिवाणूनाशक वापरा. बाधित फांद्या छाटा."
    },
    "Peach___healthy": {
        "English": "No treatment needed. Continue regular care.",
        "हिन्दी": "किसी उपचार की आवश्यकता नहीं है। नियमित देखभाल जारी रखें।",
        "मराठी": "कोणत्याही उपचारांची गरज नाही. नियमित काळजी घेणे सुरू ठेवा."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "English": "Apply sulfur-based fungicide. Ensure good air circulation.",
        "हिन्दी": "सल्फर आधारित कवकनाशी लगाएं। हवा का अच्छा संचरण सुनिश्चित करें।",
        "मराठी": "गंधक-आधारित बुरशीनाशक वापरा. हवा खेळती राहील याची खात्री करा."
    },
    "Cherry_(including_sour)___healthy": {
        "English": "No treatment needed. Continue regular care.",
        "हिन्दी": "किसी उपचार की आवश्यकता नहीं है। नियमित देखभाल जारी रखें।",
        "मराठी": "कोणत्याही उपचारांची गरज नाही. नियमित काळजी घेणे सुरू ठेवा."
    },
    "Blueberry___healthy": {
        "English": "No treatment needed. Continue regular care.",
        "हिन्दी": "किसी उपचार की आवश्यकता नहीं है। नियमित देखभाल जारी रखें।",
        "मराठी": "कोणत्याही उपचारांची गरज नाही. नियमित काळजी घेणे सुरू ठेवा."
    },
    "Pepper,_bell___Bacterial_spot": {
        "English": "Apply copper bactericide. Avoid overhead irrigation.",
        "हिन्दी": "कॉपर जीवाणुनाशक लगाएं। पौधों के ऊपर सीधे पानी देने से बचें।",
        "मराठी": "तांबेयुक्त जिवाणूनाशक वापरा. तुषार सिंचन पद्धत टाळा."
    },
    "Pepper,_bell___healthy": {
        "English": "No treatment needed. Continue regular care.",
        "हिन्दी": "किसी उपचार की आवश्यकता नहीं है। नियमित देखभाल जारी रखें।",
        "मराठी": "कोणत्याही उपचारांची गरज नाही. नियमित काळजी घेणे सुरू ठेवा."
    },
    "Soybean___healthy": {
        "English": "No treatment needed. Continue regular care.",
        "हिन्दी": "किसी उपचार की आवश्यकता नहीं है। नियमित देखभाल जारी रखें।",
        "मराठी": "कोणत्याही उपचारांची गरज नाही. नियमित काळजी घेणे सुरू ठेवा."
    },
    "Squash___Powdery_mildew": {
        "English": "Apply potassium bicarbonate or neem oil. Improve air circulation.",
        "हिन्दी": "पोटेशियम बाइकार्बोनेट या नीम के तेल का प्रयोग करें। वायु संचरण में सुधार करें।",
        "मराठी": "पोटॅशियम बायकार्बोनेट किंवा कडुनिंबाचे तेल वापरा. हवा खेळती ठेवा."
    },
    "Raspberry___healthy": {
        "English": "No treatment needed. Continue regular care.",
        "हिन्दी": "किसी उपचार की आवश्यकता नहीं है। नियमित देखभाल जारी रखें।",
        "मराठी": "कोणत्याही उपचारांची गरज नाही. नियमित काळजी घेणे सुरू ठेवा."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "English": "No cure available. Remove infected trees. Control psyllid insects.",
        "हिन्दी": "कोई स्थायी इलाज उपलब्ध नहीं है। संक्रमित पेड़ों को हटा दें। सिलिड कीटों को नियंत्रित करें।",
        "मराठी": "या रोगावर कोणताही खात्रीशीर इलाज नाही. संसर्ग झालेली झाडे काढून टाका आणि सिलीड कीटक नियंत्रित करा."
    },
}

def clean_digits(val_str, lang_choice):
    if lang_choice == "English":
        return val_str
    
    # Simple localization digit replacer rule
    digit_map = {
        '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
        '5': '५', '6': '६', '7': '७', '8': '८', '9': '९'
    }
    return "".join(digit_map.get(char, char) for char in str(val_str))

def get_treatment(cls, lang):
    return treatments.get(cls, {}).get(lang, I18N[lang]["fallback_advice"])

def get_translated_label(raw_pred, lang):
    parts = raw_pred.split('___')
    raw_crop = parts[0].replace('_', ' ').strip()
    raw_disease = parts[1].replace('_', ' ').strip() if len(parts) > 1 else 'healthy'
    
    clean_crop_key = "Corn (maize)" if "Corn" in raw_crop else ("Pepper, bell" if "Pepper" in raw_crop else ("Cherry (including sour)" if "Cherry" in raw_crop else raw_crop))
    
    crop_tr = CROP_TRANSLATIONS.get(clean_crop_key, {}).get(lang, raw_crop)
    disease_tr = DISEASE_TRANSLATIONS.get(raw_disease, {}).get(lang, raw_disease)
    
    if 'healthy' in raw_pred.lower():
        return f"{crop_tr} — " + ("स्वस्थ है" if lang == "हिन्दी" else ("निरोगी आहे" if lang == "मराठी" else "Healthy"))
    else:
        return f"{crop_tr} — {disease_tr}"

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

def generate_gradcam(model, tensor, class_idx):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_f = model.layer4.register_forward_hook(forward_hook)
    handle_b = model.layer4.register_full_backward_hook(backward_hook)

    output = model(tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    handle_f.remove()
    handle_b.remove()

    grads = gradients[0].detach()
    acts = activations[0].detach()
    weights = grads.mean(dim=[2, 3], keepdim=True)

    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().numpy()

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = (cam * 255).astype(np.uint8)

    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig = tensor.squeeze().permute(1, 2, 0).numpy()
    orig = orig * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    orig = np.clip(orig * 255, 0, 255).astype(np.uint8)

    overlay = cv2.addWeighted(orig, 0.55, heatmap, 0.45, 0)
    return Image.fromarray(overlay)

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model, class_names = load_model()

# Header Area layout with a container class to customize selector label
col_title, col_lang = st.columns([2.8, 1.2])
with col_lang:
    st.markdown('<div class="lang-container">', unsafe_allow_html=True)
    lang = st.selectbox("🌐 Language / भाषा चुनें / भाषा निवडा", ["English", "हिन्दी", "मराठी"])
    st.markdown('</div>', unsafe_allow_html=True)

# Set dynamic configuration language variable
tr = I18N[lang]

st.markdown(f"""
<div class="hero">
    <div class="hero-badge">{tr["badge_text"]}</div>
    <h1>{tr["title_pre"]}<span class="g">{tr["title_post"]}</span></h1>
    <p>{tr["hero_sub"]}</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card"><div class="stat-number">{tr["num_acc"]}</div><div class="stat-label">{tr["stat_acc"]}</div></div>
    <div class="stat-card"><div class="stat-number">{tr["num_cls"]}</div><div class="stat-label">{tr["stat_cls"]}</div></div>
    <div class="stat-card"><div class="stat-number">{tr["num_imgs"]}</div><div class="stat-label">{tr["stat_imgs"]}</div></div>
    <div class="stat-card"><div class="stat-number">{tr["num_crops"]}</div><div class="stat-label">{tr["stat_crops"]}</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(tr["uploader_lbl"], type=["jpg", "jpeg", "png"])

image = None
if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
    except Exception as e:
        st.error(f"Could not read image: {e}")

if not image:
    st.markdown(f'<div class="placeholder-box"><div style="font-size:1.6rem;margin-bottom:0.4rem;">🌿</div><div class="placeholder-text">{tr["placeholder"]}</div></div>', unsafe_allow_html=True)
else:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(f'<div class="glass-panel"><div class="panel-title">{tr["preview_title"]}</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        with st.spinner("🔬 ..."):
            tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, 1)[0]
                top3 = torch.topk(probs, 3)

            pred_idx = top3.indices[0].item()
            pred = class_names[pred_idx]
            conf = top3.values[0].item() * 100
            
            parts = pred.split('___')
            raw_crop = parts[0].replace('_', ' ').strip()
            raw_disease = parts[1].replace('_', ' ').strip() if len(parts) > 1 else 'healthy'
            
            clean_crop_key = "Corn (maize)" if "Corn" in raw_crop else ("Pepper, bell" if "Pepper" in raw_crop else ("Cherry (including sour)" if "Cherry" in raw_crop else raw_crop))
            
            translated_crop = CROP_TRANSLATIONS.get(clean_crop_key, {}).get(lang, raw_crop)
            translated_disease = DISEASE_TRANSLATIONS.get(raw_disease, {}).get(lang, raw_disease)
            healthy = 'healthy' in pred.lower()

            gradcam_img = generate_gradcam(model, tensor, pred_idx)

        localized_conf = clean_digits(f"{conf:.1f}%", lang)

        if healthy:
            banner = (
                '<div class="healthy-banner">'
                f'<div class="banner-title">✅ {translated_crop} — {tr["healthy_title"]}</div>'
                f'<div class="banner-sub">{tr["healthy_sub"]}</div>'
                '</div>'
            )
        else:
            banner = (
                '<div class="disease-banner">'
                f'<div class="banner-title">⚠️ {translated_disease}</div>'
                f'<div class="banner-sub">({translated_crop})</div>'
                '</div>'
            )

        metrics = (
            '<div class="metric-row">'
            f'<div class="metric-card"><div class="metric-label">{tr["crop_lbl"]}</div>'
            f'<div class="metric-value">{translated_crop}</div></div>'
            f'<div class="metric-card"><div class="metric-label">{tr["conf_lbl"]}</div>'
            f'<div class="metric-value">{localized_conf}</div></div>'
            '</div>'
        )

        conf_bar = pbar(conf, 9)

        treatment = (
            '<div class="treatment-box">'
            f'<div class="treatment-title">{tr["treatment_lbl"]}</div>'
            f'<div class="treatment-text">{get_treatment(pred, lang)}</div>'
            '</div>'
        )

        top3_rows = ''
        for i in range(3):
            cls = class_names[top3.indices[i].item()]
            prob = top3.values[i].item() * 100
            lbl = get_translated_label(cls, lang)
            localized_prob_str = clean_digits(f"{prob:.1f}%", lang)
            top3_rows += f'<span class="pred-label">{lbl}: {localized_prob_str}</span>' + pbar(prob, 7)

        top3_block = (
            '<details style="margin-top:0.8rem;">'
            '<summary style="cursor:pointer;font-size:0.82rem;font-weight:700;color:#1e3a08;'
            'padding:0.5rem;background:rgba(140,190,100,0.30);border-radius:8px;list-style:none;">'
            f'{tr["top3_lbl"]}</summary>'
            '<div style="padding:0.6rem 0.2rem 0.2rem;">' + top3_rows + '</div>'
            '</details>'
        )

        html = (
            '<div class="glass-panel">'
            f'<div class="panel-title">{tr["analysis_title"]}</div>'
            + banner + metrics + conf_bar + treatment + top3_block +
            '</div>'
        )

        st.markdown(html, unsafe_allow_html=True)

    # Grad-CAM section
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-panel"><div class="panel-title">{tr["gradcam_title"]}</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1.2, 1, 1.2])
    with col2:
        st.image(gradcam_img, use_container_width=True)
    st.markdown(f'<div class="gradcam-note">{tr["gradcam_desc"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(f'<div class="section-label">{tr["how_works"]}</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="steps-grid">
    <div class="step-card"><div class="step-number">{tr["step1_num"]}</div><div class="step-title">{tr["step1_t"]}</div><div class="step-desc">{tr["step1_d"]}</div></div>
    <div class="step-card"><div class="step-number">{tr["step2_num"]}</div><div class="step-title">{tr["step2_t"]}</div><div class="step-desc">{tr["step2_d"]}</div></div>
    <div class="step-card"><div class="step-number">{tr["step3_num"]}</div><div class="step-title">{tr["step3_t"]}</div><div class="step-desc">{tr["step3_d"]}</div></div>
</div>
""", unsafe_allow_html=True)

# Supported Crops Map Elements
pill_crops = [
    ("🍎", "Apple"), ("🌽", "Corn (maize)"), ("🍇", "Grape"), ("🥔", "Potato"), ("🍅", "Tomato"),
    ("🍑", "Peach"), ("🍒", "Cherry (including sour)"), ("🫐", "Blueberry"), ("🌶️", "Pepper, bell"),
    ("🍓", "Strawberry"), ("🌿", "Soybean"), ("🎃", "Squash"), ("🍊", "Orange"), ("🍇", "Raspberry")
]

crop_pills_html = ""
for emoji, c_key in pill_crops:
    trans_name = CROP_TRANSLATIONS.get(c_key, {}).get(lang, c_key)
    crop_pills_html += f'<div class="crop-pill"><span class="crop-emoji">{emoji}</span>{trans_name}</div>'

st.markdown(f'<div class="section-label">{tr["supported_crops"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="crops-grid">{crop_pills_html}</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="footer">
    {tr["footer_text"]}
</div>
""", unsafe_allow_html=True)
