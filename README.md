# 🌿 CropGuard AI — Crop Disease Detector

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-99.57%25-brightgreen?style=for-the-badge&logo=checkmarx" />
  <img src="https://img.shields.io/badge/Model-ResNet--50-blue?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Classes-38%20Diseases-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/For-Indian%20Farmers%20🇮🇳-orange?style=for-the-badge" />
</p>

<p align="center">
  <b>Instant AI-powered crop disease detection — upload a leaf photo, get a diagnosis and treatment plan in seconds.</b>
</p>

---

## 🚀 Live Demo

<p align="center">
  <a href="https://crop-disease-detector-ai.streamlit.app/">
    <img src="https://img.shields.io/badge/🌿%20Try%20CropGuard%20AI%20Live-Click%20Here-brightgreen?style=for-the-badge" />
  </a>
</p>

👉 **[https://crop-disease-detector-ai.streamlit.app/](https://crop-disease-detector-ai.streamlit.app/)**

> No installation needed — just open the link and upload a leaf photo!

---

## 📖 How to Use

| Step | What to Do |
|------|-----------|
| 1️⃣ | Open the live link above |
| 2️⃣ | Click **Browse files** or drag and drop a leaf photo (JPG or PNG) |
| 3️⃣ | Wait 1-2 seconds for the AI to analyse |
| 4️⃣ | See the detected disease name and confidence score |
| 5️⃣ | Read the treatment advice and act immediately |

> 💡 **Tip** — Use a clear, close-up photo of a single leaf in good natural lighting for best results!

---

## ✨ Features

| Feature | Details |
|--------|---------|
| 🎯 Accuracy | 99.57% on PlantVillage dataset |
| 🌾 Disease Classes | 38 classes across 14 crop types |
| 💊 Treatment Advice | Actionable treatment for every disease |
| 📊 Top 3 Predictions | With confidence scores |
| ⚡ Speed | Results in under 2 seconds |
| 🇮🇳 Farmer Focused | Kisan helpline number included |

---

## 🌱 Supported Crops (14 Types)

| | | | | |
|--|--|--|--|--|
| 🍎 Apple | 🌽 Corn | 🍇 Grape | 🥔 Potato | 🍅 Tomato |
| 🍑 Peach | 🍒 Cherry | 🫐 Blueberry | 🌶️ Pepper | 🍓 Strawberry |
| 🌿 Soybean | 🎃 Squash | 🍊 Orange | 🍇 Raspberry | |

---

## 🧠 Model & Training

| Detail | Info |
|--------|------|
| Architecture | ResNet-50 (Transfer Learning) |
| Dataset | PlantVillage (87,000+ images) |
| Training Platform | Google Colab (T4 GPU) |
| Epochs | 10 |
| Best Val Accuracy | 99.57% |
| Framework | PyTorch |
| Optimizer | Adam |

### Training Progress

| Epoch | Train Loss | Train Acc | Val Acc |
|-------|-----------|-----------|---------|
| 1 | 0.847 | 74.23% | 89.12% |
| 5 | 0.142 | 95.80% | 98.21% |
| 10 | 0.068 | 97.87% | **99.57%** |

---

## 📁 Project Structure

```
crop-disease-detector/
│
├── app.py                        # Streamlit web app
├── class_names.json              # 38 disease class labels
├── requirements.txt              # Python dependencies
├── Crop_Disease_Detector.ipynb   # Training notebook (Google Colab)
├── bg.jpg                        # Background image
└── best_model.pth                # Trained weights (download below)
```

---

## 📥 Download Model Weights

`best_model.pth` is not stored in this repo due to file size.

👉 **[Download best_model.pth from Google Drive](YOUR_GOOGLE_DRIVE_LINK)**

---

## 🔬 How It Works

```
📸 Upload leaf photo
        ↓
🔄 ResNet-50 processes image (224x224)
        ↓
📊 Softmax gives confidence scores for 38 classes
        ↓
💊 Top prediction + treatment advice displayed
```

---

## 📞 Farmer Support

**Kisan Call Centre: 1800-180-1551** — Free, 24/7 🇮🇳

---

## 👩‍💻 About

Built by **Khushboo Pingle** as part of an AI/ML project.
Trained on the PlantVillage dataset using ResNet-50 with PyTorch on Google Colab T4 GPU.

⭐ *Star this repo if it helped you!*