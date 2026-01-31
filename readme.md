# 🧠 Deepfake Image Detector AI

A modern **AI-powered Deepfake Detection system** that classifies images as **Real** or **Deepfake** using **Deep Learning** and provides an **interactive, animated Streamlit frontend**.

This project is ideal for:
- 🎓 Academic / college projects
- 🧪 Deep learning experimentation
- 💼 Portfolio & demos
- 🛡️ Media authenticity research

---

## ✨ Features

- 🤖 **Deep Learning Model** (ResNet18 – Transfer Learning)
- 🖼️ **Image-based Deepfake Detection** (Real vs Fake)
- 📊 **Confidence Score (%)** for predictions
- 🎨 **Attractive Streamlit UI** with animations & dark theme
- ⚡ Works on **CPU & GPU**
- 🧩 Clean, modular project structure

---

## 🗂️ Project Structure

```
Deepfake/
│
├── dataset/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
│
├── models/
│   └── deepfake_detector.pth
│
├── app.py          # Streamlit frontend
├── train.py        # Model training
├── predict.py      # Single image prediction
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

- **Base Model:** ResNet18 (pretrained on ImageNet)
- **Technique:** Transfer Learning
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Input Size:** 224 × 224 RGB images
- **Classes:**
  - `Real`
  - `Fake`

---

## 📦 Installation

### 1️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Ensure your dataset is placed correctly inside the `dataset/` folder.

```bash
python train.py
```

After training, the model will be saved to:

```
models/deepfake_detector.pth
```

---

## 🔍 Test with a Single Image

```bash
python predict.py
```

You can update the image path inside `predict.py` to test different images.

---

## 🎨 Run the Streamlit App

Launch the interactive web interface:

```bash
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

### 🖥️ App Features
- Image upload
- Live preview
- AI prediction (Real / Deepfake)
- Confidence progress bar
- Animated & modern UI

---

## 📊 Example Output

- ✅ **REAL IMAGE — 92.34% confidence**
- 🚨 **DEEPFAKE IMAGE — 87.11% confidence**

---

## ⚠️ Notes & Tips

- Balanced datasets improve accuracy
- CPU training is slower (GPU recommended)
- Model accuracy depends on dataset quality
- Best results achieved with face-focused images