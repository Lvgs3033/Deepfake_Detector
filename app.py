import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image


st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ Deepfake Image Detector")
st.write("Upload an image and check whether it is **Real** or **Deepfake**.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(
        torch.load("models/deepfake_detector.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model

model = load_model()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

classes = ["Fake", "Real"]


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect"):
        with torch.no_grad():
            img_tensor = transform(image).unsqueeze(0).to(device)
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)

        label = classes[pred.item()]
        confidence = confidence.item() * 100

        if label == "Real":
            st.success(f"✅ **REAL IMAGE** ({confidence:.2f}% confidence)")
        else:
            st.error(f"🚨 **DEEPFAKE IMAGE** ({confidence:.2f}% confidence)")
