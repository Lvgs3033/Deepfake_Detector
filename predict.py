import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/deepfake_detector.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
model.eval()

classes = ["fake", "real"]

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    return classes[pred.item()]

print(predict("Sample.png"))
