import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)


train_loader = DataLoader(
    train_dataset,
    batch_size=16,       
    shuffle=True,
    num_workers=0
)


model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished | Avg Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), os.path.join(MODEL_DIR, "deepfake_detector.pth"))
print("✅ Model saved successfully!")
