# PyTorch training script for mangrove vs. non-mangrove classification
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# Paths
DATA_DIR = 'dataset'
MODEL_PATH = 'mangrove_mobilenetv2.pth'
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets and loaders (only mangrove and forest)
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR), transform=transform)
assert set(train_dataset.classes) == {'mangrove', 'forest'}, f"Classes found: {train_dataset.classes}"
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 2 classes
model = model.cuda() if torch.cuda.is_available() else model

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/total:.4f} - Acc: {correct/total:.4f}")

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
