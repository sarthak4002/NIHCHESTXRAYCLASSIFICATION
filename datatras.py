import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
import numpy as np

# ====================
# CONFIG
# ====================
DATASET_DIR = 'C:/Users/sarth/OneDrive/Desktop/Small dataset'
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# ====================
# DEVICE SETUP
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================
# TRANSFORMS (Tuned for better generalization)
# ====================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====================
# LOAD DATASET
# ====================
full_dataset = datasets.ImageFolder(root=DATASET_DIR)

indices = list(range(len(full_dataset)))
labels = [sample[1] for sample in full_dataset.samples]

train_idx, temp_idx = train_test_split(indices, stratify=labels, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, stratify=np.array(labels)[temp_idx], test_size=0.5, random_state=42)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_test_transform
test_dataset.dataset.transform = val_test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class_names = full_dataset.classes
print("Classes:", class_names)

# ====================
# MODEL: Swin Transformer v2 Base
# ====================
model = models.swin_v2_b(weights='IMAGENET1K_V1')
in_features = model.head.in_features
model.head = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, len(class_names))
)
model = model.to(device)

# ====================
# LOSS, OPTIMIZER, SCHEDULER
# ====================
# Slight label smoothing to avoid overconfidence
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# ====================
# TRAINING & EVAL FUNCTIONS
# ====================
def train(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct = 0.0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels)

    return running_loss / len(loader.dataset), correct.double() / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct = 0.0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)

    return running_loss / len(loader.dataset), correct.double() / len(loader.dataset)

# ====================
# TRAINING LOOP
# ====================
best_val_acc = 0.0
best_model_state = None

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()

# ====================
# FINAL TEST EVALUATION
# ====================
model.load_state_dict(best_model_state)
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"\n✅ Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

# ====================
# SAVE MODEL
# ====================
torch.save(model.state_dict(), 'swin_transformer_best.pth')
