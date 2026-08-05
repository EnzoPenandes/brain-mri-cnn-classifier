#Imports

import matplotlib.pyplot as plt

import time
import os

import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import BrainTumorCNN

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import config


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ==========================================
# LOAD DATASETS
# ==========================================

train_dataset = datasets.ImageFolder(
    root=config.TRAIN_DIR,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=config.TEST_DIR,
    transform=transform
)

# ==========================================
# CREATE DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device : {device}")

# ==========================================
# MODEL
# ==========================================

model = BrainTumorCNN().to(device)

# ==========================================
# LOSS FUNCTION
# ==========================================

criterion = nn.CrossEntropyLoss()

# ==========================================
# OPTIMIZER
# ==========================================

optimizer = optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE
)

# ==========================================
# VERIFY DATASET
# ==========================================

print("=" * 50)
print("Dataset Information")
print("=" * 50)

print(f"Training Images : {len(train_dataset)}")
print(f"Testing Images  : {len(test_dataset)}")

print(f"\nClasses : {train_dataset.classes}")

print(f"\nClass Mapping :")
print(train_dataset.class_to_idx)

print("=" * 50)

# ==========================================
# VERIFY DATALOADER
# ==========================================

images, labels = next(iter(train_loader))

print(f"\nBatch Shape : {images.shape}")
print(f"Labels Shape : {labels.shape}")

# ==========================================
# TRAINING HISTORY
# ==========================================

train_losses = []
train_accuracies = []

best_accuracy = 0.0

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nStarting Training...\n")

for epoch in range(config.EPOCHS):

    start_time = time.time()

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)

    train_accuracies.append(epoch_accuracy)

    epoch_time = time.time() - start_time

    print("=" * 55)
    print(f"Epoch {epoch+1}/{config.EPOCHS}")
    print(f"Loss      : {epoch_loss:.4f}")
    print(f"Accuracy  : {epoch_accuracy:.2f}%")
    print(f"Time      : {epoch_time:.2f} sec")
    print("=" * 55)

    if epoch_accuracy > best_accuracy:

        best_accuracy = epoch_accuracy

        torch.save(
            model.state_dict(),
            "outputs/models/cnn_best.pth"
        )

print("\nTraining Complete!")

print(f"Best Accuracy : {best_accuracy:.2f}%")
print("Best Model Saved -> outputs/models/cnn_best.pth")

print("\nTraining Complete!")

print(f"Best Accuracy : {best_accuracy:.2f}%")
print("Best Model Saved -> outputs/models/cnn_best.pth")

# ==========================================
# SAVE TRAINING GRAPHS
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(
    range(1, config.EPOCHS + 1),
    train_losses,
    marker="o",
    linewidth=2
)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.savefig("outputs/graphs/loss_curve.png")

plt.close()



plt.figure(figsize=(8,5))

plt.plot(
    range(1, config.EPOCHS + 1),
    train_accuracies,
    marker="o",
    linewidth=2,
    color="green"
)

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.grid(True)

plt.savefig("outputs/graphs/accuracy_curve.png")

plt.close()


print("Graphs Saved!")