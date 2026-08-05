import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from models.cnn import BrainTumorCNN
import config


# ==========================================
# IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ==========================================
# LOAD TEST DATASET
# ==========================================

test_dataset = datasets.ImageFolder(
    root=config.TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# LOAD MODEL
# ==========================================

model = BrainTumorCNN().to(device)

model.load_state_dict(
    torch.load(
        "outputs/models/cnn_best.pth",
        map_location=device
    )
)

model.eval()


# ==========================================
# EVALUATION
# ==========================================

true_labels = []
predicted_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted.cpu().numpy())


accuracy = accuracy_score(true_labels, predicted_labels)

precision = precision_score(
    true_labels,
    predicted_labels,
    average="weighted"
)

recall = recall_score(
    true_labels,
    predicted_labels,
    average="weighted"
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    average="weighted"
)


print("="*50)
print("Evaluation Results")
print("="*50)

print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=test_dataset.classes,
    yticklabels=test_dataset.classes
)

plt.title("Brain Tumor Classification")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("outputs/confusion_matrix/confusion_matrix.png")

plt.close()

print("\nConfusion Matrix Saved!")