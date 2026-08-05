import torch
import torch.nn.functional as F

from PIL import Image

from torchvision import transforms

from models.cnn import BrainTumorCNN

import config


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
# IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ==========================================
# IMAGE PATH
# ==========================================

image_path = input("Enter MRI Image Path : ")

image = Image.open(image_path).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)


# ==========================================
# PREDICTION
# ==========================================

with torch.no_grad():

    output = model(image)

    probabilities = F.softmax(output, dim=1)

    confidence, prediction = torch.max(probabilities, 1)


classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


print("\n==============================")

print("Prediction Results")

print("==============================")

print(f"\nPrediction : {classes[prediction.item()]}")

print(f"Confidence : {confidence.item()*100:.2f}%")

print("\nClass Probabilities\n")

for i, class_name in enumerate(classes):

    print(
        f"{class_name:<15}: {probabilities[0][i].item()*100:.2f}%"
    )