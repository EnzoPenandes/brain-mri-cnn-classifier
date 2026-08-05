import torch
import torch.nn.functional as F
import gradio as gr

from PIL import Image
from torchvision import transforms

from models.cnn import BrainTumorCNN
import config

# -----------------------------
# DEVICE
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------
# LOAD MODEL
# -----------------------------

model = BrainTumorCNN().to(device)

model.load_state_dict(
    torch.load(
        "outputs/models/cnn_best.pth",
        map_location=device
    )
)

model.eval()

# -----------------------------
# TRANSFORM
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
])

classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# -----------------------------
# PREDICT FUNCTION
# -----------------------------

def predict(image):

    image = image.convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = F.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, 1)

    return {
        classes[i]: float(probabilities[0][i])
        for i in range(4)
    }

# -----------------------------
# GRADIO UI
# -----------------------------

demo = gr.Interface(

    fn=predict,

    inputs=gr.Image(type="pil"),

    outputs=gr.Label(num_top_classes=4),

    title="MRI Brain Tumor Classifier",

    description="Upload an MRI image to classify brain tumors."

)

demo.launch()