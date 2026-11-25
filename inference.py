from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import build_model

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIZE = 512
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
THRESHOLD = 0.5


class ResizePad:
    """Same aspect-preserving resize and centred white padding as the notebook."""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        img = img.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))),
            Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        canvas.paste(img, ((self.size - img.width) // 2,
                           (self.size - img.height) // 2))
        return canvas


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {CHECKPOINT_PATH.name}. Put your trained best.pt "
            "checkpoint in the same folder as app.py."
        )

    # The notebook checkpoint contains model weights and simple metadata only.
    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location=DEVICE, weights_only=True
        )
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    arch = checkpoint.get("arch", "resnet50")
    model = build_model(arch=arch)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    class_to_idx = checkpoint.get("class_to_idx", {})
    if "tumor" not in class_to_idx:
        raise ValueError(
            "The checkpoint does not contain class_to_idx['tumor']. "
            "Check that this is the best.pt file produced by your training notebook."
        )
    tumor_idx = int(class_to_idx["tumor"])
    return model, tumor_idx


def analyse_image(image, model, tumor_idx):
    """Return the notebook-style tumour probability and Grad-CAM visualisations."""
    model_input = ResizePad(SIZE)(image.convert("RGB"))
    rgb = np.asarray(model_input, dtype=np.float32) / 255.0
    x = preprocess(model_input).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probability = torch.softmax(logits, dim=1)[0, tumor_idx].item()

    target_layers = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=x,
            targets=[ClassifierOutputTarget(tumor_idx)],
        )[0]

    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    heatmap_bgr = cv2.applyColorMap(
        np.uint8(np.clip(grayscale_cam, 0, 1) * 255),
        cv2.COLORMAP_JET,
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    prediction = "Tumour" if probability > THRESHOLD else "Normal"
    return {
        "prediction": prediction,
        "tumor_probability": float(probability),
        "model_input": model_input,
        "overlay": Image.fromarray(overlay),
        "heatmap": Image.fromarray(heatmap_rgb),
    }
