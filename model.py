import torch.nn as nn
from torchvision import models


def build_model(arch="resnet50"):
    """Recreate the classifier head used in the training notebook."""
    if arch == "resnet18":
        model = models.resnet18(weights=None)
    else:
        model = models.resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2),
    )
    return model
