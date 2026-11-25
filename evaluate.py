"""Generate validation metrics using the same checkpoint and validation preprocessing.

Expected data layout:
data/
  val/
    normal/
    tumor/
"""
from pathlib import Path
import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

from inference import (
    BASE_DIR, CHECKPOINT_PATH, DEVICE, SIZE, MEAN, STD, ResizePad, load_model,
)

DATA_DIR = BASE_DIR / "data" / "val"
OUTPUT_PATH = BASE_DIR / "evaluation_metrics.json"
BATCH_SIZE = 16
THRESHOLD = 0.5


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {DATA_DIR}\n"
            "Create data/val/normal and data/val/tumor and place the validation patches there."
        )

    val_transform = transforms.Compose([
        ResizePad(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.ImageFolder(str(DATA_DIR), transform=val_transform)
    if not dataset.samples:
        raise ValueError(f"No validation images found in {DATA_DIR}")

    model, tumor_idx = load_model()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    expected_mapping = checkpoint.get("class_to_idx", {})
    if expected_mapping and dataset.class_to_idx != expected_mapping:
        raise ValueError(
            "Validation class mapping does not match the training checkpoint. "
            f"Checkpoint: {expected_mapping}; validation folder: {dataset.class_to_idx}"
        )
    if "normal" not in dataset.class_to_idx or "tumor" not in dataset.class_to_idx:
        raise ValueError("Expected class folders named 'normal' and 'tumor'.")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            probs = torch.softmax(logits, dim=1)[:, tumor_idx]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    probs = np.asarray(all_probs, dtype=float)
    labels = np.asarray(all_labels, dtype=int)
    tumor_label = dataset.class_to_idx["tumor"]
    normal_label = dataset.class_to_idx["normal"]
    binary_labels = (labels == tumor_label).astype(int)
    predictions = (probs > THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        binary_labels, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "Accuracy (%)": float(accuracy_score(binary_labels, predictions) * 100),
        "Balanced accuracy (%)": float(balanced_accuracy_score(binary_labels, predictions) * 100),
        "Precision (tumor)": float(precision_score(binary_labels, predictions, zero_division=0)),
        "Recall / Sensitivity": float(recall_score(binary_labels, predictions, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "F1 (tumor)": float(f1_score(binary_labels, predictions, zero_division=0)),
        "F1 (macro)": float(f1_score(binary_labels, predictions, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(binary_labels, predictions)),
    }
    if len(np.unique(binary_labels)) == 2:
        metrics["AUC (ROC)"] = float(roc_auc_score(binary_labels, probs))
        metrics["Average precision (PR)"] = float(average_precision_score(binary_labels, probs))
    else:
        metrics["AUC (ROC)"] = None
        metrics["Average precision (PR)"] = None

    output = {
        "n_samples": int(len(dataset)),
        "threshold": THRESHOLD,
        "class_to_idx": dataset.class_to_idx,
        "confusion_labels": ["normal", "tumor"],
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "metrics": metrics,
        "note": "Metrics are calculated on the validation split supplied by the user.",
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Saved validation metrics to: {OUTPUT_PATH}")
    print(f"Validation samples: {len(dataset)}")
    for name, value in metrics.items():
        print(f"{name:26s}: {'N/A' if value is None else f'{value:.4f}'}")
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")


if __name__ == "__main__":
    main()
