"""Train a reproducible medicinal plant image classifier on DIMPSAR.

Dataset: Project-AgML/DIMPSAR_medicinal_plant_classification on Hugging Face.
The script creates a train/validation/test split, fine-tunes the last ResNet18
block and classifier, selects the best checkpoint on validation accuracy, and
reports final metrics on the held-out test split.
"""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel, Image as HFImage, load_dataset
from PIL import Image as PILImage
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

DATASET_ID = "Project-AgML/DIMPSAR_medicinal_plant_classification"
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-4
NUM_WORKERS = 0  # reliable on Windows, notebooks, and Colab
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
MODEL_PATH = ARTIFACTS / "best_model.pt"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_columns(dataset: Any) -> tuple[str, str]:
    image_col = next(
        (name for name, feature in dataset.features.items() if isinstance(feature, HFImage)),
        None,
    )
    if image_col is None:
        image_col = next(
            (name for name in dataset.column_names if "image" in name.lower() or "img" in name.lower()),
            None,
        )
    if image_col is None:
        raise ValueError(f"Couldn't identify an image column. Available columns: {dataset.column_names}")

    label_col = next(
        (
            name
            for name, feature in dataset.features.items()
            if name != image_col and isinstance(feature, ClassLabel)
        ),
        None,
    )
    if label_col is None:
        label_col = next(
            (
                name
                for name in dataset.column_names
                if name != image_col
                and any(term in name.lower() for term in ("label", "class", "species", "category"))
            ),
            None,
        )
    if label_col is None:
        raise ValueError(f"Couldn't identify the label column. Available columns: {dataset.column_names}")
    return image_col, label_col


def build_label_mapping(dataset: Any, label_col: str) -> tuple[list[str], dict[Any, int]]:
    feature = dataset.features[label_col]
    if isinstance(feature, ClassLabel):
        names = list(feature.names)
        return names, {index: index for index in range(len(names))}

    raw_values = dataset[label_col]
    unique_values = sorted(set(raw_values), key=lambda value: str(value))
    names = [str(value) for value in unique_values]
    return names, {value: index for index, value in enumerate(unique_values)}


def open_as_rgb(value: Any) -> PILImage.Image:
    """Handle decoded PIL images and the usual Hugging Face image dictionaries."""
    if isinstance(value, PILImage.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            import io

            return PILImage.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return PILImage.open(value["path"]).convert("RGB")
    if isinstance(value, (str, Path)):
        return PILImage.open(value).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value).__name__}")


class HerbalDataset(Dataset):
    def __init__(self, hf_dataset: Any, indices: np.ndarray, image_col: str,
                 label_col: str, label_to_idx: dict[Any, int], transform: Any) -> None:
        self.dataset = hf_dataset
        self.indices = [int(index) for index in indices]
        self.image_col = image_col
        self.label_col = label_col
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        row = self.dataset[self.indices[item]]
        image = open_as_rgb(row[self.image_col])
        target = self.label_to_idx[row[self.label_col]]
        return self.transform(image), int(target)


def make_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    # Fine-tune the last convolutional block and the new classification head.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    average_loss = total_loss / max(len(loader.dataset), 1)
    return average_loss, float(accuracy_score(targets, predictions))


def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    return targets, predictions


def main() -> None:
    set_seed()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading public dataset: {DATASET_ID}")

    hf_dataset = load_dataset(DATASET_ID, split="train")
    image_col, label_col = choose_columns(hf_dataset)
    class_names, label_to_idx = build_label_mapping(hf_dataset, label_col)
    encoded_labels = np.asarray([label_to_idx[value] for value in hf_dataset[label_col]], dtype=np.int64)
    class_count = len(class_names)
    if class_count < 2:
        raise ValueError(f"Expected at least 2 classes; found {class_count}.")

    all_indices = np.arange(len(hf_dataset))
    train_val_idx, test_idx = train_test_split(
        all_indices, test_size=0.20, random_state=SEED, stratify=encoded_labels
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.125,  # 10% of all examples; 70/10/20 overall split
        random_state=SEED,
        stratify=encoded_labels[train_val_idx],
    )
    print(f"Samples: total={len(all_indices)}, train={len(train_idx)}, "
          f"validation={len(val_idx)}, test={len(test_idx)}")
    print(f"Classes: {class_count}; image column: {image_col}; label column: {label_col}")

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_ds = HerbalDataset(hf_dataset, train_idx, image_col, label_col, label_to_idx, train_transform)
    val_ds = HerbalDataset(hf_dataset, val_idx, image_col, label_col, label_to_idx, eval_transform)
    test_ds = HerbalDataset(hf_dataset, test_idx, image_col, label_col, label_to_idx, eval_transform)
    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=use_pin_memory)

    model = make_model(class_count, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    started = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_accuracy)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_accuracy:.4f}"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "class_names": class_names,
                    "num_classes": class_count,
                    "architecture": "resnet18",
                    "image_size": IMAGE_SIZE,
                    "seed": SEED,
                    "dataset_id": DATASET_ID,
                    "best_epoch": best_epoch,
                    "best_validation_accuracy": best_val_accuracy,
                },
                MODEL_PATH,
            )

    if best_state is None:
        raise RuntimeError("Training did not create a valid model checkpoint.")
    model.load_state_dict(best_state)
    y_true, y_pred = predict_all(model, test_loader, device)
    test_accuracy = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(class_count)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(RESULTS / "classification_report.csv", index_label="class")
    pd.DataFrame(history).to_csv(RESULTS / "training_history.csv", index=False)

    metrics = {
        "dataset_id": DATASET_ID,
        "dataset_rows": len(hf_dataset),
        "num_classes": class_count,
        "class_names": class_names,
        "split": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "test_macro_precision": float(report["macro avg"]["precision"]),
        "test_macro_recall": float(report["macro avg"]["recall"]),
        "test_macro_f1": float(report["macro avg"]["f1-score"]),
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "device": str(device),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="Train")
    axes[0].plot([row["epoch"] for row in history], [row["val_loss"] for row in history], label="Validation")
    axes[0].set(title="Loss by epoch", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend()
    axes[1].plot([row["epoch"] for row in history], [row["train_accuracy"] for row in history], label="Train")
    axes[1].plot([row["epoch"] for row in history], [row["val_accuracy"] for row in history], label="Validation")
    axes[1].set(title="Accuracy by epoch", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "training_curves.png", dpi=160)
    plt.close(fig)

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(class_count)))
    fig, ax = plt.subplots(figsize=(18, 16))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=ax, xticks_rotation=90, colorbar=False, values_format="d")
    ax.set_title("Held-out test confusion matrix")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}; validation accuracy: {best_val_accuracy:.4f}")
    print(f"Held-out test accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics and plots saved to: {RESULTS}")
    print("Note: use the held-out test metrics in your README only after reviewing the saved report.")


if __name__ == "__main__":
    main()
