from pathlib import Path
from dataclasses import dataclass
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from ultralytics import YOLO
from tqdm import tqdm
import yaml


# --------------------------------------------------
# Model Config & Environment variables --- Load on import
# --------------------------------------------------

@dataclass
class DocClassifierConfig:
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 10
    image_size: int = 224
    num_workers: int = 4
    pretrained_weights: str = "IMAGENET1K_V1"

@dataclass
class YOLOConfig:
    model_architecture: YOLO = YOLO("yolo11s.pt")
    num_epochs: int = 50
    batch_size: int = 8
    image_size: int = 540
    num_workers: int = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Builders
# --------------------------------------------------

def build_transforms(image_size: int):
    imagenet_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        imagenet_norm,
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        imagenet_norm,
    ])

    return train_tfms, eval_tfms


def build_dataloaders(dataset_path: Path, cfg: DocClassifierConfig):
    train_tfms, eval_tfms = build_transforms(cfg.image_size)

    train_ds = datasets.ImageFolder(dataset_path / "train", transform=train_tfms)
    val_ds   = datasets.ImageFolder(dataset_path / "val",   transform=eval_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    return train_ds, train_loader, val_loader


def build_model(num_classes: int, cfg: DocClassifierConfig, device: str):
    model = models.efficientnet_b0(weights=cfg.pretrained_weights)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )
    return model.to(device)


def compute_class_weights(train_ds, device: str):
    counts: Counter[int] = Counter(train_ds.targets)
    total = sum(counts.values())

    weights = torch.tensor(
        [total / counts[i] for i in range(len(counts))],
        dtype=torch.float32,
    )
    return weights.to(device)


# --------------------------------------------------
# Training
# --------------------------------------------------

def _run_training(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        cfg: DocClassifierConfig,
        device: str,
        train_ds,
        dest_path: Path,
):

    for epoch in range(cfg.num_epochs):

        # --- TRAIN ---
        model.train()
        train_loss = 0.0

        for x, y in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.num_epochs}",
        ):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- VALIDATE ---
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out: torch.Tensor = model(x)
                preds: torch.Tensor = out.argmax(dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total

        print(
            f"  Train loss: {train_loss:.3f} | Val acc: {val_acc:.4f}"
        )

    # --- SAVE ---
    dest_path.mkdir(parents=True, exist_ok=True)
    model_path = dest_path / "efficientnetb0.pt"
    i = 1
    while model_path.exists():
        model_path = dest_path / f"efficientnetb0_{i}.pt"

    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": train_ds.classes,
            "config": cfg,
        },
        model_path,
    )

    print(f"(!) Model saved to {dest_path}")

def train_doc_classifier(
        dataset_path: str | Path,
        dest_path: str | Path = ".",
        cfg: DocClassifierConfig = DocClassifierConfig(),
) -> None:
    """
    Call this function to train an EfficientNet_B0 document classifier.

    The training sequence expects a data structure like:
    dataset_path/
    ├── train/
    │   ├── doc_type_0/
    │   ├── doc_type_1/
    │   │ ...
    │   └── doc_type_k/
    └── val/
        ├── doc_type_0/
        ├── doc_type_1/
        │ ...
        └── doc_type_k/
    :param dataset_path: Path to the dataset with above structure
    :param dest_path: Path to output destination
    :param cfg: The configuration set by the DocClassifierConfig() dataclass
    :return: None
    """

    dataset_path = Path(dataset_path)
    dest_path = Path(dest_path)

    # Build data
    train_ds, train_loader, val_loader = build_dataloaders(dataset_path, cfg)

    num_classes = len(train_ds.classes)

    print("Classes:", train_ds.classes)
    print("Mapping:", train_ds.class_to_idx)

    # Model
    model = build_model(num_classes, cfg, DEVICE)

    # Class weights
    class_weights = compute_class_weights(train_ds, DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # Training loop
    _run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        cfg=cfg,
        device=DEVICE,
        train_ds=train_ds,
        dest_path=dest_path,
    )


# Now, the YOLO funciton (a lot easier)
def train_yolo(
        dataset_path: str | Path,
        dest_path: str | Path = ".",
        cfg: YOLOConfig = YOLOConfig()
) -> None:
    """
    Call this function to train a YOLO model on pre-structured YOLO data. This function may be used multiple times
    during training model weights for an implementation, each YOLO model should be a result of this method
    individually.

    Expected data structure:
    dataset_path/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml
    :param dataset_path: Path to the dataset with above structure
    :param dest_path: Path to output destination
    :param cfg: The configuration set by the YOLOConfig() dataclass
    :return: None
    """

    dataset_path = Path(dataset_path)
    dest_path = Path(dest_path)

    # resolve tricky YAML location annoyingness
    yaml_path = dataset_path / "data.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"{yaml_path} not found")

    with yaml_path.open() as f:
        data_cfg = yaml.safe_load(f)

    # FORCE the path to be absolute dataset root
    data_cfg["path"] = str(yaml_path.parent.resolve())

    # write a temporary resolved yaml
    resolved_yaml = yaml_path.parent / "_resolved_data.yaml"
    with resolved_yaml.open("w") as f:
        yaml.safe_dump(data_cfg, f)

    # Display yaml to user
    print(resolved_yaml.read_text())

    # Train the model...
    model = cfg.model_architecture
    model.train(
        data=str(resolved_yaml),
        epochs=cfg.num_epochs,
        imgsz=cfg.image_size,
        batch=cfg.batch_size,
        workers=cfg.num_workers
    )


# All models
def train_all_models(
        data_root: str | Path,
        dest_path: str | Path = ".",
        doc_cfg: DocClassifierConfig = DocClassifierConfig(),
        yolo_cfg: list[YOLOConfig] = None,
) -> None:
    pass