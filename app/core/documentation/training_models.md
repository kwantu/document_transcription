# Training Models

The `app.core.training` module provides functions and configuration dataclasses for training all models
required by the pipeline — an EfficientNet_B0 document classifier and one or more YOLO segmentation models.

## Table of Contents

1. [Overview](#overview)
2. [Configuration Dataclasses](#configuration-dataclasses)
   - [DocClassifierConfig](#docclassifierconfig)
   - [YOLOConfig](#yoloconfig)
3. [train_doc_classifier()](#train_doc_classifier)
   - [Expected Dataset Structure](#expected-dataset-structure-efficientnet)
   - [Training Behaviour](#training-behaviour)
   - [Output](#output-efficientnet)
4. [train_yolo()](#train_yolo)
   - [Expected Dataset Structure](#expected-dataset-structure-yolo)
   - [YAML Resolution](#yaml-resolution)
   - [Output](#output-yolo)
5. [Private Helpers](#private-helpers)
   - [build_transforms()](#build_transforms)
   - [build_dataloaders()](#build_dataloaders)
   - [build_model()](#build_model)
   - [compute_class_weights()](#compute_class_weights)
   - [_run_training()](#_run_training)
6. [Parameter Reference](#parameter-reference)
7. [Notes & Recommendations](#notes--recommendations)

---

## Overview

Training is split into two independent concerns: document classification and region segmentation.

The **document classifier** is an EfficientNet_B0 model pretrained on ImageNet and fine-tuned to distinguish
between document types (e.g. Smart ID vs. ID Book). It is the first stage of `full_pipeline()` and determines
which YOLO model and parameter set to use downstream.

The **YOLO segmentation models** are trained per document class to localise the `metadata`, `photo`, and
other regions within a document image. One YOLO model is trained per document type; they are not shared.

Both functions expect datasets produced by `idprocessing_train_dataset()` from `dataset.py`.

---

## Configuration Dataclasses

### `DocClassifierConfig`

Controls all aspects of the EfficientNet_B0 training run.

| Field                | Type    | Default            | Description                                                                                         |
|----------------------|---------|--------------------|-----------------------------------------------------------------------------------------------------|
| `batch_size`         | `int`   | `32`               | Number of images per training batch.                                                                |
| `learning_rate`      | `float` | `0.001`            | Initial learning rate for the AdamW optimiser.                                                      |
| `num_epochs`         | `int`   | `10`               | Number of full passes over the training set.                                                        |
| `image_size`         | `int`   | `224`              | Images are resized to `image_size × image_size` before training. Must match the EfficientNet input. |
| `num_workers`        | `int`   | `4`                | Number of subprocesses used by the PyTorch `DataLoader` for data loading.                           |
| `pretrained_weights` | `str`   | `"IMAGENET1K_V1"`  | Torchvision pretrained weight set used to initialise the backbone before fine-tuning.               |

---

### `YOLOConfig`

Controls the YOLO training run passed to Ultralytics.

| Field                | Type   | Default            | Description                                                                                                |
|----------------------|--------|--------------------|------------------------------------------------------------------------------------------------------------|
| `model_architecture` | `YOLO` | `YOLO("yolo11s.pt")` | The Ultralytics YOLO model instance to train. The default uses the small YOLOv11 segmentation architecture. |
| `num_epochs`         | `int`  | `50`               | Number of training epochs.                                                                                 |
| `batch_size`         | `int`  | `8`                | Number of images per batch. Small batches are recommended for segmentation tasks with limited GPU memory.  |
| `image_size`         | `int`  | `540`              | Input image size for YOLO training (pixels, square crop).                                                  |
| `num_workers`        | `int`  | `2`                | Number of DataLoader worker subprocesses.                                                                  |

---

## `train_doc_classifier()`

```python
def train_doc_classifier(
    dataset_path: str | Path,
    dest_path: str | Path = ".",
    cfg: DocClassifierConfig = DocClassifierConfig(),
) -> None
```

Fine-tunes an EfficientNet_B0 classifier on a labelled image dataset. The number of output classes is inferred
automatically from the directory structure of `dataset_path/train/`.

### Expected Dataset Structure (EfficientNet)

The dataset must be compatible with PyTorch's `ImageFolder` format — one subdirectory per class, nested under
`train/` and `val/` splits. This structure is produced automatically by `idprocessing_train_dataset()`.

```
dataset_path/
├── train/
│   ├── doc_type_0/
│   ├── doc_type_1/
│   └── doc_type_k/
└── val/
    ├── doc_type_0/
    ├── doc_type_1/
    └── doc_type_k/
```

### Training Behaviour

**Augmentation:** Training images are augmented with a random rotation of up to ±5° and colour jitter
(brightness and contrast ±0.2) before being normalised with standard ImageNet statistics
(`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Validation images receive only resize and
normalisation — no augmentation.

**Class imbalance:** Class weights are computed from the training set label distribution and passed to
`CrossEntropyLoss`. If one document class is over-represented, its loss contribution is down-weighted
proportionally, helping the model generalise across unbalanced datasets.

**Optimiser:** AdamW with the learning rate set in `DocClassifierConfig`. No learning rate scheduler is
currently applied.

**Architecture:** The final linear layer of the EfficientNet_B0 classifier head is replaced with a new
`nn.Linear` layer whose output dimension matches the number of classes detected in the training dataset.
All other weights are initialised from `cfg.pretrained_weights` and are unfrozen — the entire network is
fine-tuned end-to-end.

Per-epoch output during training:

```
Train loss: 0.312 | Val acc: 0.9583
```

### Output (EfficientNet)

The trained model is saved as a `.pt` checkpoint to `dest_path`. If `efficientnetb0.pt` already exists,
the file is saved with an incrementing suffix (`efficientnetb0_1.pt`, `efficientnetb0_2.pt`, etc.) to
prevent overwrites.

The checkpoint dictionary contains:

| Key           | Contents                                                  |
|---------------|-----------------------------------------------------------|
| `model_state` | `model.state_dict()` — all layer weights and biases.      |
| `classes`     | Ordered list of class names from `ImageFolder`.           |
| `config`      | The `DocClassifierConfig` instance used for this run.     |

The checkpoint format matches what `pipeline.py` expects when loading `effnet_classifier.pt` at inference
time. To deploy a newly trained classifier, rename or copy the output file to
`pipeline_models/effnet_classifier.pt`.

### Example

```python
from app.core.training import train_doc_classifier, DocClassifierConfig

train_doc_classifier(
    dataset_path="data/datasets/dataset_2025-01-01@1200_n800d2/efficientnet_data",
    dest_path="model_weights/",
    cfg=DocClassifierConfig(num_epochs=15, learning_rate=0.0005)
)
```

---

## `train_yolo()`

```python
def train_yolo(
    dataset_path: str | Path,
    dest_path: str | Path = ".",
    cfg: YOLOConfig = YOLOConfig()
) -> None
```

Trains a YOLO segmentation model on a single structured YOLO dataset. This function should be called once
per document class — each document type requires its own separately trained YOLO model.

### Expected Dataset Structure (YOLO)

The dataset must follow the standard YOLO directory layout, including a `data.yaml`. This structure is
produced automatically by `idprocessing_train_dataset()`.

```
dataset_path/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

### YAML Resolution

A known issue with Ultralytics YOLO training is that the `path` field in `data.yaml` must resolve to an
absolute path at training time. `train_yolo()` handles this automatically: it reads the existing
`data.yaml`, overwrites the `path` field with the absolute path of the dataset root, and writes a
temporary resolved copy (`_resolved_data.yaml`) alongside the original. The resolved YAML is printed to
stdout before training begins, allowing quick verification. The temporary file is not cleaned up after
training completes.

### Output (YOLO)

Training output is managed entirely by Ultralytics and is written to a `runs/` directory in the working
directory by default. This includes per-epoch metrics, confusion matrices, and the final weights at
`runs/segment/trainN/weights/best.pt` and `last.pt`. The `dest_path` parameter is accepted by the
function signature but is not currently forwarded to the Ultralytics trainer — output location must be
configured via the Ultralytics settings or the `project` argument if added to the `model.train()` call.

To deploy a trained YOLO model, copy `best.pt` from the Ultralytics run output to `pipeline_models/`
and rename it to match the expected filename (`smartid_yolo.pt` or `idbook_yolo.pt`).

### Example

```python
from app.core.training import train_yolo, YOLOConfig
from ultralytics import YOLO

# Train Smart ID segmentation model
train_yolo(
    dataset_path="data/datasets/dataset_2025-01-01@1200_n800d2/smartid_yolo_data",
    cfg=YOLOConfig(num_epochs=80, image_size=640)
)

# Train ID Book segmentation model separately
train_yolo(
    dataset_path="data/datasets/dataset_2025-01-01@1200_n800d2/idbook_yolo_data",
    cfg=YOLOConfig(num_epochs=80, image_size=640)
)
```

---

## Private Helpers

These functions are called internally by `train_doc_classifier()` and are not intended for direct use.

### `build_transforms()`

```python
def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]
```

Returns a pair of torchvision transform pipelines — one for training (with augmentation) and one for
evaluation (resize and normalise only). Both apply ImageNet normalisation as the final step.

---

### `build_dataloaders()`

```python
def build_dataloaders(dataset_path: Path, cfg: DocClassifierConfig)
    -> tuple[ImageFolder, DataLoader, DataLoader]
```

Constructs `ImageFolder` datasets and `DataLoader` instances for the `train/` and `val/` splits found
under `dataset_path`. The training loader shuffles samples each epoch; the validation loader does not.
Returns the raw `train_ds` alongside both loaders, as `train_ds` is needed downstream for class name and
weight computation.

---

### `build_model()`

```python
def build_model(num_classes: int, cfg: DocClassifierConfig, device: str) -> nn.Module
```

Instantiates an EfficientNet_B0 with pretrained weights and replaces the final classifier layer with a
new `nn.Linear(in_features, num_classes)`. The model is moved to `device` before being returned.

---

### `compute_class_weights()`

```python
def compute_class_weights(train_ds: ImageFolder, device: str) -> torch.Tensor
```

Computes per-class loss weights from the training set label distribution using inverse frequency:
`weight[i] = total_samples / count[i]`. Returns a float tensor on `device`, ready to be passed directly
to `nn.CrossEntropyLoss(weight=...)`.

---

### `_run_training()`

```python
def _run_training(
    model, train_loader, val_loader,
    criterion, optimizer,
    cfg: DocClassifierConfig,
    device: str,
    train_ds,
    dest_path: Path,
) -> None
```

The inner training loop. Runs `cfg.num_epochs` iterations of forward pass, loss computation, backprop,
and validation accuracy evaluation. Saves the model checkpoint to `dest_path` on completion.
Validation accuracy is computed as exact top-1 match over the full validation set.

---

## Parameter Reference

### Recommended Starting Configurations

| Scenario                              | `num_epochs` | `learning_rate` | `batch_size` | Notes                                              |
|---------------------------------------|--------------|-----------------|:------------:|----------------------------------------------------|
| Small dataset (<500 images per class) | `20–30`      | `0.0005`        | `16`         | Reduce LR to avoid overfitting on small sets       |
| Standard dataset                      | `10`         | `0.001`         | `32`         | Default config — suitable starting point           |
| Large dataset (>2000 images per class)| `10–15`      | `0.001`         | `64`         | Increase batch size if GPU memory allows           |

### YOLO Image Size Guidance

| Use case                  | `image_size` | Notes                                              |
|---------------------------|--------------|----------------------------------------------------|
| Default / development     | `540`        | Default config                                     |
| Higher resolution scans   | `640`        | Better for fine segmentation boundaries            |
| Memory-constrained GPU    | `416`        | Reduces VRAM usage at some cost to accuracy        |

---

## Notes & Recommendations

- **Train order matters for deployment.** The EfficientNet classifier uses class indices assigned by
`ImageFolder`, which sorts class directories alphabetically. Ensure that the class-to-index mapping
(`train_ds.class_to_idx`, printed during training) matches the `id_class` integers expected by
`GeometryConfig` and `ID_HANDLERS` in `pipeline.py`. Misalignment here will cause the pipeline to apply
the wrong parameters and YOLO model silently.

- **`train_all_models()` is not yet implemented.** The function exists in `training.py` with a `pass`
body. Training all models currently requires calling `train_doc_classifier()` and `train_yolo()` (once
per document class) in sequence manually.

- **No learning rate scheduling.** The current training loop uses a fixed learning rate for all epochs.
For longer runs or larger datasets, adding a scheduler such as `torch.optim.lr_scheduler.StepLR` or
`CosineAnnealingLR` may improve convergence.

- **`dest_path` is unused in `train_yolo()`.** Ultralytics manages its own output directory under
`runs/`. The `dest_path` argument currently has no effect and is accepted only for API consistency.
This is a known gap noted in the source.

- **Reproducibility.** Neither `train_doc_classifier()` nor `train_yolo()` accepts a random seed.
For reproducible EfficientNet runs, set `torch.manual_seed()` and `random.seed()` before calling the
function. For YOLO, refer to the Ultralytics `seed` training argument.