# Creating Datasets

The `app.core.dataset` module contains sufficient code to create datasets from a prepared sample of images.

## Overview

`idprocessing_train_dataset()` is the single public entry point. One call produces all datasets required to train the 
full pipeline — an EfficientNet document classifier dataset and a per-class YOLO segmentation dataset — organised under a single timestamped output directory. An optional merged YOLO dataset across all document classes can also be produced.

Dataset creation is reproducible via an optional random seed, and a machine-readable `info.json` summary is 
written to the output root on completion.

---

## Expected Source Structure

The source directory must contain one subdirectory per document class. Each class folder must contain an 
`images/` folder, a `labels/` folder, and a `classes.txt` file listing the YOLO segmentation class names 
(one per line):

```
source_path/
├── doc_type_0/
│   ├── images/
│   ├── labels/
│   └── classes.txt
├── doc_type_1/
│   ├── images/
│   ├── labels/
│   └── classes.txt
└── doc_type_k/
    ├── images/
    ├── labels/
    └── classes.txt
```

Images must be `.jpg`, `.jpeg`, or `.png`. Labels must be `.txt` files in YOLO format. Every image must 
have a corresponding label file with the same stem, and vice versa — unpaired files will raise a `ValueError`.

---

## Output Structure

The output is written to a timestamped subdirectory under `dest_path`, named using the format:

```
dataset_<YYYY-MM-DD@HHMM>_n<N>d<D>/
```

Where `N` is the total number of images across all classes and `D` is the number of document types.

```
dest_path/
└── dataset_<id>/
    ├── efficientnet_data/
    │   ├── train/
    │   │   ├── doc_type_0/
    │   │   └── doc_type_1/
    │   ├── val/
    │   │   ├── doc_type_0/
    │   │   └── doc_type_1/
    │   └── test/               # only if test_r > 0
    │       ├── doc_type_0/
    │       └── doc_type_1/
    ├── doc_type_0_yolo_data/
    │   ├── train/
    │   │   ├── images/
    │   │   └── labels/
    │   ├── val/
    │   │   ├── images/
    │   │   └── labels/
    │   └── data.yaml
    ├── doc_type_1_yolo_data/
    │   └── ...
    ├── merged_yolo_dataset/    # only if create_merged_yolo=True
    │   ├── train/
    │   ├── val/
    │   └── data.yaml
    └── info.json
```

---

## `idprocessing_train_dataset()`

```python
def idprocessing_train_dataset(
    source_path: str | Path,
    dest_path: str | Path = ".",
    effnet_ratios: tuple[float, float, float] = (0.8, 0.2, 0.0),
    yolo_val_frac: float = 0.2,
    create_merged_yolo: bool = False,
    seed: int | None = None,
    shuffle: bool = True
) -> None
```

### Parameters

| Parameter            | Type                         | Default           | Description                                                                                                                                                            |
|----------------------|------------------------------|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `source_path`        | `str \| Path`                | —                 | Root directory containing one subdirectory per document class. **Required.**                                                                                           |
| `dest_path`          | `str \| Path`                | `"."`             | Directory under which the timestamped dataset folder is created.                                                                                                       |
| `effnet_ratios`      | `tuple[float, float, float]` | `(0.8, 0.2, 0.0)` | Train / val / test split fractions for the EfficientNet dataset. Must sum to exactly `1.0`. Set a fraction to `0.0` to omit that split.                                |
| `yolo_val_frac`      | `float`                      | `0.2`             | Fraction of each class's images allocated to the YOLO validation split. Applied independently per document class.                                                      |
| `create_merged_yolo` | `bool`                       | `False`           | If `True`, creates an additional YOLO dataset merging images from all document classes into a single train/val split. See [Merged YOLO Dataset](#merged-yolo-dataset). |
| `seed`               | `int \| None`                | `None`            | Random seed for reproducible shuffling. Applies to both EfficientNet and YOLO splits.                                                                                  |
| `shuffle`            | `bool`                       | `True`            | Whether to shuffle images before splitting. Should generally be left as `True`.                                                                                        |

### What it produces

**1. EfficientNet dataset** (`efficientnet_data/`)

Images are copied (not symlinked) into a standard PyTorch `ImageFolder`-compatible structure — one subdirectory 
per class, per split. Only the image files are copied; label files are not included, as the class label is encoded by the directory name.

Splits are computed sequentially: training images are taken first, then validation, then any remainder goes to 
test. If `test_r = 0.0` (the default), no test split or folder is created.

**2. Per-class YOLO datasets** (`<doc_type>_yolo_data/`)

One YOLO dataset is created per document class, using that class's own `classes.txt` to populate `data.yaml`. 
Each dataset contains paired `images/` and `labels/` directories under `train/` and `val/`. A `data.yaml` is 
written automatically if `classes.txt` is present.

**3. `info.json`**

A JSON summary written to the dataset root on completion, containing document type names, per-split image 
counts for EfficientNet, and per-class image/label/class counts for YOLO. Useful for auditing or logging 
dataset provenance.

### Example

```python
from app.core.dataset import idprocessing_train_dataset

idprocessing_train_dataset(
    source_path="data/raw",
    dest_path="data/datasets",
    effnet_ratios=(0.8, 0.2, 0.0),
    yolo_val_frac=0.2,
    seed=42
)
```

---

## Merged YOLO Dataset

When `create_merged_yolo=True`, a single YOLO dataset is created by pooling images and labels from all 
document classes into a temporary staging directory before splitting. The merged `classes.txt` is the 
**canonical class list** from the first document, and the resulting `data.yaml` reflects this.

> Creating a merged YOLO dataset will create a **`_merged_yolo_source/`** directory in the output folder. This is not 
deleted by the code, but is not essential once the merged dataset has been created. This can be deleted by the user
at will.

---

## Private Helper: `_create_yolo_dataset()`

```python
def _create_yolo_dataset(
    source_path: str | Path,
    dest_path: str | Path = None,
    val_frac: float = 0.2,
    shuffle: bool = True,
    seed: int | None = None
) -> None
```

Internal function called by `idprocessing_train_dataset()` for each per-class and merged YOLO dataset. Not intended 
for direct use — prefer `idprocessing_train_dataset()`.

Expects a directory containing `images/` and `labels/` subdirectories. Validates that every image has a paired 
label and vice versa before splitting. If `classes.txt` is present in `source_path`, a `data.yaml` is written to the dataset root; if absent, a notice is printed and no YAML is created.

If `dest_path` is not specified, the dataset is created inside the source directory under `<source_stem>_dataset/`.

---

## Notes

- **Label sensitivity** — `_create_yolo_dataset()` does not validate the contents of `.txt` label files, only 
their existence. Ensure YOLO label indices are consistent with the corresponding `classes.txt` before running.
- **Image deduplication** — no deduplication is performed. If the same image filename exists in two class 
directories and `create_merged_yolo=True`, the second file will silently overwrite the first in the merged staging area.
- **Test split** — the EfficientNet test split takes all remaining images after train and val are allocated 
(i.e. it absorbs any rounding remainder). For small datasets this can cause the test set to be slightly larger 
than `test_r` implies. Default arguments result in zero test data, and implies that the function is only used on
images intended for training. The user should source testing images seperately.