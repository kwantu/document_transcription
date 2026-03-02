# ID Processing Pipeline (SA Documents)

> An open-source pipeline for extracting structured identity information from South African Smart ID cards and ID Books using computer vision and OCR.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Tesseract v5](https://img.shields.io/badge/Tesseract-v5-green.svg)](https://github.com/tesseract-ocr/tesseract)
[![Model: EfficientNetB0](https://img.shields.io/badge/Classifier-EfficientNetB0-orange.svg)](https://arxiv.org/abs/1905.11946)
[![Model: YOLO11s](https://img.shields.io/badge/Segmentation-YOLO11s-purple.svg)](https://github.com/ultralytics/ultralytics)

---

## Overview

This pipeline accepts a South African identity document image (or PDF) via an API and returns a structured JSON 
object containing key identity fields &mdash; including surname, forenames, and identity number. It supports 
two document classes:

- **Smart ID Card** (class `0`)
- **ID Book** (class `1`)

The pipeline is designed to run on a development server and is published as open source for review as 
a [Digital Public Good (DPG)](https://digitalpublicgoods.net/). All model weights are included in this repository.

---

## Quickstart

```python
from app.core.pipeline import full_pipeline

result = full_pipeline(
    input_path="path/to/document.jpg",
    dest_path="path/to/output/",
    save_process=False
)

print(result)
# {
#   "Surname": "SMITH",
#   "Names": "JOHN JAMES",
#   "Identity Number": "8001015009087"
# }
```

Accepted input formats: `.jpg`, `.jpeg`, `.png`, `.pdf` (first page only).

---

## How It Works

The pipeline processes each document through six sequential stages:

1. **API Ingestion** — Document image is received and normalised. PDFs are rasterised at 200 DPI.
2. **Document Classification** — An EfficientNetB0 classifier assigns a document class (Smart ID or ID Book) to the image.
3. **Region Segmentation** — A class-specific YOLO11s model localises and crops the metadata strip and photo regions.
4. **Geometry Correction** — The metadata crop is reoriented, rescaled, and deskewed using OpenCV before being passed to the OCR engine.
5. **OCR Extraction** — Tesseract v5 extracts raw text from the preprocessed metadata region.
6. **Field Formatting** — Rule-based filtering cleans the raw OCR output and extracts key fields into a structured dictionary, which is saved as a JSON file.

---

## Models

All model weights are included in the `models/` directory of this repository.

| File | Architecture | Purpose | Training data                |
|---|---|---|------------------------------|
| `efficientnet_b0_doc_classifier.pt` | EfficientNetB0 | Classifies document as Smart ID or ID Book | ~1,000 images, 2 classes     |
| `smartid_yolo.pt` | YOLO11s | Segments metadata and photo regions on Smart ID cards | ~1,000 images, 3 box classes |
| `idbook_yolo.pt` | YOLO11s | Segments metadata and photo regions on ID Books | ~1,000 images, 3 box classes |

The EfficientNetB0 classifier uses standard ImageNet normalisation and a two-class output head. Both YOLO models detect three label classes: document boundary (`0`), photo (`1`), and metadata strip (`2`).

---

## Installation

### 1. System dependency — Tesseract v5

Tesseract is a system-level install and **must be version 5**. It is not installed via pip.

**Ubuntu / Debian:**
```bash
sudo apt install tesseract-ocr
tesseract --version  # confirm v5
```

**macOS (Homebrew):**
```bash
brew install tesseract
```

**Windows:** Download the installer from the [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki).

### 2. Python dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include:

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | EfficientNetB0 document classifier |
| `ultralytics` | YOLO11s region segmentation |
| `opencv-python` | Image preprocessing |
| `pytesseract` | Tesseract Python bindings |
| `pyzbar` | Barcode decoding (ID Books) |
| `pdf2image` | PDF rasterisation (via `convert_from_path`) |

### 3. Clone the repository

```bash
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>
```

The `models/` directory containing all three weight files is included. No separate download step is required.

---

## Usage

### Basic usage

```python
from pipeline import full_pipeline

result = full_pipeline(
    input_path="document.png",
    dest_path="output/",     # output directory (created if it doesn't exist)
    save_process=False       # set True to save debug images and raw OCR
)
```

### With debug output enabled

Setting `save_process=True` writes intermediate visualisations to disk, which is useful during development or for auditing extraction results:

```python
result = full_pipeline(
    input_path="document.pdf",
    dest_path="output/",
    save_process=True
)
```

### Calling `img_2_json_v2()` directly

`full_pipeline()` uses `img_2_json_v2()` internally. You can also call it directly if you need finer control over parameters:

```python
from pipeline import load_input, load_params, SMARTID, IDBOOK, img_2_json_v2

for img_id, img in load_input("document.jpg"):
    geom_params, prep_params, post_params = load_params(img)
    yolo_model = SMARTID if geom_params.id_class == 0 else IDBOOK

    result = img_2_json_v2(
        yolo_model=yolo_model,
        img=img,
        img_id=img_id,
        dest_path="output/",
        geom_params=geom_params,
        prep_params=prep_params,
        post_params=post_params,
        save_process=True
    )
```

---

## Output

### JSON output

Each processed image produces an `output.json` in its output directory:

```json
{
  "Surname": "SMITH",
  "Names": "JOHN JAMES",
  "Identity Number": "8001015009087"
}
```

Fields may be `null` if extraction was unsuccessful for that field.

### Output directory structure

**Standard (`save_process=False`):**
```
output/<img_id>/
├── output.json
└── photo.png
```

**With debug output (`save_process=True`, v1):**
```
output/<img_id>/
├── output.json
├── photo.png
├── input.png
├── reoriented.png
├── rotated.png
├── preprocessing.png
└── raw_ocr.txt
```

**With debug output (`save_process=True`, v2):**
```
output/<img_id>/
├── output.json
├── photo.png
└── debug/
    ├── info.json         # scale factor, best preprocessing variant, OCR confidence, extraction method
    ├── raw_ocr.txt
    ├── input.png
    ├── reoriented.png
    ├── rotated.png
    └── preprocessing.png
```

---

## Project Structure

```
app/core/
├── pipeline.py           # Main entry point and pipeline orchestration
├── utils.py              # Image processing, OCR cleaning, and field formatting utilities
├── model_weights/
│   ├── doc_classifier_ENetB0.pt
│   ├── smartid_YOLO.pt
│   └── idbook_YOLO.pt
└── documentation/
    ├── technical_docs.md
    └── project_summary.md
```

---

## Supported Formats

| Format | Notes |
|---|---|
| `.jpg` / `.jpeg` | Full support |
| `.png` | Full support |
| `.pdf` | First page only |

---

## Known Limitations

- **PDF processing is limited to the first page.** This is a deliberate design assumption — identity documents 
are single-page by nature. Multi-page PDFs will have only the first page processed.
- **Barcode decoding for ID Books** is partially implemented. When a barcode is successfully decoded and passes ID 
number validation, it takes priority over OCR extraction. Full barcode integration is still in development.
- **`full_pipeline()` now uses `img_2_json_v2()`** — the confidence-based preprocessor is the active default. The 
original `img_2_json()` (v1) remains in the source as a reference but is no longer called.
- All models were trained on approximately 1,000 images per class. Performance on heavily damaged, poorly lit, 
or non-standard documents may be reduced.

---

## Licence

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
(CC BY-NC-SA 4.0)** licence.

You are free to share and adapt this work for non-commercial purposes, provided you give appropriate credit 
and distribute any derivative works under the same licence.

[View full licence →](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## Technical Documentation

For full function-level reference — including all configurable parameters, algorithm descriptions, and 
tuning guidance — see [`technical_docs.md`](./DOCUMENTATION.md).

