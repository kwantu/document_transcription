# ID Processing Pipeline — Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [pipeline.py](#pipelinepy)
   - [Dataclasses & Configuration](#dataclasses--configuration)
   - [Model Loading](#model-loading)
   - [load_params()](#load_params)
   - [load_input()](#load_input)
   - [_legacy_img_2_json](#_legacy_img_2_json)
   - [img_2_json_v2()](#img_2_json_v2)
   - [full_pipeline()](#full_pipeline)
4. [utils.py](#utilspy)
   - [display()](#display)
   - [pdf2img()](#pdf2img)
   - [reorient_img()](#reorient_img)
   - [deskew_img()](#deskew_img)
   - [rescale()](#rescale)
   - [clean_raw_ocr_output()](#clean_raw_ocr_output)
   - [search_for_line()](#search_for_line)
   - [barcode_id_num()](#barcode_id_num)
   - [format_fields_smartid()](#format_fields_smartid)
   - [format_fields_idbook()](#format_fields_idbook)
   - [ocr_to_dict_smartid() / ocr_to_dict_idbook()](#ocr_to_dict_smartid--ocr_to_dict_idbook)
   - [validate_id()](#validate_id)
5. [Parameter Reference](#parameter-reference)
6. [Known Limitations & Development Notes](#known-limitations--development-notes)

---

## Overview

This pipeline accepts a document image (or PDF) and returns a structured JSON object containing key identity fields 
extracted via OCR. It supports two document classes: **South African Smart ID cards** (class `0`) and **ID Books** 
(class `1`).

Processing proceeds through six stages: document classification (EfficientNet_B0), region segmentation (YOLO), 
geometry correction, image preprocessing, OCR extraction, and rule-based field formatting.

---

## Project Structure

```
project/
├── pipeline.py       # Main entry point, model loading on import, pipeline orchestration
├── utils.py          # Image processing, OCR cleaning, field formatting helpers
├── dataset.py
├── training.py
└── pipeline_models/
    ├── effnet_classifier.pt
    ├── smartid_yolo.pt
    └── idbook_yolo.pt
```

---

## pipeline.py

### Dataclasses & Configuration

Three dataclasses parametrise the pipeline. They are instantiated by `load_params()` based on the detected document 
class and passed into `img_2_json_v2()`.

---

#### `GeometryConfig`

Controls geometry correction behaviour.

| Field                    | Type    | Default          | Description                                                                                                                                                                                       |
|--------------------------|---------|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `id_class`               | `int`   | _Required Field_ | Document class (`0` = Smart ID, `1` = ID Book). **Required.**                                                                                                                                     |
| `metadata_target_height` | `int`   | `400`            | Target pixel height for the metadata crop after rescaling.                                                                                                                                        |
| `correction_angle`       | `float` | `0`              | The expected angle (in degrees from +x) between the `metadata` and `person` label centres on a correctly-oriented document. Used by `reorient_img()` to determine whether and how much to rotate. |

**Recommended values:**

| Document | `metadata_target_height` | `correction_angle` |
|----------|--------------------------|--------------------|
| Smart ID | `440`                    | `10`               |
| ID Book  | `380`                    | `-100`             |

---

#### `PreprocessConfig`

Controls OpenCV image preprocessing applied before OCR.

| Field                   | Type              | Default       | Description                                                                                                                                                        |
|-------------------------|-------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `denoise_type`          | `str`             | `"bilateral"` | Denoising method. `"bilateral"` preserves edges; `"gaussian"` is faster but softer.                                                                                |
| `k_denoise`             | `int`             | `3`           | Kernel size for Gaussian blur (only used if `denoise_type="gaussian"`). Must be odd.                                                                               |
| `bilateral_d`           | `int`             | `5`           | Diameter of each pixel neighbourhood for bilateral filtering.                                                                                                      |
| `bilateral_sigma_color` | `int`             | `20`          | Filter sigma in colour space. Higher values blend more dissimilar colours together.                                                                                |
| `bilateral_sigma_space` | `int`             | `20`          | Filter sigma in coordinate space. Higher values blend pixels further apart.                                                                                        |
| `thresh_block`          | `int`             | `13`          | Block size for adaptive Gaussian thresholding. Must be odd and > 1.                                                                                                |
| `thresh_c`              | `int`             | `15`          | Constant subtracted from the mean in adaptive thresholding. Higher values produce a darker (denser) binary image. **This is the most sensitive tuning parameter.** |
| `morph_kernel`          | `tuple[int, int]` | `(2, 1)`      | Kernel shape for the morphological CLOSE operation that thickens thin text strokes.                                                                                |
| `ocr_psm`               | `int`             | `6`           | Tesseract page segmentation mode (PSM). PSM 6 assumes a uniform block of text. See [Tesseract PSM reference](#tesseract-psm-modes).                                |

**Recommended `thresh_c` values by document type:**

| Document | `thresh_c` |
|----------|------------|
| Smart ID | `4`        |
| ID Book  | `20`       |

---

#### `PostprocessConfig`

Controls OCR text cleaning and field extraction.

| Field           | Type        | Default                                   | Description                                                                                                |
|-----------------|-------------|-------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `allowed_chars` | `set`       | ASCII letters + digits + `"- :"`          | Characters retained during cleaning. Everything else is replaced with `filler_char`.                       |
| `filler_char`   | `str`       | `""`                                      | Replacement for disallowed characters. Empty string removes noise entirely.                                |
| `field_list`    | `list[str]` | `["Surname", "Names", "Identity Number"]` | Target fields for extraction.                                                                              |
| `confidence`    | `float`     | `0.4`                                     | Minimum `SequenceMatcher` ratio for a line to be accepted as a field header match in `search_for_line()`.  |

---

#### `ImageParams`

Required class for `full_pipeline()`. Param loader returns `ImageParams` dataclass with no default values.

| Field                | Type                | Description                                                        |
|----------------------|---------------------|--------------------------------------------------------------------|
| `doc_class_info`     | `dict`              | Inferred document class and confidence from EfficientNet_B0 Model. |
| `geometry_config`    | `GeometryConfig`    | Controls geometry correction behaviour                             |
| `preprocess_config`  | `PreprocessConfig`  | OpenCV image preprocessing parameters.                             |
| `postprocess_config` | `PostprocessConfig` | OCR text cleaning and field extraction parameters.                 |

---

### Model Loading

Models are loaded once at module import time and assigned to module-level globals.

```python
DEVICE   # "cuda" if available, else "cpu"
SMARTID  # YOLO model for Smart ID segmentation
IDBOOK   # YOLO model for ID Book segmentation
DOC_DETECTION  # EfficientNetB0 document classifier (2-class)
```

The EfficientNetB0 classifier uses standard ImageNet normalisation 
(`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Model weights are expected in a `models/` directory 
co-located with `pipeline.py`.

The mapping from class integer to post-processing handler is defined in:

```python
ID_HANDLERS = {
    0: ocr_to_dict_smartid,
    1: ocr_to_dict_idbook
}
```

---

### `load_params()`

```python
def load_params(img: np.ndarray) -> ImageParams
```

Runs the input image through the EfficientNetB0 classifier and returns the appropriate configuration dataclasses for 
the detected document class. This is the first stage of the full pipeline.

**Preprocessing applied before inference:** resized to 224×224, converted to RGB, normalised with ImageNet statistics.

**Returns:** `(GeometryConfig, PreprocessConfig, PostprocessConfig)` — raises `ValueError` if the predicted class is not `0` or `1`.

---

### `load_input()`

```python
def load_input(img_path: str | Path) -> list[tuple[str, np.ndarray]]
```

Normalises file input into a list of `(img_id, img)` pairs, where `img_id` is the filename stem and `img` is a BGR 
NumPy array. Supports `.jpg`, `.jpeg`, `.png`, and `.pdf`.

PDFs are rendered at 200 DPI. Only the first page is currently processed.

**Raises:** `ValueError` for unsupported file types; `RuntimeError` if a raster image cannot be read.

---

### ~~`_legacy_img_2_json()`~~

```python
def _legacy_img_2_json(
    yolo_model, img, img_id,
    dest_path=None,
    geom_params=None, prep_params=None, post_params=None,
    save_process=True, save_params=False
) -> dict[str, Any]
```

The core v1 pipeline function. Processed a single image end-to-end and wrote output to disk.

**Processing stages:**

1. YOLO inference to detect `metadata` (cls `2`) and `photo` (cls `1`) regions.
2. `reorient_img()` — 90°/180°/270° rotation correction.
3. For ID Books: `barcode_id_num()` is attempted on the reoriented image before any further cropping.
4. Metadata and photo crops are extracted and rotated to match the corrected orientation.
5. `rescale()` — scales metadata to `geom_params.metadata_target_height`.
6. `deskew_img()` — small-angle text-contour deskew.
7. Grayscale → denoise (bilateral or Gaussian) → adaptive threshold → morphological CLOSE.
8. Tesseract OCR with configured PSM.
9. `ocr_to_dict_*()` — cleaning, field search, fallbacks.
10. JSON written to `<dest_path>/<img_id>/output.json`.

**Output directory structure (when `save_process=True`):**

```
<dest_path>/<img_id>/
├── output.json          # Extracted fields
├── photo.png            # Cropped photo region
├── input.png            # Original image
├── reoriented.png       # Post-reorientation visualisation
├── rotated.png          # Post-deskew visualisation
├── preprocessing.png    # Metadata, denoised, binary, OCR text panel
└── raw_ocr.txt          # Raw Tesseract output
```

If `save_params=True`, a `params.json` is also written containing all config values and runtime metadata (scale factor, 
detected classes, extraction method, etc.).

---

### `img_2_json_v2()`

```python
def img_2_json_v2(
    yolo_model, img, img_id,
    dest_path=None,
    geom_params=None, prep_params=None, post_params=None,
    save_process=True
) -> dict[str, str]
```

An updated pipeline variant that replaces the fixed preprocessing chain with **confidence-based variant selection**. 
Rather than applying a single preprocessing path, four variants are generated and evaluated; the one yielding the 
highest mean Tesseract word confidence is selected.

**Preprocessing variants:**

| Variant key      | Description                                                  |
|------------------|--------------------------------------------------------------|
| `gray`           | Raw grayscale only                                           |
| `clahe`          | CLAHE histogram equalisation (clip limit 2.0, 8×8 tile grid) |
| `clahe_otsu`     | CLAHE followed by Otsu global thresholding                   |
| `bilateral_otsu` | Bilateral filter (d=9, σ=75) followed by Otsu thresholding   |

CLAHE (`clahe`) is particularly effective on low-contrast or unevenly lit IDs.

The barcode path and field extraction logic are identical to v1. If a barcode is decoded and passes `validate_id()`, 
it takes precedence over the OCR extraction method.

**Debug output structure (when `save_process=True`):**

```
<dest_path>/<img_id>/
├── output.json
├── photo.png
└── debug/
    ├── info.json            # Scale factor, best variant, confidence values, extraction method etc
    ├── raw_ocr.txt
    ├── input.png
    ├── reoriented.png
    ├── rotated.png
    └── preprocessing.png    # Metadata, best variant, OCR text panel
```

`info.json` example:
```json
{
  "id_class": 0,
  "id_class_confidence": 0.9997946619987488,
  "scale_factor": 3.0555555555555554,
  "rescaled_metadata_shape": {
    "w": 232,
    "h": 440
  },
  "id_num_extraction_method": "field_search",
  "found_photo": true,
  "metadata_confidence": 0.9185846447944641,
  "photo_confidence": 0.913300633430481,
  "best_ocr_conf": 72.66666666666667,
  "best_ocr_variant": "gray"
}
```

---

### `full_pipeline()`

```python
def full_pipeline(
    input_path: str,
    dest_path: str | Path | None = None,
    save_process: bool = False
) -> tuple[dict[str, str], bool]
```

The public entry point. Wraps `load_input()`, `load_params()`, YOLO model selection, and `img_2_json_v2()` into a single 
call.

Returns a dictionary keyed by information fields, and a boolean variable suggesting ID number validity (`utils.validate_id_num()`).

> **Note:** `full_pipeline()` currently uses `img_2_json_v2()`, making the more brittle `_legacy_img_2_json()` (v1) redundant.

---

## utils.py

### `display()`

```python
def display(img, dpi=80.0, cmap=None, show=True) -> plt.Figure
```

Renders an image with matplotlib, scaling the figure to preserve the native pixel dimensions at the given DPI. 
Accepts a NumPy array or a file path. Returns the figure for optional downstream saving.

---

### `pdf2img()`

```python
def pdf2img(pdf_path: str | Path, dpi: int = 200) -> np.ndarray
```

Converts the first page of a PDF to a BGR NumPy array at 200 DPI using `pdf2image`, is called by `load_input()`, which 
is _then_ called by `full_pipeline()`

---

### `reorient_img()`

```python
def reorient_img(result, correction_angle=0.0, show=False, return_fig=False)
    -> tuple[np.ndarray, tuple[float, Optional[int]], Optional[Figure]]
```

Corrects 90°/180°/270° orientation errors using the spatial relationship between YOLO-detected bounding boxes.

**Algorithm:**

1. Computes the centres of the `metadata` (cls `2`) and `person` (cls `1`) boxes.
2. Calculates the angle `θ` of the vector from `metadata` to `person` centre with respect to the +x-axis (corrected 
for array y-axis inversion).
3. Computes `delta = (θ − correction_angle + 360) % 360` — the angular offset from the expected orientation.
4. Applies a 90°/180°/270° OpenCV rotation based on which quadrant `delta` falls into. No rotation is applied 
if `delta < 45°`.

`correction_angle` encodes what `θ` should be for a correctly-oriented document. For example, if the person photo 
should be to the right of the metadata strip, `correction_angle ≈ 0°`.

**Returns:** `(rotated_img, (delta, cv2_rotation_code), fig_or_None)`

---

### `deskew_img()`

```python
def deskew_img(img, show=False, return_fig=False)
    -> tuple[np.ndarray, float, Optional[Figure]]
```

Performs small-angle deskew by fitting `minAreaRect` boxes to dilated text-line contours.

**Algorithm:**

1. Grayscale + Otsu binarisation (text = white).
2. Morphological OPEN to remove noise specks; kernel size is heuristic (`max(3, min(h,w)//300)`).
3. Horizontal dilation to merge individual characters into line blobs; kernel width is `max(15, w//40)`.
4. Padding added before contour detection to prevent edge clipping.
5. `cv2.minAreaRect` angles collected from all contours exceeding 0.5% of total image area.
6. Mean angle computed and applied via `cv2.warpAffine` with cubic interpolation and border replication.

Contours below the area threshold are drawn in red in the debug visualisation; valid ones in green.

**Returns:** `(deskewed_img, angle_degrees, fig_or_None)`

---

### `rescale()`

```python
def rescale(img, target_height=400, max_scale_factor=4.0, interpolation=cv2.INTER_LANCZOS4)
    -> tuple[np.ndarray, float]
```

Upscales or downscales the image so that its height matches `target_height`. Lanczos4 interpolation is used by 
default for quality upscaling. A warning is printed if the scale factor exceeds `max_scale_factor` (default 4.0), 
since text quality is likely too poor for reliable OCR at that point.

**Returns:** `(rescaled_img, scale_factor)`

---

### `clean_raw_ocr_output()`

```python
def clean_raw_ocr_output(text, allowed_chars=None, filler_char="") -> str
```

Three-step text sanitiser:

1. Splits on newlines, strips leading/trailing whitespace, discards empty lines.
2. Replaces any character not in `allowed_chars` with `filler_char`.
3. Rejoins lines with `\n`.

Default `allowed_chars`: ASCII letters and digits, space, hyphen, colon.

---

### `search_for_line()`

```python
def search_for_line(text, line, confidence=0.4) -> tuple[int | None, float]
```

Finds the line in `text` that best matches `line` using `difflib.SequenceMatcher`. Returns the index and score of 
the best match, or `(None, score)` if the best score is below `confidence`.

Used by field formatting functions to locate label rows such as `"Surname:"` or `"Identity Number:"` in OCR output, 
tolerating common OCR substitutions.

---

### `barcode_id_num()`

```python
def barcode_id_num(img) -> str | None
```

Decodes the first barcode found in the image using `pyzbar`, returning its data as a UTF-8 string. Returns `None` 
if no barcode is detected. Operates on a grayscale conversion of the input.

Used in the ID Book path, where the barcode encodes the identity number directly.

---

### `try_extract_id()`

```python
def try_extract_id(s: str) -> str | None
```

Extract and normalise an integer 13-digit ID from a string. Important function used to extract ID number from each 
line from clean OCR text.

---

### `format_fields_smartid()`

```python
def format_fields_smartid(text, confidence=0.4) -> tuple[dict[str, str], str]
```

Extracts `Surname`, `Names`, and `Identity Number` from cleaned Smart ID OCR text using a two-stage approach:

**Stage 1 — Field search:** Locates each field label (e.g. `"Identity Number:"`) via `search_for_line()` and 
reads the following line as the value. For `Identity Number`, extracted digits must be exactly 13 characters.

**Stage 2 — Numeric fallback (if ID number not found in stage 1):** Scores each line by the proportion of digit 
characters and selects the most numeric one.

| Fallback outcome                  | Method string          |
|-----------------------------------|------------------------|
| Exact 13-digit match              | `"numeric:absolute"`   |
| >13 digits, truncated to first 13 | `"numeric:truncation"` |
| Both stages failed                | `"unsuccessful"`       |

---

### `format_fields_idbook()`

```python
def format_fields_idbook(text, confidence=0.4) -> tuple[dict[str, str], str]
```

Extracts `Identity Number`, `Surname` (from `VAN/SURNAME:`), and `Names` (from `VOORNAME/FORENAMES:`) from 
cleaned ID Book OCR text.

**ID number extraction cascade:**

**Stage 1 — First-line Search:** Assume the ID number lies on the first line of the OCR, idea due to format of 
document. `method = "first_row"`

**Stage 2 — Look for 'I.D. No' in string:** Search for the string that best matches our ID number entry explicitly. 
`method = "string_match_search"` if the search came up successful.

**Stage 3 — Numeric Fallback** using `try_extract_id()`. `method = "numeric_scoring"`

Whitespace in the printed ID number on ID Books means raw digit extraction from the first OCR line is 
generally reliable. Field names are renamed on return to match the Smart ID convention (`"Surname"`, `"Names"`) in 
**stage 4 — field search**.

---

### `ocr_to_dict_smartid()` / `ocr_to_dict_idbook()`

```python
def ocr_to_dict_smartid(text, allowed_chars=None, filler_char="", confidence=0.5)
    -> tuple[dict[str, str], str]

def ocr_to_dict_idbook(text, allowed_chars=None, filler_char="", confidence=0.5)
    -> tuple[dict[str, str], str]
```

Convenience wrappers that call `clean_raw_ocr_output()` followed by the appropriate `format_fields_*()` 
function. These are the functions registered in `ID_HANDLERS` and called by the pipeline.

---

### `validate_id()`

```python
def validate_id(id_number: str) -> bool
```

Validates a South African ID number against three criteria:

1. **Format:** Must be exactly 13 digits.
2. **Date of birth:** Digits `0–5` must encode a valid `YYMMDD` date (century is inferred: `> current year % 100` 
→ 1900s, otherwise 2000s).
3. **Luhn checksum:** Standard Luhn algorithm applied to all 13 digits.

Used in `img_2_json_v2()` to gate barcode results before accepting them as the canonical identity number.

---

## Parameter Reference

### Tesseract PSM Modes

| PSM  | Description                                 | Recommended use               |
|------|---------------------------------------------|-------------------------------|
| `3`  | Fully automatic page segmentation           | Multi-block documents         |
| `6`  | Assume a single uniform block of text       | **Default — metadata strips** |
| `7`  | Treat image as a single text line           | Narrow single-field crops     |
| `11` | Sparse text — find as much text as possible | Noisy / partial crops         |

### Key Tuning Parameters Summary

| Parameter                | Location            | Impact                                                   |
|--------------------------|---------------------|----------------------------------------------------------|
| `thresh_c`               | `PreprocessConfig`  | Most sensitive to document type and lighting             |
| `correction_angle`       | `GeometryConfig`    | Critical for reorientation to work correctly             |
| `metadata_target_height` | `GeometryConfig`    | Controls text pixel height entering OCR (~25–30px ideal) |
| `confidence`             | `PostprocessConfig` | Trade-off between field match precision and recall       |
| `ocr_psm`                | `PreprocessConfig`  | Adjust if OCR segments incorrectly                       |

---

## Known Limitations & Development Notes

- **`full_pipeline()` uses `img_2_json_v2()`,** the more brittle `img_2_json()` function is still embedded in the 
source code, but only as a relic.
- **Single page PDFs only.** `load_input()` processes only the first page of a multipage PDF, and is a suitably strong 
assumption made by project leaders. This can easily be corrected, however, and is only an advisory note.
- **`deskew_img()` noise kernel** is described in code as a "pretty heuristic" (`max(3, min(h,w)//300)`) and 
may benefit from further tuning on a wider range of scan qualities.
- **ID Book first-row logic** assumes the identity number always appears on the first OCR line after 
preprocessing. Layout variations (e.g. damaged documents, unusual crops) may cause this to fail and fall 
through to numeric scoring, which isn't a huge issue.
- **Few instances of changing `PostprocessConfig`** as the default values are suited to all NLP tasks encountered.