# 'idprocessing' API

The current stage 1 architecture of the API is as follows:

1. OCR pipeline `pipeline.py`, collecting YOLO inference, image preprocessing, OCR extraction, and OCR &rarr; JSON formatting
2. Image utilities `utils.py`, any helper functions that `pipeline.py` needs to run smoothly.

---

# Document OCR Pipeline

This module provides an end-to-end pipeline to extract structured information from document images (e.g., SmartID, ID book) using YOLO object detection, image preprocessing, OCR, and post-processing.

---

## Configuration Dataclasses

### `GeometryConfig`
Stores geometry-specific parameters for the image pipeline.

**Fields:**  
- `id_class` (`int`): Identifier for type of document (default `0` for SmartID).  
- `metadata_target_height` (`int`): Height in pixels for the metadata region (default `400`).  
- `correction_angle` (`float`): Angle offset for proper reorientation (default `0.0`).

---

### `PreprocessConfig`
Parameters for image preprocessing before OCR.

**Fields:**  
- `k_denoise` (`int`): Kernel size for Gaussian denoising (default `3`).  
- `thresh_block` (`int`): Block size for adaptive thresholding (default `13`).  
- `thresh_c` (`int`): Constant for adaptive thresholding (default `3`).  
- `morph_kernel` (`tuple[int, int]`): Morphological kernel size (default `(2,1)`).  
- `ocr_psm` (`int`): Page segmentation mode for Tesseract OCR (default `6`).

---

### `PostprocessConfig`
Parameters for cleaning and formatting OCR results.

**Fields:**  
- `allowed_chars` (`set`): Valid characters for OCR cleaning. Defaults to alphanumeric + `- :`.  
- `filler_char` (`str`): Character to replace invalid characters (default empty).  
- `field_list` (`list[str]`): Fields to extract from OCR (default `["Surname", "Names", "Identity Number"]`).  
- `confidence` (`float`): Confidence threshold for line matching (default `0.4`).

---

## ID Handlers

Dictionary mapping `id_class` to a field extraction function:

```python
ID_HANDLERS = {
    0: ocr_to_dict_smartid,
    # 1: ocr_to_dict_idbook
}
```

---

## `img_2_json`

Full end-to-end pipeline from image to structured JSON output.

**Signature:**

```python
img_2_json(
    yolo_model: YOLO,
    img_path: str,
    dest_path: str | None = None,
    geom_params: GeometryConfig | None = None,
    prep_params: PreprocessConfig | None = None,
    post_params: PostprocessConfig | None = None,
    save_process: bool = True
) -> dict[str, Any]
```

**Steps Performed:**  
1. Run YOLO to detect `metadata` and `photo`.  
2. Reorient image based on YOLO-detected metadata and correction angle.  
3. Rescale metadata region to target height.  
4. Deskew small-angle text misalignment.  
5. Convert to grayscale, denoise, binarize, and thicken text.  
6. Run Tesseract OCR.  
7. Save intermediate process images (optional).  
8. Clean OCR text and extract structured fields.  
9. Save all parameters, outputs, and extracted fields as JSON.

**Parameters:**  
- `yolo_model` (`YOLO`): YOLO model instance.  
- `img_path` (`str` | `Path`): Path to input image.  
- `dest_path` (`str` | `Path` | `None`): Directory to save outputs. Defaults to image parent folder.  
- `geom_params` (`GeometryConfig` | `None`): Geometry configuration.  
- `prep_params` (`PreprocessConfig` | `None`): Preprocessing configuration.  
- `post_params` (`PostprocessConfig` | `None`): Postprocessing configuration.  
- `save_process` (`bool`): Save intermediate figures for debugging (default `True`).

**Returns:**  
- `output_dict` (`dict[str, str]`): Dictionary of extracted fields, e.g.:

```json
{
  "Surname": "SMITH",
  "Names": "JOHN",
  "Identity Number": "0123456789012"
}
```

**Notes:**  
- Raises `RuntimeError` if metadata cannot be detected.  
- Handles duplicate output directories by appending `_1`, `_2`, etc.  
- Saves the following in `dest_path/<image_stem>/`:  
  - `photo.png`
  - `raw_ocr.txt`  
  - `reoriented.png`, `rotated.png`, `preprocessing.png` (optional, if `save_process=True`)  
  - `params.json` (all pipeline parameters)  
  - `output.json` (final structured fields)  

---

## Usage Example

```python
from ultralytics import YOLO
from pipeline import img_2_json, GeometryConfig, PreprocessConfig, PostprocessConfig

yolo_model = YOLO("yolo11s.pt")

geom = GeometryConfig(id_class=0, metadata_target_height=420, correction_angle=10.0)
prep = PreprocessConfig(k_denoise=3, thresh_block=13, thresh_c=3)
post = PostprocessConfig(confidence=0.5)

result = img_2_json(
    yolo_model,
    "path/to/image.jpg",
    dest_path="output",
    geom_params=geom,
    prep_params=prep,
    post_params=post
)

print(result)
```

---

# Image Processing Utilities

This module provides a set of utilities for image handling, preprocessing, and OCR post-processing/formatting. It contains essential reorientation, deskewing, rescaling, and field extraction structures.

The above pipeline relies on many functions included inside this module.

---

### Functions

### `display(img, dpi=80.0, cmap=None, show=True) -> plt.Figure`

Simply displays an image using Matplotlib.

**Parameters:**  
- `img`: `np.ndarray` or path-like (`str` | `Path`) image.  
- `dpi`: Dots per inch for figure sizing (default `80.0`).  
- `cmap`: Optional colormap for plotting.  
- `show`: If `True`, displays the image immediately.  

**Returns:**  
- `matplotlib.figure.Figure`: Figure containing the image.  

---

### `reorient_img(result, correction_angle=0.0, show=False, return_fig=False) -> tuple[np.ndarray, float, Optional[Figure]]`
Reorients an image using YOLO-detected metadata/person positions and a correction angle. Handles 90°, 180°, 270° rotations.  

**Parameters:**  
- `result`: `YOLO` result object containing `orig_img` and `boxes`.  
- `correction_angle`: Reference angle for proper orientation (default `0.0`).  
- `show`: Visualize orientation arrows if `True`.  
- `return_fig`: Return a figure of the process if `True`.  

**Returns:**  
- `rotated_image`: NumPy array of the rotated image.  
- `delta`: Offset angle used for reorientation.  
- `fig`: Optional figure visualizing reorientation.

---

### `deskew_img(img, show=False, return_fig=False) -> tuple[ndarray, float, Optional[Figure]]`
Performs small-angle deskewing using contours of text lines.  

**Parameters:**  
- `img`: NumPy image tensor.  
- `show`: Display debug plots if `True`.  
- `return_fig`: Return debug figure if `True`.  

**Returns:**  
- `rotated`: Deskewed image.  
- `theta`: Calculated deskew angle.  
- `fig`: Optional figure showing processing steps.

---

### `rescale(img, target_height=400, max_scale_factor=4.0, interpolation=cv2.INTER_LANCZOS4) -> tuple[np.ndarray, float]`
Rescales an image to a target height while preserving aspect ratio.  

**Parameters:**  
- `img`: NumPy image tensor.  
- `target_height`: Desired height in pixels (default `400`).  
- `max_scale_factor`: Prints warning if scaling exceeds this factor (default `4.0`).  
- `interpolation`: OpenCV interpolation method (default `INTER_LANCZOS4`).  

**Returns:**  
- `rescaled`: Rescaled image.  
- `sf`: Scale factor applied.

---

### `clean_raw_ocr_output(text, allowed_chars=None, filler_char="") -> str`
Cleans OCR output by removing empty lines, stripping whitespace, and filtering characters.  

**Parameters:**  
- `text`: Raw OCR string.  
- `allowed_chars`: Optional set of valid characters (defaults to alphanumeric + `- :`).  
- `filler_char`: Replaces invalid characters (default empty string).  

**Returns:**  
- Cleaned OCR string.

---

### `search_for_line(text, line, confidence=0.4) -> tuple[int, float] | tuple[None, float]`
Finds the line best matching a target string using sequence matching.  

**Parameters:**  
- `text`: Cleaned OCR text.  
- `line`: Target string to find.  
- `confidence`: Minimum matching score (default `0.4`).  

**Returns:**  
- Tuple of `(best_line_index, score)` or `(None, score)` if confidence threshold not met.

---

### `extract_int_from_string(s) -> str | None`
Extracts all digits from a string while preserving order.  

**Parameters:**  
- `s`: Input string.  

**Returns:**  
- String containing only digits, or `None` if no digits found.

---

### `format_fields_smartid(text, confidence=0.4) -> tuple[dict[str, str], str]`
Extracts key fields (`Surname`, `Names`, `Identity Number`) from cleaned OCR text. Uses line search and numeric fallback.  

**Parameters:**  
- `text`: Cleaned OCR text.  
- `confidence`: Line matching threshold (default `0.4`).  

**Returns:**  
- `fields`: Dictionary with extracted values (`None` if missing).  
- `method`: `"field_search"`, `"numeric"`, or `"unsuccessful"`.

---

### `ocr_to_dict_smartid(text, allowed_chars=None, filler_char="", confidence=0.5) -> tuple[dict[str, str], str]`
Wrapper combining OCR cleaning and field extraction.  

**Parameters:**  
- `text`: Raw OCR text.  
- `allowed_chars`: Optional set of valid characters.  
- `filler_char`: Replacement for invalid characters.  
- `confidence`: Line search threshold (default `0.5`).  

**Returns:**  
- `fields`: Dictionary with `Surname`, `Names`, `Identity Number`.  
- `method`: Extraction method used.