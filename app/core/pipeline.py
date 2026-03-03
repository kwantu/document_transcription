from app.core.utils import * # imports key packages, too (cv2, YOLO, ...)
from dataclasses import dataclass, field, asdict
import json
import torch
from torchvision import models, transforms
import torch.nn as nn
import pytesseract
from typing import Any


@dataclass
class GeometryConfig: # REQUIRED dataclass
    id_class: int
    metadata_target_height: int = 400
    correction_angle: float = 0.0

@dataclass
class PreprocessConfig:
    denoise_type: str = "bilateral"     # "bilateral" or "gaussian"
    k_denoise: int = 3                  # if gaussian
    bilateral_d: int = 5                # if bilateral
    bilateral_sigma_color: int = 20     # ^
    bilateral_sigma_space: int = 20     # ^^
    thresh_block: int = 13
    thresh_c: int = 15                  # threshold bias. higher => darker result
    morph_kernel: tuple[int, int] = (2, 1)
    ocr_psm: int = 6

@dataclass
class PostprocessConfig:
    allowed_chars: set = field(default_factory=lambda: set(string.ascii_letters + string.digits + "- :"))
    filler_char: str = ""
    field_list: list[str] = field(default_factory=lambda: ["Surname", "Names", "Identity Number"])
    confidence: float = 0.4

@dataclass
class ImageParams: # Helps us access structured data in future functions
    doc_class_info: dict # efficientnet data
    geometry_config: GeometryConfig
    preprocess_config: PreprocessConfig
    postprocess_config: PostprocessConfig


# ID post-processing functions from utils.py
ID_HANDLERS = {
    0: ocr_to_dict_smartid,
    1: ocr_to_dict_idbook
}

# Load our 3 model weights into memory now
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path(__file__).resolve().parent / "pipeline_models"

IDBOOK = YOLO(MODEL_PATH / "idbook_yolo.pt")
SMARTID = YOLO(MODEL_PATH / "smartid_yolo.pt")
IDBOOK.to(DEVICE)
SMARTID.to(DEVICE)

DOC_DETECTION = models.efficientnet_b0(weights="IMAGENET1K_V1")
DOC_DETECTION.classifier[1] = nn.Linear(
    DOC_DETECTION.classifier[1].in_features, 2
)

checkpoint = torch.load(MODEL_PATH / "effnet_classifier.pt", map_location=DEVICE, weights_only=False)
DOC_DETECTION.load_state_dict(checkpoint["model_state"])

DOC_DETECTION.to(DEVICE)
DOC_DETECTION.eval()

_doc_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# --- ID Type Recognition & Param Loader ---
def load_params(img: np.ndarray) -> ImageParams:
    """
    Pass the image through the first stage of the process, allowing us to parametrise the subsequent
    'img_2_json' function.
    :param img: Sample image.
    :return: Pipeline parameters (geom_params, prep_params, post_params)
    """

    # Send through document detection model:
    # 1. Preprocess
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    x = _doc_transform(img_rgb).unsqueeze(0).contiguous().to(DEVICE)

    # 2. Make an inference
    doc_class = -1 # set to base value
    conf = -1.0
    with torch.no_grad():
        logits: torch.Tensor = DOC_DETECTION(x)
        doc_class = logits.argmax(dim=1).item()

        probs = torch.softmax(logits, dim=1)
        conf = probs[0, doc_class].item()
    info = {
        "doc_class": doc_class,
        "confidence": conf
    }

    # structure params
    if doc_class == 0: # SMARTID
        geometry_params = GeometryConfig(
            id_class=doc_class,
            metadata_target_height=440,
            correction_angle=10
        )
        preprocess_params = PreprocessConfig(
            thresh_c=4
        )
        return ImageParams(info, geometry_params, preprocess_params, PostprocessConfig())
    elif doc_class == 1: # IDBOOK
        geometry_params = GeometryConfig(
            id_class=doc_class,
            metadata_target_height=380,
            correction_angle=-100
        )
        preprocess_params = PreprocessConfig(
            thresh_c=20
        )
        return ImageParams(info, geometry_params, preprocess_params, PostprocessConfig())
    else:
        raise ValueError(f"Invalid document class found, class={doc_class}")


# --- Image normalisation layer ---
def load_input(img_path: str | Path) -> list[tuple[str, np.ndarray]]:
    """
    Loads input variables for an image path. Accepted image formats are: jpeg/jpg, png, pdf.
    PDFs return multiple images if there is more than one page.
    :param img_path: Path to image
    :return: List of pairs of image id, or name, and the pixel image in NumPy array form (img_id, img).
    """

    path = Path(img_path)
    ext = path.suffix.lower()

    if ext in {".jpg", ".jpeg", ".png"}:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return [(path.stem, img)]

    if ext == ".pdf":
        img = pdf2img(img_path)
        return [(path.stem, img)]

    raise ValueError("Unsupported file type")


# --- Raw image -> JSON ---
def _legacy_img_2_json(
        yolo_model: YOLO,
        img: np.ndarray,
        img_id: str,
        dest_path: str | Path | None = None, # one folder for results
        geom_params: GeometryConfig | None = None,
        prep_params: PreprocessConfig | None = None,
        post_params: PostprocessConfig | None = None,
        save_process: bool = True, # False if we just want formatted output (final deployment for example)
        save_params: bool = False # Not worried about this really anymore
) -> dict[str, Any]:
    """
    The end-to-end pipeline that takes the YOLO model associated with geom_params.id_class, and an image path, to:
        1. Infer labels (YOLO).
        2. Reorient, rescale & deskew.
        3. Grayscale, denoise, binarise & thicken text.
        4. Extract OCR from Tesseract.
        5. Clean & format raw OCR string.
        6. Store process images, params, and outputs in DEST_PATH.
    :param yolo_model: YOLO objet recognition model that extracts metadata & photo.
    :param img: Image data.
    :param img_id: Image id, for consistent naming.
    :param dest_path: Directory of output folder, named after the image. If the image is called
        'image001.jpeg' for example, the output will be stored in DEST_PATH/img001/ (from img_id).
        If no DEST_PATH specified, output is placed next to the raw image.
        If we are provided with a pdf, we are going to load each page as its own seperate image.
    :param geom_params: GeometryConfig dataclass.
    :param prep_params: PreprocessConfig dataclass.
    :param post_params: PreprocessConfig dataclass.
    :param save_process: Bool, saves preprocessing figures to see in the output directory.
    :param save_params: Bool, saves parameters for each image (not useful anymore really).
    :return: Field dictionary, {"Names": JOHN, "Surname": SMITH, "Identity Number": 0123456789012} for example.
    """

    # Guard against empty image files
    if img is None or img.size == 0:
        raise RuntimeError(f"Empty image from {img_id}")

    # 0. Directories and parameters
    dest_root = Path(dest_path) if dest_path else Path(".")
    dest_root.mkdir(parents=True, exist_ok=True)

    # require geom_params since we are working with multiple ID types.
    if geom_params is None:
        raise ValueError("geom_params must be provided")
    # otherwise, continue with basic preprocess params.
    prep_params = prep_params or PreprocessConfig()
    post_params = post_params or PostprocessConfig()

    # 1. YOLO (receives PIXELS, np.ndarray, not a path to an img/document)
    r = yolo_model(img)[0]
    fig_original = display(img, show=False)

    # 2. Reorient + rescale
    reoriented_img, (delta, cv2_rotation_ang), fig_reoriented = reorient_img(
        result=r,
        correction_angle=geom_params.correction_angle,
        return_fig=save_process,
        show=False
    )

    # BEFORE ANYTHING, try to read barcode from reoriented image
    barcode_num = None
    if geom_params.id_class == 1: # IDBOOK
        barcode_num = barcode_id_num(reoriented_img)

    metadata, photo = None, None
    for box in r.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == 2:
            metadata = img[y1:y2, x1:x2]
        elif cls == 1:
            photo = img[y1:y2, x1:x2]

    if metadata is None:
        raise RuntimeError("Metadata not detected")

    # Also need to REORIENT METADATA + PHOTO
    if cv2_rotation_ang is not None:
        metadata = cv2.rotate(metadata, cv2_rotation_ang)
        photo = cv2.rotate(photo, cv2_rotation_ang)

    scaled, sf = rescale(metadata, target_height=geom_params.metadata_target_height)
    h, w = scaled.shape[:2]  # for param saving

    # 3. Deskew
    rotated, _, fig_rotated = deskew_img(
        scaled,
        return_fig=save_process,
        show=False
    )

    # 4. Preprocessing
    # -- grayscale
    # -- denoise
    # -- adaptive threshold
    # -- thicken
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    denoised = gray

    if prep_params.denoise_type == "gaussian":
        denoised = cv2.GaussianBlur(
            gray,
            (prep_params.k_denoise, prep_params.k_denoise),
            0
        )

    elif prep_params.denoise_type == "bilateral":
        denoised = cv2.bilateralFilter(
            gray,
            d=prep_params.bilateral_d,
            sigmaColor=prep_params.bilateral_sigma_color,
            sigmaSpace=prep_params.bilateral_sigma_space,
        )

    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        prep_params.thresh_block,
        prep_params.thresh_c
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, prep_params.morph_kernel
    )
    thick = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 5. OCR
    raw_ocr = pytesseract.image_to_string(
        thick, config=f"--psm {prep_params.ocr_psm}"
    )

    # 6. Output dir
    base = img_id
    out_dir = dest_root / base
    i = 1
    while out_dir.exists():
        out_dir = dest_root / f"{base}_{i}"
        i += 1
    out_dir.mkdir(parents=True)

    # 7. Save outputs
    if photo is not None:
        cv2.imwrite(str(out_dir / "photo.png"), photo)

    if save_process:
        (out_dir / "raw_ocr.txt").write_text(raw_ocr, encoding="utf-8")

        fig_ex, axs = plt.subplots(1, 4, figsize=(12, 4))

        axs[0].imshow(cv2.cvtColor(metadata, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Metadata")

        axs[1].imshow(denoised, cmap="gray")
        axs[1].set_title("Denoised")

        axs[2].imshow(thick, cmap="gray")
        axs[2].set_title("Binary")

        axs[3].text(0.01, 0.99, raw_ocr, va="top", wrap=True)
        axs[3].set_title("OCR")

        for ax in axs:
            ax.axis("off")

        fig_ex.tight_layout()

        fig_original.savefig(out_dir / "input.png")
        fig_reoriented.savefig(out_dir / "reoriented.png")
        fig_rotated.savefig(out_dir / "rotated.png")
        fig_ex.savefig(out_dir / "preprocessing.png")

        plt.close("all")

    # 8. Clean & format dictionary to save as JSON
    ocr_to_dict = ID_HANDLERS.get(geom_params.id_class)
    output_dict, extr_method = ocr_to_dict(
        raw_ocr,
        post_params.allowed_chars,
        post_params.filler_char,
        post_params.confidence
    )

    # barcode_num handling
    if barcode_num:
        output_dict["Identity Number"] = barcode_num
        extr_method = "barcode_decoding"

    # 9. Save params
    if save_params:
        params: dict[str, Any] = {
            "geometry": asdict(geom_params),
            "preprocess": asdict(prep_params),
            "postprocess": asdict(post_params),
            "img_id": img_id,
            "classes": r.names,
            "scale_factor": sf,
            "reorientation_delta": delta,
            "reorientation_cv2": cv2_rotation_ang,
            "rescaled_metadata_shape": {"w": int(w), "h": int(h)},
            "id_no_extraction_method": extr_method,
            "found_photo": photo is not None
        }

        # Since json doesn't like sets, we have to do this:
        params["postprocess"]["allowed_chars"] = list(params["postprocess"]["allowed_chars"])

        with open(out_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)

    with open(out_dir / "output.json", "w") as f:
        json.dump(output_dict, f, indent=2)

    return output_dict

def img_2_json_v2(
        yolo_model: YOLO,
        img: np.ndarray,
        img_id: str,
        dest_path: str | Path | None = None, # one folder for results
        effnet_dict: dict | None = None, # EfficientNet output info
        geom_params: GeometryConfig | None = None,
        prep_params: PreprocessConfig | None = None,
        post_params: PostprocessConfig | None = None,
        save_process: bool = True, # False if we just want formatted output (final deployment for example)
) -> dict[str, str]:
    """
    Updated version of img_2_json. Tries 4 preprocessing variants & stages each one against eachother using
    pytesseract confidence scores.
    """

    if img is None or img.size == 0:
        raise RuntimeError(f"Empty image from {img_id}")

    if geom_params is None:
        raise ValueError("geom_params must be provided")

    prep_params = prep_params or PreprocessConfig()
    post_params = post_params or PostprocessConfig()

    dest_root = Path(dest_path) if dest_path else Path(".")
    dest_root.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1. YOLO DETECTION
    # -------------------------------------------------
    r = yolo_model(img)[0]
    fig_original = display(img, show=False)

    reoriented_img, (delta, cv2_rotation_ang), fig_reoriented = reorient_img(
        result=r,
        correction_angle=geom_params.correction_angle,
        return_fig=save_process,
        show=False
    )

    barcode_num = None
    if geom_params.id_class == 1:
        barcode_num = barcode_id_num(reoriented_img)

    metadata = None
    photo = None

    best_meta_conf = -1.0
    best_photo_conf = -1.0

    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 2 and conf > best_meta_conf:
            best_meta_conf = conf
            metadata = img[y1:y2, x1:x2]
        elif cls == 1 and conf > best_photo_conf:
            best_photo_conf = conf
            photo = img[y1:y2, x1:x2]

    if metadata is None:
        raise RuntimeError("Metadata not detected")

    if cv2_rotation_ang is not None:
        metadata = cv2.rotate(metadata, cv2_rotation_ang)
        if photo is not None:
            photo = cv2.rotate(photo, cv2_rotation_ang)

    # -------------------------------------------------
    # 2. RESCALE
    # -------------------------------------------------
    scaled, sf = rescale(metadata, target_height=geom_params.metadata_target_height)

    # -------------------------------------------------
    # 3. DESKEW
    # -------------------------------------------------
    rotated, _, fig_rotated = deskew_img(
        scaled,
        return_fig=save_process,
        show=False
    )

    # -------------------------------------------------
    # 4. PREPROCESS VARIANTS
    #   - Grayscale
    #   - CLAHE (tiled histogram equalisation)
    #   - CLAHE + OTSU
    #   - Bilateral Filter + Otsu
    # -------------------------------------------------
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    # CLAHE improves low contrast IDs
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Varients 1 and 2: Raw Grayscale and CLAHE Only
    variants = {"gray": gray, "clahe": clahe_img}

    # Variant 3: CLAHE + Otsu
    _, otsu = cv2.threshold(
        clahe_img, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    variants["clahe_otsu"] = otsu

    # Variant 4: Bilateral + Otsu
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    _, bil_otsu = cv2.threshold(
        bilateral, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    variants["bilateral_otsu"] = bil_otsu

    # -------------------------------------------------
    # 5. OCR WITH CONFIDENCE SCORING
    # -------------------------------------------------
    best_text = ""
    best_conf = -1
    best_variant = None

    for name, variant_img in variants.items():

        data = pytesseract.image_to_data(
            variant_img,
            config=f"--oem 1 --psm {prep_params.ocr_psm}",
            output_type=pytesseract.Output.DICT
        )

        words = []
        confs = []

        for txt, conf in zip(data["text"], data["conf"]):
            if txt.strip() != "" and conf != "-1":
                words.append(txt)
                confs.append(int(conf))

        if len(confs) == 0:
            continue

        # get raw text SEPARATELY
        raw_text = pytesseract.image_to_string(
            variant_img,
            config=f"--oem 1 --psm {prep_params.ocr_psm}"
        )

        mean_conf = float(np.mean(confs))

        if mean_conf > best_conf:
            best_conf = mean_conf
            best_variant = name
            best_text = raw_text

    # -------------------------------------------------
    # 6. OUTPUT
    # -------------------------------------------------
    base = img_id
    out_dir = dest_root / base
    i = 1
    while out_dir.exists():
        out_dir = dest_root / f"{base}_{i}"
        i += 1
    out_dir.mkdir(parents=True)

    if photo is not None:
        cv2.imwrite(str(out_dir / "photo.png"), photo)

    # -------------------------------------------------
    # 7. POSTPROCESS -> STRUCTURED OUTPUT & SAVE IMAGES
    # -------------------------------------------------
    ocr_to_dict = ID_HANDLERS.get(geom_params.id_class)

    output_dict, extr_method = ocr_to_dict(
        best_text,
        post_params.allowed_chars,
        post_params.filler_char,
        post_params.confidence
    )

    if barcode_num and validate_id(barcode_num):
        output_dict["Identity Number"] = barcode_num
        extr_method = "barcode_decoding"

    with open(out_dir / "output.json", "w") as f:
        json.dump(output_dict, f, indent=2)

    if save_process:

        # assert folder structure
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "raw_ocr.txt").write_text(best_text, encoding="utf-8")

        # Structure additional info
        h, w = scaled.shape[:2]
        info = {
            "id_class": geom_params.id_class,
            "id_class_confidence": effnet_dict["confidence"],
            "scale_factor": sf,
            "rescaled_metadata_shape": {"w": int(w), "h": int(h)},
            "id_no_extraction_method": extr_method,
            "found_photo": photo is not None,
            "metadata_confidence": best_meta_conf,
            "photo_confidence": best_photo_conf,
            "best_ocr_conf": best_conf,
            "best_ocr_variant": best_variant,
        }

        with open(debug_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        fig_ex, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(cv2.cvtColor(metadata, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Metadata")

        if best_variant is not None:
            axs[1].imshow(variants[best_variant], cmap="gray")
            axs[1].set_title(f"Best: {best_variant}")
        else: # in the unlikely event
            axs[1].imshow(np.zeros_like(gray), cmap="gray")
            axs[1].set_title("Best: None")

        axs[2].text(0.01, 0.99, best_text, va="top", wrap=True)
        axs[2].set_title(f"OCR (conf={best_conf:.1f})")

        for ax in axs:
            ax.axis("off")

        fig_ex.tight_layout()

        fig_original.savefig(debug_dir / "input.png")
        fig_reoriented.savefig(debug_dir / "reoriented.png")
        fig_rotated.savefig(debug_dir / "rotated.png")
        fig_ex.savefig(debug_dir / "preprocessing.png")

        plt.close("all")

    return output_dict


# --- Full Pipeline Function ---
def full_pipeline(
        input_path: str | Path,
        dest_path: str | Path | None = None,
        save_process: bool = False,
) -> tuple[dict[str, str], bool]:
    """
    The full end-to-end pipeline, including doc recognition (that we have not yet made).
    If we have a pdf containing more than one page, the results dictionary will have multiple items.
    :param input_path: Path to the image
    :param dest_path: Path to the output directory (None => default destination handling)
    :param save_process: bool, optional: saves verbose preprocessing images w/ ocr results
    :return: Formatted fields dictionary for results of each img_id
    """
    input_path = str(input_path)

    results = {}
    img_id, img = load_input(input_path)[0] # CURRENTLY only one pair output (sort of redundant)
    # in the future, we are to loop like `for img_id, img in list[tuple[str, np.ndarray]] load_input(input_path):`

    # Stage 1: Detect doc class & load params
    params: ImageParams = load_params(img)
    geom_params = params.geometry_config
    prep_params = params.preprocess_config
    post_params = params.postprocess_config

    # Stage 2: Select YOLO model based on doc class
    yolo_model = None
    if geom_params.id_class == 0:
        yolo_model = SMARTID
    elif geom_params.id_class == 1:
        yolo_model = IDBOOK

    # Stage 3: Send the image through the pipeline
    output: dict[str, str] = img_2_json_v2(
        img=img,
        img_id=img_id,
        dest_path=dest_path,
        save_process=save_process,
        yolo_model=yolo_model,
        effnet_dict=params.doc_class_info,
        geom_params=geom_params,
        prep_params=prep_params,
        post_params=post_params,
    )

    is_valid = validate_id(output["Identity Number"])

    return output, is_valid