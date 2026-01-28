from dataclasses import dataclass, field, asdict
import json
import pytesseract
from app.core.utils import * # imports key packages, too (cv2, YOLO, ...)
from typing import Any

@dataclass
class GeometryConfig: # will require this down the line for each ID type.
    id_class: int                       # smartid = 0, idbook = 1
    metadata_target_height: int = 400   # ~440 for smartid, 350 for idbook
    correction_angle: float = 0.0       # 10 for smartid, -100 for idbook


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
    allowed_chars: set = field(
        default_factory=lambda: set(string.ascii_letters + string.digits + "- :")
    )
    filler_char: str = ""
    field_list: list[str] = field(
        default_factory=lambda: ["Surname", "Names", "Identity Number"]
    )
    confidence: float = 0.4


# ID post-processing functions from utils.py
ID_HANDLERS = {
    0: ocr_to_dict_smartid,
    1: ocr_to_dict_idbook
}


# --- ID Type Recognition & Param Loader ---
def load_params(img_path: Path | str) -> tuple[GeometryConfig, PreprocessConfig, PostprocessConfig]:
    """
    Pass the image through the first stage of the process, allowing us to parametrise the subsequent
    'img_2_json' function.
    :param img_path: str | Path object to the sample.
    :return: Pipeline parameters (geom_params, prep_params, post_params)
    """
    img = cv2.imread(str(img_path)) # load image

    # send through doc recognition stage (not yet made)
    doc_class = 0

    # structure params
    if doc_class == 0: # SMARTID
        return GeometryConfig(
            id_class=doc_class,
            metadata_target_height=440,
            correction_angle=10
        ), PreprocessConfig(
            thresh_c=4
        ), PostprocessConfig() # defaults, for now on postprocessing
    elif doc_class == 1: # IDBOOK
        return GeometryConfig(
            id_class=doc_class,
            metadata_target_height=380,
            correction_angle=-100
        ), PreprocessConfig(
            thresh_c=20
        ), PostprocessConfig()
    else:
        raise ValueError(f"Invalid document class found, class={doc_class}")


# --- Raw image -> JSON ---
def img_2_json(
        yolo_model: YOLO,
        img_path: str | Path,
        dest_path: str | Path | None = None,  # one folder for results
        geom_params: GeometryConfig | None = None,
        prep_params: PreprocessConfig | None = None,
        post_params: PostprocessConfig | None = None,
        save_process: bool = True  # want False if we just want formatted output
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
    :param img_path: str | Path object to the sample.
    :param dest_path: Directory of output folder, named after the image. If the image at 'img_path' is called
        'image001.jpeg' for example, the output will be stored in DEST_PATH/img001/.
        If no DEST_PATH specified, output is placed next to the raw image.
    :param geom_params: GeometryConfig dataclass.
    :param prep_params: PreprocessConfig dataclass.
    :param post_params: PreprocessConfig dataclass.
    :param save_process: Bool, saves preprocessing figures to see in the output directory.
    :return: Field dictionary, {"Names": JOHN, "Surname": SMITH, "Identity Number": 0123456789012} for example.
    """
    # 0. Directories and parameters
    img_path = Path(img_path)
    dest_root = Path(dest_path) if dest_path else img_path.parent
    dest_root.mkdir(parents=True, exist_ok=True)

    # require geom_params since we are working with multiple ID types.
    # otherwise, continue with basic preprocess params.
    prep_params = prep_params or PreprocessConfig()
    post_params = post_params or PostprocessConfig()

    # 1. YOLO
    r = yolo_model(img_path)[0]
    img = r.orig_img.copy()
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
    base = img_path.stem
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
    params: dict[str, Any] = {
        "geometry": asdict(geom_params),
        "preprocess": asdict(prep_params),
        "postprocess": asdict(post_params),
        "img_path": str(img_path),
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

    if save_process:
        with open(out_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)

    with open(out_dir / "output.json", "w") as f:
        json.dump(output_dict, f, indent=2)

    return output_dict


# --- Full Pipeline Function ---
def full_pipeline(
        img_path: str,
        dest_path: str | Path | None = None,
        save_process: bool = False,
) -> dict[str, Any]:
    """
    The full end-to-end pipeline, including doc recognition (that we have not yet made)
    :param img_path: Path to the image
    :param dest_path: Path to the output directory
    :param save_process: bool, optional: saves verbose preprocessing images w/ ocr results
    :return: dict[str, Any] Formatted fields dictionary
    """

    # Stage 1: Detect doc class & load params
    geom_params, prep_params, post_params = load_params(img_path)

    # Stage 2: Select YOLO model based on doc class
    doc_class: int = geom_params.id_class
    yolo_model = None

    if doc_class == 0: # SMARTID
        yolo_model = YOLO("path/to/smartid_weights.pt")
    elif doc_class == 1: # IDBOOK
        yolo_model = YOLO("path/to/idbook_weights.pt")

    # Stage 3: Send the image through the pipeline
    output = img_2_json(
        img_path=img_path,
        dest_path=dest_path,
        save_process=save_process,
        yolo_model=yolo_model,
        geom_params=geom_params,
        prep_params=prep_params,
        post_params=post_params,
    )

    return output