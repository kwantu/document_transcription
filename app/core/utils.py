from difflib import SequenceMatcher
import string
from typing import Optional
from matplotlib.figure import Figure
from numpy import ndarray
from ultralytics import YOLO
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pyzbar.pyzbar import decode
from datetime import datetime


# --- IMAGE VIEW FUNCTION ---
def display(
        img: np.ndarray | str | Path,
        dpi: float = 80.0,
        cmap: str | None = None,
        show: bool = True
) -> plt.Figure:
    """
    Display an image using matplotlib.
    :param img: Either np.ndarray (NumPy image tensor), or path-like object.
    :param dpi: Dots per inch, defaults to 80.0.
    :param cmap: Matplotlib cmap, defaults to None.
    :param show: Shows figure. Setting to False just returns the image as a fig.
    :return: Returns matplotlib figure of the image.
    """
    if isinstance(img, str) or isinstance(img, Path):  # we have a path => read the image
        img = cv2.imread(img)
    elif isinstance(img, np.ndarray):  # if we have an image
        img = img.copy()  # don't want to mess anything up
    else:
        raise TypeError("Image must be an ndarray, or path-like object (str | Path)")

    height, width = img.shape[:2]

    # Find the size of the figure for the image to retain its shape
    figure_size = width / dpi, height / dpi

    # plot the image
    fig = plt.figure(figsize=figure_size)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap=cmap) # plot in RGB
    if show:
        plt.show()
    return fig


# --- PRODUCTION READY REORIENTATION ---
def reorient_img(
        result: YOLO,
        correction_angle: float = 0.0,
        show: bool = False,
        return_fig: bool = False
) -> tuple[np.ndarray, tuple[float, Optional[int]], Optional[Figure]]:
    """
    Algorithm that reorients an image by any angle in {90, 180, 270} using the position of metadata/photo, coupled
    with what we are calling the 'correction angle'. This correction angle is the angle with +x that a perfectly
    oriented example should have. If the angle between the centres of the labels and our 'correction angle' is near
    zero, then don't adjust, since the image is already oriented correctly.
    :param result: YOLO result of an image.
    :param correction_angle: Ideal angle to +x of the line between the 'metadata' and 'person' labels.
    :param show: Bool, whether you want to see the lines by which the algorithm determines rotation
    :param return_fig: Bool, put True if you want to return the figure (img is ALWAYS returned, this variable
    determines if you want to return a fig of the process).
    :return: Rotated image, delta (offset that determines reorientation), reorientation figure (if return_fig).
    """
    img = result.orig_img
    centres = {}
    for box in result.boxes:
        cls = str(int(box.cls))
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centres.setdefault(cls, []).append((cx, cy))

    cx1, cy1 = centres.get("2")[0]  # set as lists above
    cx2, cy2 = centres.get("1")[0]  # set as lists above
    dx, dy = cx2 - cx1, cy2 - cy1

    # angle of line with +x
    # remember, we are working with inverted angle intuition due to array indexxing
    theta = (-np.degrees(np.arctan2(dy, dx)) + 360) % 360
    delta = (theta - correction_angle + 360) % 360  # wrap to [0, 360) also

    # logical reorientation
    rotation: Optional[int] = None
    if 45 <= delta < 135:       # indicator points DOWN
        rotation = cv2.ROTATE_90_CLOCKWISE
    elif 135 <= delta < 225:    # indicator points left (uncommon)
        rotation = cv2.ROTATE_180
    elif 225 <= delta < 315:    # indicator points UP
        rotation = cv2.ROTATE_90_COUNTERCLOCKWISE

    if rotation is not None:
        img = cv2.rotate(img, rotation)

    if show or return_fig:

        fig = plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))

        # arrows
        comparison_axis = np.hypot(dx, dy)
        plt.arrow(
            cx1, cy1,
            comparison_axis * np.cos(np.deg2rad(correction_angle)),
            -comparison_axis * np.sin(np.deg2rad(correction_angle)),
            width=2, color="green", label="Corrected +x"
        )

        plt.arrow(
            cx1, cy1,
            dx, dy,
            width=3, length_includes_head=True,
            color="orange", label="Indicator"
        )

        plt.axis("off")
        plt.title(f"Visualising rotation angle")
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        plt.tight_layout()

        if show:
            plt.show()

        if return_fig:
            return img, (delta, rotation), fig

    return img, (delta, rotation), None


# --- SMALL ANGLE ROTATION --- (using contours)
def deskew_img(
        img: np.ndarray,
        show: bool = False,
        return_fig: bool = False
) -> tuple[ndarray, float, Figure] | tuple[ndarray, float, None]:
    """
    Performs a small angle deskew using the contours of detected lines of text.
    :param img: NumPy image tensor of the image you want to deskew.
    :param show: Bool, whether you want to see the lines by which the algorithm determines rotation
    :param return_fig: Bool, put True if you want to return the figure (img is ALWAYS returned, this variable
    determines if you want to return a fig of the process).
    :return: Deskewed image, deskew angle, process fig (if return_fig)
    """
    (h, w) = img.shape[:2]

    # --- gray + otsu ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    # --- noise removal ---
    # kill tiny specks + thin artefacts, preserve actual strokes
    nk = max(3, min(h, w) // 300) # pretty heuristic right now
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (nk, nk))
    opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, noise_kernel, iterations=1)

    # --- horizontal dilation to joing characters into lines ---
    # we are in, to an extent, a "small angle" regime, but we are trying to calculate the small angle
    # kernel width should be tuned to text size. we try 30-60 for now and leave fine-tuning for later
    kw = max(15, w // 40)
    kh = max(3, h // 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    dilated = cv2.dilate(opened, kernel, iterations=2)

    # --- ad PADDING to our dilated image ---
    # this is so that our minAreaRect is not obstructed by image boarders
    pad = min(h, w) // 10
    dilated = cv2.copyMakeBorder(
        dilated,
        pad, pad, pad, pad,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0) # black pixels
    )

    # --- find contours + compute angle ---
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_boxes_img = img.copy()
    # need to add the same padding to our contour boxes image, too
    contour_boxes_img = cv2.copyMakeBorder(
        contour_boxes_img,
        pad, pad, pad, pad,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255) # white pixels added temporarily to original image
    )
    min_area_thresh = (h * w) * 0.005  # ignore box if < 0.5% of total image area (not counting padding)

    angles = []
    for c in contours:
        area = cv2.contourArea(c)

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(np.int32)

        if area < min_area_thresh:
            # draw below-threshold contours in red
            # remember, currently in BGR. we convert when plotting
            cv2.drawContours(contour_boxes_img, [box], 0, (0, 0, 255), 2)
            continue  # skip adding angle
        else:
            # normal contours in green
            cv2.drawContours(contour_boxes_img, [box], 0, (0, 255, 0), 2)

        ang = rect[-1]
        if ang < -45:
            ang += 90
        elif ang > 45:
            ang -= 90
        angles.append(ang)

        # draw box for debug overlay
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(contour_boxes_img, [box], 0, (0, 255, 0), 2)

    # --- a robust aggregation and rotation ---
    # print(f"Angles: {angles}")
    a = np.array(angles)
    theta = float(np.mean(a)) if len(a) else 0.0

    rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), theta, 1.0)
    rotated = cv2.warpAffine(
        img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    if show or return_fig:
        fig, axs = plt.subplots(2, 3, figsize=(12, 7))
        axs = axs.ravel()

        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original")

        axs[1].imshow(otsu, cmap='gray')
        axs[1].set_title("Otsu (text=white)")

        axs[2].imshow(opened, cmap='gray')
        axs[2].set_title("Noise-removed (open)")

        axs[3].imshow(dilated, cmap='gray')
        axs[3].set_title("Dilated w/padding")

        axs[4].imshow(cv2.cvtColor(contour_boxes_img, cv2.COLOR_BGR2RGB))
        axs[4].set_title(f"Detected line boxes (n={len(angles)})")

        axs[5].imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

        if theta > 1e-5:
            axs[5].set_title(f"Rotated ANTICLOCKWISE by {theta:.2f}°")
        elif theta < -1e-5:
            axs[5].set_title(f"Rotated CLOCKWISE by {-theta:.2f}°")
        else:
            axs[5].set_title("No rotation required")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()

        if show:
            plt.show()

        if return_fig:
            return rotated, theta, fig

    return rotated, theta, None


# --- RESCALE TO TARGET HEIGHT ---
def rescale(
        img,
        target_height: int = 400, # should have ~420 for smartid, ??? for idbook
        max_scale_factor: float | None = 4.0,
        interpolation=cv2.INTER_LANCZOS4 # can try INTER-CUBIC
) -> tuple[np.ndarray, float]:
    """
    Function to rescale the image to a target height. Scale factors > 3 are flagged as a warning, since image quality
    is likely to be low.
    :param img: NumPy image tensor
    :param target_height: How tall must the image be for text height to land in the 20-30px range.
    :param max_scale_factor: When image quality is too poor, this prints a warning.
    :param interpolation: OpenCV interpolation type.
    :return: Rescaled image and scale factor.
    """
    # infer scale factor
    sf = target_height / img.shape[0]
    if max_scale_factor is not None and sf > max_scale_factor:
        print(f"WARNING! High scale factor (sf = {sf:.2f})")

    # scale
    rescaled = cv2.resize(
        img, None,
        fx=sf, fy=sf,
        interpolation=interpolation
    )
    return rescaled, sf


# --- OCR Cleaning and Formatting ---
def clean_raw_ocr_output(
        text: str,
        allowed_chars: set | None = None,
        filler_char: str = ""
) -> str:
    """
    Cleans OCR output by:
        1. Removing empty lines and stripping whitespace.
        2. Filtering through valid characters, and replacing characters not in allowed_chars with filler_char.
        3. Reassembling the cleaned text and returning as a string
    :param text: Raw OCR output (string)
    :param allowed_chars: Set of valid characters we want to search for. Typically upper/lower ASCII, numbers,
    space, hyphon, colon.
    :param filler_char: What we replace noisy characters with
    :return: Cleaned OCR output (string)
    """
    # 0. sort out allowed_chars set
    if allowed_chars is None:
        allowed_chars = set(string.ascii_letters + string.digits + "- :")

    # 1. remove empty lines & strip font/back whitespace
    lines = [l.strip() for l in text.splitlines() if l]

    # 2. filter through allowed characters against 'allowed_chars'
    cleaned_lines = [
        "".join(c if c in allowed_chars else filler_char for c in line)
        for line in lines
    ]

    # 3. reassemble text
    return "\n".join(cleaned_lines)

# --- TOKEN-BASED LINE SEARCH ---
def search_for_line(
        text: str,
        line: str,
        confidence: float = 0.4
) -> tuple[int, float] | tuple[None, float]:
    """
    Searches through text for the row that best matches a string.
    :param text: (Ideally) Cleaned OCR text
    :param line: Line you want to search for. "Identity Number:" for example.
    :param confidence: Score based rejection of the best idx.
    :return: Index of best match, score of best match.
    """
    lines = text.splitlines()

    best_idx = -1
    best_score = 0.0

    for i, l in enumerate(lines):
        score = SequenceMatcher(None, l, line).ratio()
        if score > best_score:
            best_score = score
            best_idx = i

    if best_score < confidence:
        return None, best_score

    return best_idx, best_score

# --- BARCODE DETECTION ---
def barcode_id_num(img: np.ndarray | str | Path) -> str | None:
    """
    Use pyzbar to detect and decode barcode, extracting ID number.
    :param img: Image containing barcode.
    :return: ID number if decoded, None otherwise.
    """
    if isinstance(img, str) or isinstance(img, Path):
        img = cv2.imread(str(img))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    barcodes = decode(gray)
    id_num = None

    if barcodes is not None:
        for barcode in barcodes:
            id_num = barcode.data.decode("utf-8")
            if id_num:
                break

    return id_num


# Helpers for field formatting
def extract_int_from_string(s: str) -> str | None:
    """
    Return all digit characters in 's', preserving order.
    Ignore whitespace, noise, and symbols.
    :param s: String to extract integers from
    :return: String containing all digit characters in 's'
    """
    digits = [c for c in s if c.isdigit()]
    return "".join(digits) if digits else None

def numeric_line(lines: list[str]) -> tuple[int, str]:
    """
    Finding the index and line of the most numeric line given a list[str] of lines.
    :param lines: List of string type objects (contextually, our OCR text.splitlines())
    :return: index and most numeric line
    """
    scores = []

    for line in lines:
        chars = [c for c in line if not c.isspace()]
        if not chars:
            scores.append(0.0)
            continue

        numeric = sum(c.isdigit() for c in chars)
        scores.append(numeric / len(chars))

    idx = max(range(len(scores)), key=scores.__getitem__)
    nums = extract_int_from_string(lines[idx])

    return idx, nums

# --- BESPOKE FIELD FORMATTING FUNCTION ---
def format_fields_smartid(
        text: str,
        confidence: float = 0.4
) -> tuple[dict[str, str], str]:
    """
    Bespoke hard coded smartid formatting function using rule-based filtering.
    Aim to store:
        - Surname
        - Names
        - Identity Number
    Notes:
        1. We are being strict on the ID number containing 13 digits.
        2. Current stage of the pipeline is not cross-referencing ID number with other values like DoB, so we only
            look for the three above fields at the moment. However, there is room for easy scalability.
    :param text: Clean OCR text
    :param confidence: Score based rejection of the best idx to go into the line-search
    :return: Dictionary of best fields
    """

    field_list = ["Surname", "Names", "Identity Number"]
    fields: dict[str, str | None] = {k: None for k in field_list}
    lines = text.splitlines()

    # Field search
    for f in field_list:
        idx, score = search_for_line(text, f"{f}:", confidence=confidence)

        if idx is None or idx + 1 >= len(lines):
            continue

        raw = lines[idx + 1]

        if f == "Identity Number":
            digits = extract_int_from_string(raw)
            if digits is not None and len(digits) == 13:
                fields[f] = digits
        else:
            fields[f] = raw

    if fields["Identity Number"] is not None:
        return fields, "field_search"

    # Numeric fallback
    scores = []

    for line in lines:
        chars = [c for c in line if not c.isspace()]
        if not chars:
            scores.append(0.0)
            continue

        numeric = sum(c.isdigit() for c in chars)
        scores.append(numeric / len(chars))

    most_numeric_idx = max(range(len(scores)), key=scores.__getitem__)
    id_num = extract_int_from_string(lines[most_numeric_idx])

    if id_num is None:
        return fields, "unsuccessful"

    if len(id_num) == 13:
        fields["Identity Number"] = id_num
        return fields, "numeric:absolute"

    if len(id_num) > 13:
        fields["Identity Number"] = id_num[:13] # get first 13 nums.
        return fields, "numeric:truncation"

    # Failure in both methods
    return fields, "unsuccessful"


def format_fields_idbook(
        text: str,
        confidence: float = 0.4
) -> tuple[dict[str, str], str]:
    """
    We aim to build a function that targets key fields for the idbook, using these key steps.
        1. Search for the barcode, and aim to read (not yet developed), this will give us ALL key information.
        2. Target the first line of the OCR, with these key ideas in mind:
            a) Tesseract *commonly* mistakes 1 <-> I in the first part of the string. This can lead to
                'I.D. No.' as '1.D. No' for example.
            b) ID books have id numbers with whitespace.
        3. Try this for now.
    :param text: Clean OCR text
    :param confidence: Score based rejection of the best idx to go into the line-search
    :return: Dictionary of best fields
    """
    # Step 0: Check for empty OCR output
    if text is None:
        return {}, "unsuccessful:null_ocr"

    # Assume we are working with clean text, please refer to clean_raw_ocr_output()
    lines = text.splitlines()
    field_list: list[str] = ["VAN/SURNAME", "VOORNAME/FORENAMES"]
    fields: dict[str, str | None] = {k: None for k in field_list + ["Identity Number"]}
    method: str = "error:no_method_given"

    # Step 1: Barcode
    # INSERT CODE

    # Step 2: Search explicitly for the letters 'IDNo' in a line?
    # Think later if step 3 is unreliable...

    # Step 3: First Line
    id_num = extract_int_from_string(lines[0])

    if id_num is None:
        fields["Identity Number"] = None
        method = "first_row:null"
    elif len(id_num) == 13: # Assume we have a correct ID num
        fields["Identity Number"] = id_num
        method = "first_row:len=13"
    elif len(id_num) == 14: # Assume I -> 1
        fields["Identity Number"] = id_num[1:]
        method = "first_row:len=14"
    elif len(id_num) > 14: # Assume I -> 1 AND noise
        fields["Identity Number"] = id_num[1:14]
        method = "first_row:len>14"
    else:
        fields["Identity Number"] = None
        method = "unsuccessful:first_row" # 'first row' approach failed.

    # Step 4: Search for the most numeric line, if we were unsuccessful
    if method == "unsuccessful:first_row":
        id_num = numeric_line(lines)[1]

        if len(id_num) >= 13:
            fields["Identity Number"] = id_num[:13]
            method = "numeric_scoring"
        else:
            fields["Identity Number"] = None
            method = "unsuccessful:numeric_scoring"

    # Field Search for other fields --- less essential for now.
    for f in field_list:
        idx, score = search_for_line(text, f"{f}:", confidence=confidence)

        if idx is None or idx + 1 >= len(lines):
            continue

        raw = lines[idx + 1]
        fields[f] = raw

    # Standardise field names
    rename = {"VAN/SURNAME": "Surname", "VOORNAME/FORENAMES": "Names"}
    fields = {rename.get(k, k): v for k, v in fields.items()}

    return fields, method


# --- RAW OCR -> DICT ---
def ocr_to_dict_smartid(
        text: str,
        allowed_chars: set | None = None,
        filler_char: str = "",
        confidence: float = 0.5
) -> tuple[dict[str, str], str]:
    """
    Wrapping cleaning & formatting functions into one. Only need one line, and one set of arguments to:
        1. Clean raw ocr text
        2. Format ocr text into a dictionary
    :param text: Raw OCR text
    :param allowed_chars: Set of valid characters we want to search for. Typically upper/lower ASCII, numbers,
    space, hyphon, colon.
    :param filler_char: What we replace noisy characters with
    :param confidence: Score based rejection of the best idx.
    :return: Fields dictionary and the ID extraction method relied upon.
    """
    clean = clean_raw_ocr_output(text, allowed_chars, filler_char)
    return format_fields_smartid(clean, confidence) # fields(dict), method(str)


def ocr_to_dict_idbook(
        text: str,
        allowed_chars: set | None = None,
        filler_char: str = "",
        confidence: float = 0.5
) -> tuple[dict[str, str], str]:
    """
    Wrapping cleaning & formatting functions into one. Only need one line, and one set of arguments to:
        1. Clean raw ocr text
        2. Format ocr text into a dictionary
    :param text: Raw OCR text
    :param allowed_chars: Set of valid characters we want to search for. Typically upper/lower ASCII, numbers,
    space, hyphon, colon.
    :param filler_char: What we replace noisy characters with
    :param confidence: Score based rejection of the best idx.
    :return: Fields dictionary and the ID extraction method relied upon.
    """
    clean = clean_raw_ocr_output(text, allowed_chars, filler_char)
    return format_fields_idbook(clean, confidence) # fields(dict), method(str)


# --- VALIDATE ID NUMBER ---
def validate_id(id_number: str) -> bool:
    """
    Validate SA ID using date and Luhn check.
    :param id_number: SA ID number (string)
    :return: Boolean indicator of a valid ID
    """

    if not isinstance(id_number, str):
        raise TypeError(f"ID number must be of type str, not {type(id_number)}")

    # Digit + Length check
    if not id_number.isdigit() or len(id_number) != 13:
        return False

    # Date Check
    try:
        year = int(id_number[0:2])
        month = int(id_number[2:4])
        day = int(id_number[4:6])
        current_year = datetime.now().year % 100
        century = 1900 if year > current_year else 2000
        datetime(century + year, month, day)
    except ValueError:
        return False

    # Luhn check
    digits = [int(d) for d in id_number]
    odd_sum = sum(digits[::2])
    even_digits = digits[1::2]
    even_concat = ''.join(str(d * 2) for d in even_digits)
    even_sum = sum(int(d) for d in even_concat)
    return (odd_sum + even_sum) % 10 == 0