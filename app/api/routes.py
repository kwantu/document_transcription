from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import shutil
import uuid
import os
import json
from pdf2image import convert_from_path
from app.services.ocr_service import process_image

router = APIRouter()

def parse_json(value):
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() in ("none", "null"):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON parameter")

@router.post("/ocr/image")
async def ocr_image(
    file: UploadFile = File(...),
    geometry: str | None = Form(None),
    preprocess: str | None = Form(None),
    postprocess: str | None = Form(None),
):
    tmp_dir = f"/tmp/{uuid.uuid4()}"
    os.makedirs(tmp_dir, exist_ok=True)

    file_path = os.path.join(tmp_dir, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ✅ PDF → IMAGE CONVERSION (CRITICAL PART)
    if file.content_type == "application/pdf":
        images = convert_from_path(file_path, dpi=300)
        if not images:
            raise HTTPException(status_code=400, detail="PDF has no pages")

        image_path = os.path.join(tmp_dir, "page_1.jpg")
        images[0].save(image_path, "JPEG")
    else:
        image_path = file_path

    # ✅ Now ALWAYS pass an IMAGE to YOLO
    result = process_image(
        image_path=image_path,
        geom=parse_json(geometry),
        prep=parse_json(preprocess),
        post=parse_json(postprocess),
    )

    return {
        "status": "success",
        "data": result,
    }
