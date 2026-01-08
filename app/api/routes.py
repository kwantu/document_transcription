from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import uuid
import os
from pdf2image import convert_from_path

from app.services.ocr_service import process_image
from app.core.pipeline import (
    GeometryConfig,
    PreprocessConfig,
    PostprocessConfig,
)

router = APIRouter()


@router.post("/documents/extract")
async def ocr_image(
    file: UploadFile = File(...),
):
    tmp_dir = f"/tmp/{uuid.uuid4()}"
    os.makedirs(tmp_dir, exist_ok=True)

    file_path = os.path.join(tmp_dir, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ✅ PDF → IMAGE
    if file.content_type == "application/pdf":
        images = convert_from_path(file_path, dpi=300)
        if not images:
            raise HTTPException(status_code=400, detail="PDF has no pages")

        image_path = os.path.join(tmp_dir, "page_1.jpg")
        images[0].save(image_path, "JPEG")
    else:
        image_path = file_path

    # ✅ HARD-CODED INTERNAL CONFIGS
    geom_cfg = GeometryConfig(
        id_class=0,
        metadata_target_height=420,
        correction_angle=10.0,
    )

    prep_cfg = PreprocessConfig(
        k_denoise=3,
        thresh_block=13,
        thresh_c=3,
    )

    post_cfg = PostprocessConfig(
        confidence=0.5,
    )

    # ✅ CONVERT TO DICTS (what ocr_service expects)
    result = process_image(
        image_path=image_path,
        geom=geom_cfg.__dict__,
        prep=prep_cfg.__dict__,
        post=post_cfg.__dict__,
    )

    return {
        "status": "success",
        "data": result,
    }
