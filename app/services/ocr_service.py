from ultralytics import YOLO
from app.core.pipeline import (
    img_2_json,
    GeometryConfig,
    PreprocessConfig,
    PostprocessConfig
)

# Load once on startup
yolo_model = YOLO("yolo11s.pt")

def process_image(
    image_path: str,
    geom: dict | None = None,
    prep: dict | None = None,
    post: dict | None = None
):
    geom_cfg = GeometryConfig(**geom) if geom else GeometryConfig()
    prep_cfg = PreprocessConfig(**prep) if prep else PreprocessConfig()
    post_cfg = PostprocessConfig(**post) if post else PostprocessConfig()

    return img_2_json(
        yolo_model=yolo_model,
        img_path=image_path,
        geom_params=geom_cfg,
        prep_params=prep_cfg,
        post_params=post_cfg,
        save_process=False
    )
