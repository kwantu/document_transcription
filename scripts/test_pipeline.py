from ultralytics import YOLO

# IMPORTANT: absolute imports
from app.core.pipeline import (
    img_2_json,
    GeometryConfig,
    PreprocessConfig,
    PostprocessConfig,
)

if __name__ == "__main__":
    yolo_model = YOLO("yolo11s.pt")

    geom = GeometryConfig(id_class=0, metadata_target_height=420, correction_angle=10.0)
    prep = PreprocessConfig(k_denoise=3, thresh_block=13, thresh_c=3)
    post = PostprocessConfig(confidence=0.5)

    result = img_2_json(
        yolo_model,
        "samples/s2.jpeg",
        dest_path="output",
        geom_params=geom,
        prep_params=prep,
        post_params=post
    )
    print("RESULT for Image:")
    print(result)

    
