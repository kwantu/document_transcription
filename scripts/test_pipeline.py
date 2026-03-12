
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO
import cv2
from pathlib import Path

# IMPORTANT: absolute imports
from app.core.pipeline import (
    img_2_json_v2,
    load_params,
    SMARTID,
    IDBOOK
)

if __name__ == "__main__":

    image_path = "samples/s2.jpg"

    # Load image
    img = cv2.imread(image_path)
    img_id = Path(image_path).stem

    # Run document classifier → get pipeline parameters
    params = load_params(img)

    # Select YOLO model based on detected class
    yolo_model = SMARTID if params.geometry_config.id_class == 0 else IDBOOK

    # Run pipeline
    result = img_2_json_v2(
        yolo_model=yolo_model,
        img=img,
        img_id=img_id,
        dest_path="output",
        effnet_dict=params.doc_class_info,
        geom_params=params.geometry_config,
        prep_params=params.preprocess_config,
        post_params=params.postprocess_config,
        save_process=True
    )

    print("\nRESULT for Image:")
    print(result)