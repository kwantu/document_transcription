from pathlib import Path
import cv2

from app.core.pipeline import (
    img_2_json_v2,
    load_params,
    SMARTID,
    IDBOOK
)

def process_image(image_path):

    img = cv2.imread(image_path)
    img_id = Path(image_path).stem

    params = load_params(img)

    yolo_model = SMARTID if params.geometry_config.id_class == 0 else IDBOOK

    result = img_2_json_v2(
        yolo_model=yolo_model,
        img=img,
        img_id=img_id,
        dest_path="output",
        effnet_dict=params.doc_class_info,
        geom_params=params.geometry_config,
        prep_params=params.preprocess_config,
        post_params=params.postprocess_config,
        save_process=False
    )

    return result