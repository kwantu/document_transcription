from pydantic import BaseModel

class GeometryConfigSchema(BaseModel):
    id_class: int = 0
    metadata_target_height: int = 400
    correction_angle: float = 0.0

class PreprocessConfigSchema(BaseModel):
    k_denoise: int = 3
    thresh_block: int = 13
    thresh_c: int = 3

class PostprocessConfigSchema(BaseModel):
    confidence: float = 0.5
