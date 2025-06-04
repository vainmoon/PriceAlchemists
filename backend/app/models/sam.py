import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import io
from PIL import Image


_checkpoint_path = "weights/sam_vit_b_01ec64.pth"  # путь к модели
_model_type = "vit_b"

_sam = sam_model_registry[_model_type](checkpoint=_checkpoint_path)
_predictor = SamPredictor(_sam)

def segment_image_from_clicks(image_bytes: bytes, clicks: list[list[int]]) -> np.ndarray:

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    _predictor.set_image(image)

    input_points = np.array(clicks)
    input_labels = np.ones(len(clicks), dtype=int)

    masks, scores, logits = _predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )

    mask = masks[0]
    segmented = image.copy()
    segmented[~mask] = 0
    return segmented

