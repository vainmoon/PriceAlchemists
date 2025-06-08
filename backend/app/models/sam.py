from typing import List, Dict, Union, Tuple, Optional
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import io
from PIL import Image
import cv2


_checkpoint_path = "weights/sam_vit_b_01ec64.pth"
_model_type = "vit_b"


_sam = sam_model_registry[_model_type](checkpoint=_checkpoint_path)
_predictor = SamPredictor(_sam)


def segment_image_from_prompts(
    image_bytes: bytes,
    prompts: List[Dict[str, Union[str, List[Dict[str, int]]]]]
) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)

    point_list: List[List[int]] = []
    box_arr: Optional[np.ndarray] = None
    use_box = False

    for item in prompts:
        prompt_type = item.get("type")
        points = item.get("points", [])

        if prompt_type == "point":
            for point in points:
                point_list.append([point["x"], point["y"]])
        elif prompt_type == "rectangle":
            if len(points) < 2:
                raise ValueError("Для прямоугольника нужно ровно 2 точки.")
            
            x0, y0 = points[0]["x"], points[0]["y"]
            x1, y1 = points[1]["x"], points[1]["y"]
            
            x_min, x_max = min(x0, x1), max(x0, x1)
            y_min, y_max = min(y0, y1), max(y0, y1)
            
            box_arr = np.array([x_min, y_min, x_max, y_max])
            use_box = True
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")

    _predictor.set_image(image)


    if point_list:
        input_points = np.array(point_list)
        input_labels = np.ones(len(point_list), dtype=int)
    else:
        input_points = None
        input_labels = None


    if use_box and input_points is not None:

        masks, _, _ = _predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=box_arr.reshape(1, 4),
            multimask_output=False
        )
    elif use_box:

        masks, _, _ = _predictor.predict(
            box=box_arr.reshape(1, 4),
            multimask_output=False
        )
    elif input_points is not None:

        masks, _, _ = _predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
    else:
        raise ValueError("Нет валидных подсказок (ни точки, ни прямоугольника).")


    mask = ~masks[0]
    mask_image = np.stack([mask.astype(np.uint8) * 255] * 3, axis=-1)
    

    success, encoded_jpg = cv2.imencode(".jpg", mask_image)
    if not success:
        raise ValueError("Failed to encode mask as JPEG")
        
    return encoded_jpg.tobytes()
