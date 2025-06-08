import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import io
from PIL import Image
import cv2

_checkpoint_path = "weights/sam_vit_b_01ec64.pth"
_model_type = "vit_b"

_sam = sam_model_registry[_model_type](checkpoint=_checkpoint_path)
_predictor = SamPredictor(_sam)

def segment_image_from_prompts(image_bytes, prompts: list[dict]) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)

    point_list = []
    box_arr: np.ndarray
    use_box = False

    for item in prompts:
        t = item.get("type")
        pts = item.get("points", [])
        if t == "point":
            for p in pts:
                point_list.append([p["x"], p["y"]])
        elif t == "rectangle":
            if len(pts) < 2:
                raise ValueError("Для прямоугольника нужно ровно 2 точки.")
            x0, y0 = pts[0]["x"], pts[0]["y"]
            x1, y1 = pts[1]["x"], pts[1]["y"]
            x_min, x_max = min(x0, x1), max(x0, x1)
            y_min, y_max = min(y0, y1), max(y0, y1)
            box_arr = np.array([x_min, y_min, x_max, y_max])
            use_box = True
        else:
            raise ValueError(f"Unsupported prompt type: {t}")

    _predictor.set_image(image)

    if len(point_list) > 0:
        input_points = np.array(point_list)
        input_labels = np.ones(len(point_list), dtype=int)
    else:
        input_points = None
        input_labels = None

    if use_box and (input_points is not None):
        masks, scores, logits = _predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=box_arr.reshape(1, 4),
            multimask_output=False
        )
    elif use_box:
        masks, scores, logits = _predictor.predict(
            box=box_arr.reshape(1, 4),
            multimask_output=False
        )
    elif input_points is not None:
        masks, scores, logits = _predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
    else:
        raise ValueError("Нет валидных подсказок (ни точки, ни прямоугольника).")

    mask = ~masks[0]

    mask_image = np.stack([mask.astype(np.uint8) * 255] * 3, axis=-1)
    
    is_success, encoded_jpg = cv2.imencode(".jpg", mask_image)
    if not is_success:
        raise ValueError("Failed to encode mask as JPEG")
        
    return encoded_jpg.tobytes()
