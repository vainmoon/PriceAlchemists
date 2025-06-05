# sam.py
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import io
from PIL import Image

# Пути и тип модели подставьте свои
_checkpoint_path = "weights/sam_vit_b_01ec64.pth"
_model_type = "vit_b"

_sam = sam_model_registry[_model_type](checkpoint=_checkpoint_path)
_predictor = SamPredictor(_sam)

def segment_image_from_prompts(image_bytes: bytes, prompts: list[dict]) -> np.ndarray:
    """
    prompts: список словарей, где каждый элемент вида:
        {
          "type": "point",
          "points": [ {"x": X1, "y": Y1}, {"x": X2, "y": Y2}, ... ]
        }
    или
        {
          "type": "rectangle",
          "points": [ {"x": x0, "y": y0}, {"x": x1, "y": y1} ]
        }
    """

    # 1) Открываем изображение и конвертируем в numpy-array
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)

    # 2) Подготовим списки для SAM: возможны два типа подсказок
    point_list = []       # будем здесь накапливать все точки (если они есть)
    box_arr: np.ndarray   # если найдём rectangle, запишем сюда [x0, y0, x1, y1]
    use_box = False       # флаг, надо ли передавать box

    for item in prompts:
        t = item.get("type")
        pts = item.get("points", [])
        if t == "point":
            # точки: item["points"] — список словарей { "x": ..., "y": ... }
            for p in pts:
                # SAM ожидает формат [x, y] как вещественные (float или int) в одном массиве
                point_list.append([p["x"], p["y"]])
        elif t == "rectangle":
            # rectangle: два угловых пункта
            if len(pts) < 2:
                raise ValueError("Для прямоугольника нужно ровно 2 точки.")
            x0, y0 = pts[0]["x"], pts[0]["y"]
            x1, y1 = pts[1]["x"], pts[1]["y"]
            # гарантируем, что x0 < x1 и y0 < y1 (на всякий случай)
            x_min, x_max = min(x0, x1), max(x0, x1)
            y_min, y_max = min(y0, y1), max(y0, y1)
            box_arr = np.array([x_min, y_min, x_max, y_max])
            use_box = True
        else:
            # если вдруг пришёл неизвестный type
            raise ValueError(f"Unsupported prompt type: {t}")

    # 3) Установить изображение в предиктор
    _predictor.set_image(image)

    # 4) Если есть хотя бы одна точка, готовим массивы для point_coords и point_labels
    if len(point_list) > 0:
        input_points = np.array(point_list)
        # по умолчанию помечаем все точки как positive (label = 1)
        input_labels = np.ones(len(point_list), dtype=int)
    else:
        input_points = None
        input_labels = None

    # 5) Вызываем predict у SamPredictor
    if use_box and (input_points is not None):
        # если и box, и точки одновременно — SAM умеет принимать оба (точки+прямоугольник)
        masks, scores, logits = _predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=box_arr.reshape(1, 4),  # SAM ожидает shape (1, 4)
            multimask_output=False
        )
    elif use_box:
        # только прямоугольник
        masks, scores, logits = _predictor.predict(
            box=box_arr.reshape(1, 4),
            multimask_output=False
        )
    elif input_points is not None:
        # только точки
        masks, scores, logits = _predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
    else:
        raise ValueError("Нет валидных подсказок (ни точки, ни прямоугольника).")

    # Сам по SAM мы получаем булевую маску: masks[0] = 2D-логический массив (True внутри сегмента)
    mask = masks[0]

    # По аналогии с вашим старым кодом: затираем пиксели внутри маски нулями
    segmented = image.copy()
    segmented[mask] = 0
    return segmented
