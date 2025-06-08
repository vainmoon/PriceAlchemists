from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Response
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import HTTPException

import numpy as np
import cv2
import json
import torch
import base64

from app.utils.images import open_image, open_mask, apply_mask, crop_image_by_mask
from app.models.sam import segment_image_from_prompts
from app.models.price_predictor import load_models, full_inference_pipeline
#from app.models.simple_faiss import get_top3_similar_item_ids
from backend.app.models.price_predictor_faiss import find_top3_similar

app = FastAPI(
    title='ML Inference API',
    description='Сервис для предсказания цены товара по изображению',
    version='1.0.0'
)
templates = Jinja2Templates(directory='app/templates')
app.mount("/static", StaticFiles(directory="app/static"))


device = "cuda" if torch.cuda.is_available() else "cpu"
models = load_models(device=device, verbose=True)


@app.get('/', response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')

@app.post('/predict')
async def predict(file:UploadFile, mask: UploadFile):
    image = open_image(file.file)
    mask_image = open_mask(mask.file)

    if mask_image.size != image.size:
        print("Размер маски и изображения не совпадают — используется оригинальное изображение.")
        input_image = image
    else:
        masked_image = apply_mask(image, mask_image)
        input_image = crop_image_by_mask(masked_image, mask_image)

    prediction = full_inference_pipeline(input_image, device=device, models=models)
    print(prediction)

    # Получаем top-3 похожих товаров
    similar_items = find_top3_similar(input_image)
    print(f"Top-3 похожих товаров: {similar_items}")

    # Заглушка для отправления изображений рекомендаций
    with open(f"data/images/{similar_items[0]}.jpg", "rb") as f:
        img_data_1 = base64.b64encode(f.read()).decode("utf-8")

    with open(f"data/images/{similar_items[1]}.jpg", "rb") as f:
        img_data_2 = base64.b64encode(f.read()).decode("utf-8")

    with open(f"data/images/{similar_items[2]}.jpg", "rb") as f:
        img_data_3 = base64.b64encode(f.read()).decode("utf-8")

    return JSONResponse(content={
        "price": prediction['price'],
        "similarProducts": [img_data_1, img_data_2, img_data_3]
    })


@app.post("/segment")
async def segment(
    file: UploadFile = File(...),          # ожидаем multipart/form-data: поле "file"
    prompts: str = Form(...)                # и поле "prompts" как JSON-строка
):
    """
    Ожидает multipart/form-data с:
      - file: UploadFile (изображение)
      - prompts: str (JSON-массив подсказок, например:
            [
              {"type":"point","points":[{"x":100,"y":200}]},
              {"type":"rectangle","points":[{"x":50,"y":50},{"x":200,"y":200}]}
            ]
        )
    """

    try:
        # 1) Прочитать байты изображения
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Файл не был передан или он пустой.")

        # 2) Распарсить JSON из поля "prompts"
        try:
            prompts_list = json.loads(prompts)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Поле 'prompts' должно быть валидной JSON-строкой.")

        if not isinstance(prompts_list, list):
            raise HTTPException(status_code=400, detail="Поле 'prompts' должно быть списком подсказок.")

        # 3) Вызвать SAM-модель
        segmented_np = segment_image_from_prompts(image_bytes, prompts_list)
        # segmented_np теперь возвращает байты JPEG

        # 4) Вернуть байты JPEG клиенту
        return Response(
            content=segmented_np,  # Уже закодировано как JPEG
            media_type="image/jpeg",
            headers={
                "Content-Type": "image/jpeg",
                "Content-Disposition": "inline"
            }
        )

    except HTTPException as he:
        # Если мы сами бросили HTTPException, отдадим его как есть
        raise he
    except Exception as e:
        # Ловим всё остальное и возвращаем 400 + текст ошибки для дебага
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
    
