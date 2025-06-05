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


from app.utils.images import image_to_img_src, open_image
from app.models.sam import segment_image_from_prompts

app = FastAPI(
    title='ML Inference API',
    description='Сервис для предсказания цены товара по изображению',
    version='1.0.0'
)
templates = Jinja2Templates(directory='app/templates')
app.mount("/static", StaticFiles(directory="app/static"))

@app.get('/', response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')

@app.post('/predict')
async def predict(file:UploadFile):
    ctx = {}
    image = open_image(file.file)
    ctx['image'] = image_to_img_src(image)
    ctx['price'] = 123.45
    return JSONResponse(content={"price": 123})

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
        # segmented_np — numpy array shape (H, W, 3), dtype uint8

        # 4) Закодировать результат в JPEG
        #    OpenCV ожидает BGR-формат, а у нас RGB, поэтому делаем [ : , : , ::-1 ]
        is_success, encoded_jpg = cv2.imencode(".jpg", segmented_np[:, :, ::-1])
        if not is_success:
            raise HTTPException(status_code=500, detail="Не удалось закодировать изображение в JPEG.")

        # 5) Вернуть байты JPEG клиенту
        return Response(
            content=encoded_jpg.tobytes(),
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