from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Response
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import HTTPException

import json
import torch
import base64

from app.utils.images import open_image, open_mask, apply_mask, crop_image_by_mask
from app.models.sam import segment_image_from_prompts
from app.models.price_predictor import load_models, full_inference_pipeline
from app.models.faiss import find_top3_similar


app = FastAPI(
    title='PriceAlchemists',
    description='Сервис для предсказания цены товара по изображению',
    version='1.0.0'
)
templates = Jinja2Templates(directory='app/templates')
app.mount("/static", StaticFiles(directory="app/static"))


device = "cuda" if torch.cuda.is_available() else "cpu"
price_predictor_model = load_models(device=device, verbose=True)

def predict_price(image):
    prediction = full_inference_pipeline(image, device=device, models=price_predictor_model)
    return prediction['price']

def get_similar_images(image):
    similar_images_id = find_top3_similar(image)
    similar_images = []
    for image_id in similar_images_id:
        with open(f'data/images/{image_id}.jpg', 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
            similar_images.append(img_data)
    return similar_images

@app.get('/', response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')


@app.get('/predict_for_web')
async def predict_for_web_get():
    return RedirectResponse(url='/', status_code=303)


@app.post('/predict_for_web', response_class=HTMLResponse)
async def predict_for_web(request: Request, file: UploadFile):
    ctx = {}
    
    if not file or not file.filename:
        ctx['error'] = 'Пожалуйста, выберите изображение перед отправкой'
        return templates.TemplateResponse(request=request, name='index.html', context=ctx)
    
    if not file.content_type.startswith('image/'):
        ctx['error'] = 'Пожалуйста, загрузите файл изображения'
        return templates.TemplateResponse(request=request, name='index.html', context=ctx)
    
    image = open_image(file.file)

    ctx['price'] = predict_price(image)
    ctx['similarProducts'] = get_similar_images(image)

    return templates.TemplateResponse(request=request, name='index.html', context=ctx)


@app.post('/predict_for_mobile')
async def predict_for_mobile(file:UploadFile, mask: UploadFile):
    image = open_image(file.file)
    mask_image = open_mask(mask.file)

    if mask_image.size != image.size:
        input_image = image
    else:
        masked_image = apply_mask(image, mask_image)
        input_image = crop_image_by_mask(masked_image, mask_image)

    prediction = predict_price(input_image)
    similar_images = get_similar_images(input_image)

    return JSONResponse(content={
        "price": prediction['price'],
        "similarProducts": similar_images
    })


@app.post("/segment")
async def segment(file: UploadFile, prompts: str):
    image = open_image(file.file)
    prompts_list = json.loads(prompts)

    segmented_np = segment_image_from_prompts(image, prompts_list)

    return Response(
        content=segmented_np,
        media_type="image/jpeg",
        headers={
            "Content-Type": "image/jpeg",
            "Content-Disposition": "inline"
        }
    )