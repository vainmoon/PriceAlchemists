from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.utils.images import image_to_img_src, open_image

app = FastAPI(
    title='ML Inference API',
    description='Сервис для предсказания цены товара по изображению',
    version='1.0.0'
)
templates = Jinja2Templates(directory='app/templates')

@app.get('/', response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')

@app.post('/predict', response_class=HTMLResponse)
async def predict(file:UploadFile, request: Request):
    ctx = {}
    image = open_image(file.file)
    ctx['image'] = image_to_img_src(image)
    ctx['price'] = 123.45
    return templates.TemplateResponse(request=request, name='index.html', context=ctx)


    
