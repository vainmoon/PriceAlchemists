from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Response
from fastapi.staticfiles import StaticFiles

import numpy as np
import cv2
import json


from app.utils.images import image_to_img_src, open_image
from app.models.sam import segment_image_from_clicks

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

@app.post('/predict', response_class=HTMLResponse)
async def predict(file:UploadFile, request: Request):
    ctx = {}
    image = open_image(file.file)
    ctx['image'] = image_to_img_src(image)
    ctx['price'] = 123.45
    return templates.TemplateResponse(request=request, name='index.html', context=ctx)

@app.post("/segment")
async def segment(image: UploadFile = File(...), clicks: str = Form(...)):
    try:
        clicks_data = json.loads(clicks)
        clicks_list = [[int(click['x']), int(click['y'])] for click in clicks_data]
        

        image_bytes = await image.read()

        result_image = segment_image_from_clicks(image_bytes, clicks_list)
        
        if result_image is None:
            raise ValueError("Segmentation failed to produce an output")
            
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', result_image)

            
        # Return image response
        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={
                "Content-Type": "image/jpeg",
                "Content-Disposition": "inline"
            }
        )
    
    except Exception as e:
        print(f"Error during segmentation: {str(e)}")  # Log the error
        import traceback
        traceback.print_exc()  # Print full error traceback
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )