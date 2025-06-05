from flask import Flask, request, jsonify
import torch
import os

# Импортируем функции для инференса
from inference import full_inference_pipeline, load_models

# Настройка логирования
import logging
logging.basicConfig(level=logging.DEBUG)

# Инициализируем Flask-приложение
app = Flask(__name__)

# Загружаем все модели один раз при запуске сервера (важно!)
device = "cuda" if torch.cuda.is_available() else "cpu"
models = load_models(device=device, verbose=True)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint для предсказания.
    
    Ожидает:
        - POST-запрос с файлом 'image' в формате form-data
    
    Возвращает:
        - JSON-ответ с:
            - category (категория товара)
            - subcategory (подкатегория товара)
            - generated_title (название товара, сгенерированное BLIP)
            - predicted_price (предсказанная цена)
    """
    if 'image' not in request.files:
        return jsonify({"error": "Изображение не предоставлено"}), 400

    image_file = request.files['image']
    temp_path = f"/tmp/{image_file.filename}"
    
    try:
        # Сохраняем изображение временно
        image_file.save(temp_path)

        # Выполняем пайплайн инференса
        result = full_inference_pipeline(image_path=temp_path, device=device, models=models)

        # Удаляем временное изображение
        os.remove(temp_path)

        return jsonify({
        "category": result["category"],
        "subcategory": result["subcategory"],
        "title": result["title"],
        "price": round(result["price"], 2)
        }).data.decode('utf-8') # для вывода русского языка

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Убедимся, что временный файл удален
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
