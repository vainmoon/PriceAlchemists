# Скрипты PriceAlchemists

Этот каталог содержит утилиты для проекта PriceAlchemists: скрипты для загрузки моделей и построения индекса FAISS.

## Доступные скрипты

### 1. `download_hf_models.py`
Загружает необходимые модели с HuggingFace:
- Модель BLIP для генерации описаний изображений
- Модель DistilBERT

Использование:
```bash
python download_hf_models.py
```

### 2. `build_faiss_index.py`
Создает индекс FAISS для поиска похожих изображений с использованием эмбеддингов ResNet50.

Основные настройки:
- Директория изображений: `backend/data/images/`
- CSV данные: `backend/data/aaa_advml_final_project.csv`
- Выходной индекс: `backend/weights/appindex.faiss`
- Выходной файл имён: `backend/weights/image_names.pkl`

Использование:
```bash
python build_faiss_index.py
```