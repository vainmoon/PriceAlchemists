import torch
import numpy as np
import pandas as pd
import faiss
import pickle
from PIL import Image
from torchvision import models, transforms
import os

# Пути
IMAGE_DIR = "data/images/"
CSV_PATH = "data/aaa_advml_final_project.csv"
INDEX_PATH = "weights/appindex.faiss"
NAMES_PATH = "weights/image_names.pkl"

# Загрузка датафрейма
df = pd.read_csv(CSV_PATH)
df["image_id"] = df["image_id"].astype(str)

# Загрузка FAISS индекса и имён
index = faiss.read_index(INDEX_PATH)
with open(NAMES_PATH, "rb") as f:
    image_ids = pickle.load(f)  # список image_id в виде строк

# Карта image_id → строка DataFrame
id_to_row = {row["image_id"]: row for _, row in df.iterrows()}

# Модель ResNet50
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Преобразование изображения
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Получение эмбеддинга
def get_embedding_from_image_resnet(image: Image.Image) -> np.ndarray:
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor).squeeze().numpy()
    emb = emb / np.linalg.norm(emb)
    return emb.astype("float32")

# Получение top-3 похожих image_id
def get_top3_similar_item_ids_faiss(image: Image.Image) -> list[str]:
    emb = get_embedding_from_image_resnet(image)
    distances, indices = index.search(np.array([emb]), k=3)
    return [image_ids[i] for i in indices[0]]

# Пример: предсказание цены по среднему ближайших
def predict_price_from_image_faiss(image: Image.Image) -> float:
    top_ids = get_top3_similar_item_ids_faiss(image)
    prices = []
    for image_id in top_ids:
        row = id_to_row.get(image_id)
        if row is not None and "price" in row:
            prices.append(row["price"])
    if prices:
        return float(np.maximum(np.mean(prices), 100))
    return 100.0

# Загрузка изображения по image_id
def load_image_by_id(image_id: str) -> Image.Image:
    path = os.path.join(IMAGE_DIR, image_id + ".jpg")
    return Image.open(path).convert("RGB")
