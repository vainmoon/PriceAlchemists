# build_index.py

import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import faiss
import pickle
import pandas as pd
from tqdm import tqdm

# Константы
IMAGE_DIR = "backend/data/images/"
CSV_PATH = "backend/data/aaa_advml_final_project.csv"           # CSV с колонкой 'filename'
CSV_COLUMN = "image_id"           # Название столбца с именами файлов
OUTPUT_INDEX = "backend/weights/appindex.faiss"
OUTPUT_NAMES = "backend/weights/image_names.pkl"

# Загрузка датафрейма
df = pd.read_csv(CSV_PATH)
image_files = df[CSV_COLUMN].dropna().astype(str).unique().tolist()

# Модель ResNet50
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Преобразование
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor).squeeze().numpy()
    return emb / np.linalg.norm(emb)

# Считаем эмбеддинги только для нужных файлов
valid_names = []
embeddings = []

for fname in tqdm(image_files):
    path = os.path.join(IMAGE_DIR, fname + '.jpg')
    if not os.path.isfile(path):
        print(f"[!] Файл не найден: {path}")
        continue
    try:
        emb = get_embedding(path)
        embeddings.append(emb)
        valid_names.append(fname)
    except Exception as e:
        print(f"[!] Ошибка при обработке {fname}: {e}")

embeddings = np.array(embeddings).astype("float32")

# FAISS-индекс
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Сохраняем
faiss.write_index(index, OUTPUT_INDEX)
with open(OUTPUT_NAMES, "wb") as f:
    pickle.dump(valid_names, f)

print(f"[✓] Индекс сохранён: {OUTPUT_INDEX}")
print(f"[✓] Названия изображений сохранены: {OUTPUT_NAMES} ({len(valid_names)} шт.)")
