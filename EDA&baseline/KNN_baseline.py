import torch
import clip
from PIL import Image
import io
import numpy as np
from joblib import load
import pandas as pd


device = "mps" if torch.backends.mps.is_available() \
    else "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

knn_model = load("Models/knn_model.joblib")
df = pd.read_csv("Data/aaa_advml_final_project.csv")


def get_embedding_from_image_bytes_knn(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
    return embedding.cpu().numpy().flatten()


def predict_price_from_image_knn(image_bytes: bytes) -> float:
    emb = get_embedding_from_image_bytes_knn(image_bytes)
    price = knn_model.predict([emb])[0]
    return float(np.maximum(price, 100))


def get_top3_similar_item_ids_knn(image_bytes: bytes) -> list[int]:
    emb = get_embedding_from_image_bytes_knn(image_bytes)
    distances, indices = knn_model.kneighbors([emb], n_neighbors=3)
    return df.iloc[indices[0]]["item_id"].tolist()
