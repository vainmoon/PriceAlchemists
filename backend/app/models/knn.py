import torch
from PIL import Image
import io
import numpy as np
from joblib import load
import pandas as pd
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

knn_model = load("weights/knn_model.joblib")
df = pd.read_csv("data/aaa_advml_final_project.csv")


def get_embedding_from_image_bytes_knn(image) -> np.ndarray:
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
    return embedding.cpu().numpy().flatten()


def predict_price_from_image_knn(image) -> float:
    emb = get_embedding_from_image_bytes_knn(image)
    price = knn_model.predict([emb])[0]
    return float(np.maximum(price, 100))


def get_top3_similar_item_ids_knn(image) -> list[int]:
    emb = get_embedding_from_image_bytes_knn(image)
    distances, indices = knn_model.kneighbors([emb], n_neighbors=3)
    return df.iloc[indices[0]]["image_id"].tolist()
