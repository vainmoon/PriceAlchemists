import torch
import clip
import io
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from joblib import load
from sklearn.metrics.pairwise import cosine_distances


device = "mps" if torch.backends.mps.is_available() \
    else "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

faiss_indexes = load("Models/faiss_indexes.joblib")
subcat_id_to_y = load("Models/faiss_prices.joblib")
subcat_centroids = load("Models/faiss_centroids.joblib")

df = pd.read_csv("Data/aaa_advml_final_project.csv")

image_subcat_dict = df.set_index("image_id")["subcategory_name"].to_dict()
image_id_to_item_id = df.set_index("image_id")["item_id"].to_dict()


def get_embedding_from_image_bytes(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(image_tensor)
    return emb.cpu().numpy().flatten().astype("float32")


def predict_price_faiss(image_bytes: bytes) -> float:
    emb = get_embedding_from_image_bytes(image_bytes).reshape(1, -1)

    centroid_matrix = np.stack(list(subcat_centroids.values()))
    centroid_keys = list(subcat_centroids.keys())
    distances = cosine_distances(emb, centroid_matrix)[0]
    closest_subcat = centroid_keys[np.argmin(distances)]

    faiss.normalize_L2(emb)
    index = faiss_indexes[closest_subcat]
    D, Indices = index.search(emb, k=5)

    neighbor_prices = subcat_id_to_y[closest_subcat][Indices[0]]
    pred_price = np.mean(neighbor_prices)
    return float(max(pred_price, 100))


def get_top3_similar_item_ids_faiss(image_bytes: bytes) -> list[int]:
    emb = get_embedding_from_image_bytes(image_bytes).reshape(1, -1)

    centroid_matrix = np.stack(list(subcat_centroids.values()))
    centroid_keys = list(subcat_centroids.keys())
    distances = cosine_distances(emb, centroid_matrix)[0]
    closest_subcat = centroid_keys[np.argmin(distances)]

    faiss.normalize_L2(emb)
    index = faiss_indexes[closest_subcat]
    D, Indices = index.search(emb, k=3)

    subcat_df = df[df["subcategory_name"] == closest_subcat]
    image_ids_in_subcat = subcat_df["image_id"].tolist()
    top_image_ids = [image_ids_in_subcat[idx] for idx in Indices[0]]
    return [image_id_to_item_id[iid] for iid in top_image_ids]
