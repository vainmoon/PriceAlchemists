import torch
import timm
import io
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = "mps" if torch.backends.mps.is_available() \
    else "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDING_SHAPE = 512

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

base_model = timm.create_model('swin_tiny_patch4_window7_224',
                               pretrained=True,
                               num_classes=0)
base_model.to(device).eval()


class SwinEmbeddingModel(nn.Module):
    def __init__(self, base_model, out_dim):
        super().__init__()
        self.base = base_model  # это model.forward_features
        self.head = nn.Linear(self.base.num_features, out_dim)

    def forward(self, x):
        features = self.base(x)
        return self.head(features)


model = SwinEmbeddingModel(base_model, EMBEDDING_SHAPE).to(device).eval()
model.load_state_dict(torch.load("weights/swin_model_weights.pth",
                                 weights_only=True))
model = model.to(device).eval()

df = pd.read_pickle("data/df_faiss.pkl")
index = faiss.read_index("data/faiss_idx.index")


def get_embedding_from_image_bytes(image) -> np.ndarray:
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image_tensor).squeeze().cpu().numpy().flatten()
    return embedding.astype("float32")


def predict_price(image_bytes: bytes) -> float:
    emb = get_embedding_from_image_bytes(image_bytes).reshape(1, -1)
    _, indices = index.search(emb, k=4)
    neighbor_prices = [df.iloc[idx]["price"]
                       for idx in indices[0][1:] if idx < len(df)]
    return float(np.mean(neighbor_prices)) if neighbor_prices else 0.0


def get_top3_similar_item_ids(image) -> list[int]:
    emb = get_embedding_from_image_bytes(image).reshape(1, -1)
    _, indices = index.search(emb, k=4)
    image_ids = [df.iloc[idx]["image_id"] for idx in indices[0]]
    item_ids = [df[df["image_id"] == iid]["item_id"].values[0]
                for iid in image_ids[1:]]
    return image_ids


if __name__ == "__main__":
    # Example of usage
    img_path = "image.jpeg"
    with open(img_path, "rb") as f:
        image_bytes = f.read()

    pred_price = predict_price(image_bytes)
    top3_item_ids = get_top3_similar_item_ids(image_bytes)

    print(f"Predicted price: {pred_price:.2f}₽")
    print(f"Top-3 similar item_ids: {top3_item_ids}")
