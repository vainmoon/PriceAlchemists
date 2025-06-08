import faiss
import pickle
import timm
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


# Загружаем Embedder
model = timm.create_model('convnext_base', pretrained=True, num_classes=0)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Загружаем индекс FAISS
index = faiss.read_index("weights/faiss_index.bin")

with open("weights/image_ids.pkl", "rb") as f:
    valid_image_ids = pickle.load(f)


def get_embedding(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(img_tensor)
    
    return embedding.cpu().numpy()


def find_top3_similar(image: Image.Image):
    query_embedding = get_embedding(image)
    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, k=3)
    top_image_ids = [valid_image_ids[i] for i in I[0]]
    return top_image_ids