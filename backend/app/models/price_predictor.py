import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os

import timm
from transformers import DistilBertModel, DistilBertTokenizer, BlipProcessor, BlipForConditionalGeneration
import torchvision.transforms as transforms



class PricePredictor(nn.Module):
    def __init__(self, image_embedding_dim=512, text_embedding_dim=512, hidden_dim=256,
                 num_categories=2, num_subcategories=14):
        super().__init__()

        # Энкодер изображения
        self.image_encoder = timm.create_model('convnext_base', pretrained=True, num_classes=0)
        self.image_proj = nn.Linear(1024, image_embedding_dim)

        # Энкодер текста (заголовка)
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_proj = nn.Linear(768, text_embedding_dim)

        # Эмбеддинги для категорий и подкатегорий
        self.category_emb = nn.Embedding(num_categories, 128)
        self.subcategory_emb = nn.Embedding(num_subcategories, 128)

        # Полносвязная сеть для предсказания цены
        total_input_dim = image_embedding_dim + text_embedding_dim + 128 + 128
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image, input_ids, attention_mask, category_ids, subcategory_ids, return_intermediates=False):
        img_feat = self.image_encoder(image)
        img_emb = self.image_proj(img_feat)

        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = self.text_proj(text_out.last_hidden_state[:, 0])

        cat_emb = self.category_emb(category_ids)
        subcat_emb = self.subcategory_emb(subcategory_ids)

        fused = torch.cat([img_emb, text_emb, cat_emb, subcat_emb], dim=1)
        price = self.mlp(fused)

        if return_intermediates:
            return price, img_emb  # <-- Возвращает цену и embedding
        else:
            return price


class CategorySubcategoryClassifier(nn.Module):
    def __init__(self, num_categories, num_subcategories):
        super().__init__()
        self.backbone = timm.create_model('convnext_base', pretrained=True, num_classes=0)
        feat_dim = 1024
        self.category_head = nn.Linear(feat_dim, num_categories)
        self.subcategory_head = nn.Linear(feat_dim, num_subcategories)

    def forward(self, x):
        features = self.backbone(x)
        category_logits = self.category_head(features)
        subcategory_logits = self.subcategory_head(features)
        return category_logits, subcategory_logits


def load_models(device="cuda", verbose=False):
    """
    Загружает обученные модели и вспомогательные компоненты.
    """
    # Преобразования изображений
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Модель BLIP для генерации названия товара
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    # Токенизатор текста
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    dataset_category_to_idx = {
        "Электроника": 0,
        "Личные вещи": 1
    }

    dataset_subcategory_to_idx = {
        "Телефоны": 0,
        "Красота и здоровье": 1,
        "Товары для детей и игрушки": 2,
        "Одежда, обувь, аксессуары": 3,
        "Детская одежда и обувь": 4,
        "Настольные компьютеры": 5,
        "Игры, приставки и программы": 6,
        "Ноутбуки": 7,
        "Часы и украшения": 8,
        "Аудио и видео": 9,
        "Товары для компьютера": 10,
        "Оргтехника и расходники": 11,
        "Фототехника": 12,
        "Планшеты и электронные книги": 13
    }

    num_categories = len(dataset_category_to_idx)
    num_subcategories = len(dataset_subcategory_to_idx)

    # Загрузка модели классификации категории
    category_model = CategorySubcategoryClassifier(num_categories, num_subcategories).to(device)
    category_model.load_state_dict(torch.load("weights/best_cat_model.pth", map_location=device, weights_only=True))
    category_model.eval()

    # Загрузка модели предсказания цены
    price_model = PricePredictor(
        num_categories=num_categories,
        num_subcategories=num_subcategories
    ).to(device)
    price_model.load_state_dict(torch.load("weights/model_epoch_5.pth", map_location=device, weights_only=True))
    price_model.eval()

    if verbose:
        print("✅ Модели успешно загружены.")

    return {
        "transform": transform,
        "blip_processor": blip_processor,
        "blip_model": blip_model,
        "tokenizer": tokenizer,
        "category_model": category_model,
        "price_model": price_model,
        "category_map": {v: k for k, v in dataset_category_to_idx.items()},
        "subcategory_map": {v: k for k, v in dataset_subcategory_to_idx.items()}
    }


def full_inference_pipeline(image, device="cuda", models=None, return_intermediates=False):
    """
    Полный пайплайн инференса:
        1. Определение категории и подкатегории
        2. Генерация названия товара с помощью BLIP
        3. Предсказание цены
    """
    if models is None:
        models = load_models(device=device)
    
    transform = models["transform"]
    blip_processor = models["blip_processor"]
    blip_model = models["blip_model"]
    tokenizer = models["tokenizer"]
    category_model = models["category_model"]
    price_model = models["price_model"]
    idx_to_category = models["category_map"]
    idx_to_subcategory = models["subcategory_map"]

    # Трансформация для классификатора
    image_tensor_val = transform(image).unsqueeze(0).to(device)

    # Шаг 1: Предсказание категории и подкатегории
    with torch.no_grad():
        cat_logits, subcat_logits = category_model(image_tensor_val)
    predicted_cat_id = cat_logits.argmax().item()
    predicted_subcat_id = subcat_logits.argmax().item()

    predicted_category = idx_to_category[predicted_cat_id]
    predicted_subcategory = idx_to_subcategory[predicted_subcat_id]

    # Шаг 2: Генерация названия с помощью BLIP
    text = 'an advertisement for' # можно удалить
    inputs = blip_processor(image, text, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    generated_title = blip_processor.decode(out[0], skip_special_tokens=True)

    # Обрезаем строго по промпту
    prompt_len = 20 # менятся в соотвествии с len(text) 
    generated_title = generated_title[prompt_len:].strip()

    # Шаг 3: Токенизация названия
    encoding = tokenizer(
        generated_title,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Трансформация изображения для модели цены
    image_tensor_price = transform(image).unsqueeze(0).to(device)

    # Преобразование ID категории/подкатегории в тензоры
    cat_tensor = torch.tensor([predicted_cat_id], dtype=torch.long).to(device)
    subcat_tensor = torch.tensor([predicted_subcat_id], dtype=torch.long).to(device)

    # Шаг 4: Предсказание цены
    with torch.no_grad():
        price_log = price_model(
            image_tensor_price,
            input_ids,
            attention_mask,
            cat_tensor,
            subcat_tensor,
            return_intermediates
        )

    predicted_price = np.expm1(price_log.cpu().item()) # возвращаем цену в исходном масштабе

    return {
        "category": predicted_category,
        "subcategory": predicted_subcategory,
        "title": generated_title,
        "price": predicted_price
    }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Запуск пайплайна инференса на изображении.")
    parser.add_argument("--image", type=str, help="Путь к файлу изображения.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Устройство для выполнения (по умолчанию: cuda, если доступна)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Файл изображения не найден по пути: {args.image}")

    result = full_inference_pipeline(image_path=args.image, device=args.device)

    print("\nФинальное предсказание:")
    print(f"Категория: {result['category']}")
    print(f"Подкатегория: {result['subcategory']}")
    print(f"Сгенерированное название: {result['title']}")
    print(f"Предсказанная цена: {result['price']:.2f} RUB")
