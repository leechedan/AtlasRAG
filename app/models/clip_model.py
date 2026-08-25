from functools import lru_cache

import torch
from PIL import Image

from app.config import settings

CLIP_VECTOR_DIM = 512  # 若 Task 2 验证脚本打印的维度不同，改这里


class ClipEmbedder:
    def __init__(self, model_name: str):
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ChineseCLIPModel.from_pretrained(model_name).to(self._device).eval()
        self._processor = ChineseCLIPProcessor.from_pretrained(model_name)

    def embed_image(self, image: Image.Image) -> list[float]:
        with torch.no_grad():
            inputs = self._processor(images=image, return_tensors="pt").to(self._device)
            features = self._model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        with torch.no_grad():
            inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self._device)
            features = self._model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()


@lru_cache(maxsize=1)
def get_clip_embedder() -> ClipEmbedder:
    return ClipEmbedder(settings.clip_model_name)
