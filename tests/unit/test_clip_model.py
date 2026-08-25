import pytest
from PIL import Image

from app.models.clip_model import CLIP_VECTOR_DIM, get_clip_embedder

pytestmark = pytest.mark.slow


def test_embed_image_and_text_same_dim_and_normalized():
    embedder = get_clip_embedder()
    image = Image.new("RGB", (224, 224), color="red")
    image_vec = embedder.embed_image(image)
    text_vec = embedder.embed_text("一张红色的图片")

    assert len(image_vec) == CLIP_VECTOR_DIM
    assert len(text_vec) == CLIP_VECTOR_DIM

    import math

    norm = math.sqrt(sum(v * v for v in image_vec))
    assert 0.99 < norm < 1.01


def test_get_clip_embedder_is_singleton():
    assert get_clip_embedder() is get_clip_embedder()
