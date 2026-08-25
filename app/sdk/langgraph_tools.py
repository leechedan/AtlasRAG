import os

import httpx
from langchain_core.tools import tool

API_BASE_URL = os.environ.get("RETRIEVAL_API_BASE_URL", "http://localhost:8000")


@tool
def search_text_tool(query: str, top_k: int = 5) -> str:
    """在检索服务里做文字检索，返回最相关的文本片段列表（用于回答需要引用本地文档知识的问题）。"""
    response = httpx.post(f"{API_BASE_URL}/search/text", json={"query": query, "top_k": top_k}, timeout=30)
    response.raise_for_status()
    hits = response.json()["text_hits"]
    if not hits:
        return "未找到相关内容"
    return "\n\n".join(f"[score={h['score']:.3f}] {h['payload'].get('text', '')}" for h in hits)


@tool
def search_image_tool(image_path: str, top_k: int = 5) -> str:
    """上传一张本地图片路径，在检索服务里做以图搜图，返回最相似图片的路径列表。"""
    with open(image_path, "rb") as f:
        response = httpx.post(f"{API_BASE_URL}/search/image", files={"image": f}, params={"top_k": top_k}, timeout=30)
    response.raise_for_status()
    hits = response.json()["image_hits"]
    if not hits:
        return "未找到相关图片"
    return "\n".join(f"[score={h['score']:.3f}] {h['payload'].get('source_path', '')}" for h in hits)
