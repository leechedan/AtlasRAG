import httpx
import respx

from app.sdk.langgraph_tools import search_image_tool, search_text_tool


@respx.mock
def test_search_text_tool_calls_api_and_returns_hits():
    respx.post("http://localhost:8000/search/text").mock(
        return_value=httpx.Response(
            200, json={"text_hits": [{"score": 0.9, "payload": {"text": "结果A"}}], "image_hits": []}
        )
    )

    result = search_text_tool.invoke({"query": "测试查询", "top_k": 5})

    assert "结果A" in result


@respx.mock
def test_search_image_tool_calls_api_with_image_path(tmp_path):
    from PIL import Image

    img_path = tmp_path / "q.png"
    Image.new("RGB", (10, 10)).save(img_path)

    respx.post("http://localhost:8000/search/image").mock(
        return_value=httpx.Response(
            200, json={"image_hits": [{"score": 0.8, "payload": {"source_path": "/a.png"}}], "related_text_hits": []}
        )
    )

    result = search_image_tool.invoke({"image_path": str(img_path), "top_k": 5})

    assert "/a.png" in result
