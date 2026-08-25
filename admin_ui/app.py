import io
import os

import gradio as gr
import httpx
from PIL import Image

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def _fetch_thumbnail(doc_id: str):
    """Fetch a hit's thumbnail through the API's GET /documents/{doc_id}/thumbnail
    (rather than reading source_path/thumbnail_path off the local filesystem):
    the admin-ui container has no volume mount onto the api/worker containers'
    /srv/data (see docker-compose.yml), so those paths never resolve here --
    HTTP is the only channel this process has to the actual image bytes.
    Returns None (silently skipped by callers) for hits with no image record
    or a missing thumbnail file, so one broken hit doesn't blank the page.
    """
    try:
        response = httpx.get(f"{API_BASE_URL}/documents/{doc_id}/thumbnail", timeout=10)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    return Image.open(io.BytesIO(response.content))


def _thumbnails_for_hits(hits: list[dict]):
    images = []
    for hit in hits:
        doc_id = hit["payload"].get("doc_id")
        if not doc_id:
            continue
        image = _fetch_thumbnail(doc_id)
        if image is not None:
            images.append(image)
    return images


def do_search_text(query: str, top_k: int, search_images_too: bool):
    modalities = ["text", "image"] if search_images_too else ["text"]
    response = httpx.post(
        f"{API_BASE_URL}/search/text", json={"query": query, "top_k": int(top_k), "modalities": modalities}, timeout=30
    )
    response.raise_for_status()
    body = response.json()
    text_lines = [f"[{h['score']:.3f}] {h['payload'].get('text', '')[:200]}" for h in body["text_hits"]]
    return "\n\n".join(text_lines) or "无结果", _thumbnails_for_hits(body["image_hits"])


def do_search_image(image_path: str, top_k: int):
    with open(image_path, "rb") as f:
        response = httpx.post(
            f"{API_BASE_URL}/search/image", files={"image": f}, params={"top_k": int(top_k)}, timeout=30
        )
    response.raise_for_status()
    body = response.json()
    return _thumbnails_for_hits(body["image_hits"])


def do_upload(file_path: str, chunk_size: int, chunk_overlap: int, separators_text: str):
    data = {}
    if chunk_size:
        data["chunk_size"] = str(int(chunk_size))
    if chunk_overlap:
        data["chunk_overlap"] = str(int(chunk_overlap))
    separators = [s for s in separators_text.split(",") if s] if separators_text else []
    if separators:
        # NOTE: must stay a dict with a list value here, not a flattened list of
        # (key, value) tuples. With httpx==0.28.1, passing `data=` as a list of
        # 2-tuples alongside `files=` causes httpx._content.encode_request to
        # treat `data` as raw request content instead of routing it through the
        # multipart encoder, which silently drops the `files` part entirely (no
        # exception -- just a DeprecationWarning -- and the server then 422s with
        # "file field required"). A dict whose value is a list is what httpx's
        # multipart encoder correctly expands into repeated same-name fields.
        # Task 17 hit and documented this exact issue in
        # tests/unit/test_api_documents.py.
        data["separators"] = separators

    with open(file_path, "rb") as f:
        response = httpx.post(
            f"{API_BASE_URL}/documents/upload",
            files={"file": f},
            data=data,
            timeout=60,
        )
    response.raise_for_status()
    return f"已提交，job_id={response.json()['job_id']}"


def do_index_directory(path: str, recursive: bool, watch: bool, chunk_size: int, chunk_overlap: int, separators_text: str):
    separators = [s for s in separators_text.split(",") if s] if separators_text else None
    response = httpx.post(
        f"{API_BASE_URL}/directories/index",
        json={
            "path": path, "recursive": recursive, "watch": watch,
            "chunk_size": int(chunk_size) if chunk_size else None,
            "chunk_overlap": int(chunk_overlap) if chunk_overlap else None,
            "separators": separators,
        },
        timeout=30,
    )
    response.raise_for_status()
    return f"已提交，job_id={response.json()['job_id']}"


def do_check_job(job_id: str):
    response = httpx.get(f"{API_BASE_URL}/jobs/{job_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def do_stats():
    response = httpx.get(f"{API_BASE_URL}/collections/stats", timeout=30)
    response.raise_for_status()
    return response.json()


with gr.Blocks(title="检索服务管理界面") as demo:
    gr.Markdown("# 多模态检索服务 — 管理/测试界面")

    with gr.Tab("文字检索"):
        query_input = gr.Textbox(label="查询文字")
        top_k_input = gr.Slider(1, 50, value=10, step=1, label="返回条数")
        search_image_checkbox = gr.Checkbox(label="同时跨模态检索图片")
        search_button = gr.Button("检索")
        text_result = gr.Textbox(label="文字结果", lines=10)
        image_result = gr.Gallery(label="图片结果", columns=4)
        search_button.click(do_search_text, [query_input, top_k_input, search_image_checkbox], [text_result, image_result])

    with gr.Tab("图片检索"):
        query_image = gr.Image(type="filepath", label="上传查询图片")
        image_top_k = gr.Slider(1, 50, value=10, step=1, label="返回条数")
        image_search_button = gr.Button("检索")
        image_search_result = gr.Gallery(label="结果", columns=4)
        image_search_button.click(do_search_image, [query_image, image_top_k], image_search_result)

    with gr.Tab("上传文档"):
        upload_file = gr.File(label="选择文件 (csv/txt/pdf/jpg/png/md)", type="filepath")
        with gr.Accordion("切分参数（不填用全局默认）", open=False):
            upload_chunk_size = gr.Number(label="chunk_size（字符数）", value=None)
            upload_chunk_overlap = gr.Number(label="chunk_overlap（字符数）", value=None)
            upload_separators = gr.Textbox(label="自定义分隔符（逗号分隔，如 \\n\\n,---）")
        upload_button = gr.Button("上传入库")
        upload_result = gr.Textbox(label="结果")
        upload_button.click(
            do_upload,
            [upload_file, upload_chunk_size, upload_chunk_overlap, upload_separators],
            upload_result,
        )

    with gr.Tab("目录入库"):
        dir_path_input = gr.Textbox(label="本地目录路径")
        recursive_checkbox = gr.Checkbox(label="递归子目录", value=True)
        watch_checkbox = gr.Checkbox(label="持续监听")
        with gr.Accordion("切分参数（不填用全局默认，watch=True 时会跟随目录持久化）", open=False):
            dir_chunk_size = gr.Number(label="chunk_size（字符数）", value=None)
            dir_chunk_overlap = gr.Number(label="chunk_overlap（字符数）", value=None)
            dir_separators = gr.Textbox(label="自定义分隔符（逗号分隔，如 \\n\\n,---）")
        dir_index_button = gr.Button("开始入库")
        dir_index_result = gr.Textbox(label="结果")
        dir_index_button.click(
            do_index_directory,
            [dir_path_input, recursive_checkbox, watch_checkbox, dir_chunk_size, dir_chunk_overlap, dir_separators],
            dir_index_result,
        )

    with gr.Tab("任务状态"):
        job_id_input = gr.Textbox(label="Job ID")
        job_check_button = gr.Button("查询")
        job_status_output = gr.JSON(label="状态")
        job_check_button.click(do_check_job, job_id_input, job_status_output)

    with gr.Tab("索引库状态"):
        stats_button = gr.Button("刷新")
        stats_output = gr.JSON(label="统计")
        stats_button.click(do_stats, None, stats_output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
