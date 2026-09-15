import io
import os
import tempfile

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


def do_convert_to_markdown(file_path: str):
    with open(file_path, "rb") as f:
        # timeout matches settings.mineru_timeout_seconds (600s, pdf/docx/pptx/xlsx
        # route through MinerU synchronously here) plus a margin.
        response = httpx.post(f"{API_BASE_URL}/documents/convert-to-markdown", files={"file": f}, timeout=650)
    response.raise_for_status()
    return response.text, response.headers.get("X-Images-Zip-Url")


def do_download_converted_images(zip_url: str | None):
    if not zip_url:
        return None
    # zip_url is the API-relative path from the X-Images-Zip-Url header; fetched
    # through API_BASE_URL (container-internal) same as _fetch_thumbnail above,
    # then handed to gr.File as a local path so the browser can download it --
    # admin-ui has no volume mount onto the api container's disk.
    response = httpx.get(f"{API_BASE_URL}{zip_url}", timeout=60)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(response.content)
        return tmp.name


def do_mineru_idle_status():
    response = httpx.get(f"{API_BASE_URL}/health/mineru-idle", timeout=10)
    response.raise_for_status()
    return response.json()


def do_list_documents():
    response = httpx.get(f"{API_BASE_URL}/documents", timeout=30)
    response.raise_for_status()
    docs = response.json()
    return [
        [d["doc_id"], d["filename"], d["file_type"], d["origin"], d["chunk_count"], ", ".join(d["source_types"])]
        for d in docs
    ]


def do_select_document(documents_table, evt: gr.SelectData):
    # documents_table is the Dataframe's current value (type="array" ->
    # list[list]); evt.index[0] is the clicked row, column 0 is the full
    # doc_id (kept untruncated in that column specifically so this lookup
    # is unambiguous -- a shortened id could collide or fail to match).
    row_index = evt.index[0]
    doc_id = documents_table[row_index][0]
    response = httpx.get(f"{API_BASE_URL}/documents/{doc_id}/chunks", timeout=30)
    response.raise_for_status()
    chunks = response.json()
    rows = [
        [c["chunk_index"], c.get("heading_path") or "", c["page"], c["source_type"], c["chunk_id"], c["text"]]
        for c in chunks
    ]
    return rows, doc_id


def do_delete_document(doc_id: str | None):
    if not doc_id:
        return "请先在文档列表中选择一行", gr.update(), gr.update(), gr.update()
    response = httpx.delete(f"{API_BASE_URL}/documents/{doc_id}", timeout=30)
    response.raise_for_status()
    return (
        f"已提交删除任务 {doc_id}，处理是异步的，请稍后点击「刷新文档列表」确认",
        gr.update(),
        [],
        None,
    )


def do_list_jobs():
    response = httpx.get(f"{API_BASE_URL}/jobs", timeout=30)
    response.raise_for_status()
    jobs = response.json()
    return [
        [j["job_id"], j["total"], j["processed"], j["success_count"], j["failed_count"]]
        for j in jobs
    ]


def do_select_job(jobs_table, evt: gr.SelectData):
    # 合并原来独立的「任务状态」查询 tab：点行除了记住 job_id（供删除按钮用），
    # 顺带把该任务的完整状态（do_check_job，即 GET /jobs/{job_id}）也查出来
    # 显示，不用再切到另一个 tab 手动输入 job_id 查一遍。
    row_index = evt.index[0]
    job_id = jobs_table[row_index][0]
    status = do_check_job(job_id)
    return job_id, status


def do_delete_job(job_id: str | None):
    if not job_id:
        return "请先在任务记录列表中选择一行", gr.update(), gr.update(), None
    response = httpx.delete(f"{API_BASE_URL}/jobs/{job_id}", timeout=30)
    response.raise_for_status()
    return f"已删除任务记录 {job_id}", do_list_jobs(), None, None


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
        upload_file = gr.File(label="选择文件 (csv/txt/pdf/jpg/png/md/docx/pptx/xlsx)", type="filepath")
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

    with gr.Tab("文件转 Markdown"):
        gr.Markdown(
            "上传单个文件，同步转换为 Markdown 文本并直接返回（不入库、不生成任务）。"
            "支持 txt/md/csv/jpg/png/docx/pptx/xlsx/pdf。含图片的文档可另外下载图片包。"
        )
        convert_file = gr.File(label="选择文件", type="filepath")
        convert_button = gr.Button("转换")
        convert_result = gr.Code(label="Markdown 结果", language="markdown", lines=20)
        images_zip_url = gr.State(None)
        download_images_button = gr.Button("下载图片包（如有）")
        download_images_file = gr.File(label="图片包")
        convert_button.click(do_convert_to_markdown, convert_file, [convert_result, images_zip_url])
        download_images_button.click(do_download_converted_images, images_zip_url, download_images_file)

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

    with gr.Tab("索引库状态"):
        stats_button = gr.Button("刷新")
        stats_output = gr.JSON(label="统计")
        stats_button.click(do_stats, None, stats_output)

    with gr.Tab("MinerU 空闲状态"):
        gr.Markdown("重启 mineru 服务前用来判断是否空闲——两个信号（在途请求计数器、距上次转换的时间）都显示空闲才建议重启。")
        mineru_idle_button = gr.Button("检查")
        mineru_idle_output = gr.JSON(label="空闲状态")
        mineru_idle_button.click(do_mineru_idle_status, None, mineru_idle_output)

    with gr.Tab("文档管理"):
        refresh_docs_button = gr.Button("刷新文档列表")
        documents_table = gr.Dataframe(
            headers=["doc_id", "文件名", "类型", "来源", "分片数", "内容类型"],
            type="array",
            interactive=False,
            label="文档列表（点击一行查看该文档的分片）",
        )
        selected_doc_id = gr.State(None)
        delete_doc_button = gr.Button("删除所选文档")
        delete_doc_result = gr.Textbox(label="删除结果")
        chunks_table = gr.Dataframe(
            headers=["序号", "标题路径", "页码", "内容类型", "chunk_id", "文本"],
            type="array",
            interactive=False,
            wrap=True,
            label="分片内容",
        )
        refresh_docs_button.click(do_list_documents, None, documents_table)
        documents_table.select(do_select_document, documents_table, [chunks_table, selected_doc_id])
        delete_doc_button.click(do_delete_document, selected_doc_id, [delete_doc_result, documents_table, chunks_table, selected_doc_id])
        demo.load(do_list_documents, None, documents_table)

    with gr.Tab("任务记录"):
        refresh_jobs_button = gr.Button("刷新任务记录")
        jobs_table = gr.Dataframe(
            headers=["job_id", "总数", "已处理", "成功数", "失败数"],
            type="array",
            interactive=False,
            label="任务记录（点击一行查看详情，用下方按钮删除）",
        )
        selected_job_id = gr.State(None)
        job_detail_output = gr.JSON(label="任务详情")
        delete_job_button = gr.Button("删除所选任务记录")
        delete_job_result = gr.Textbox(label="删除结果")
        refresh_jobs_button.click(do_list_jobs, None, jobs_table)
        jobs_table.select(do_select_job, jobs_table, [selected_job_id, job_detail_output])
        delete_job_button.click(
            do_delete_job, selected_job_id, [delete_job_result, jobs_table, selected_job_id, job_detail_output]
        )
        demo.load(do_list_jobs, None, jobs_table)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
