# admin_ui/Dockerfile
FROM python:3.12-slim

WORKDIR /srv

# 用阿里云 PyPI 镜像加速依赖下载。
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

RUN pip install --no-cache-dir gradio==5.9.1 httpx==0.28.1 pillow==11.0.0 requests==2.34.2 \
    fastapi==0.115.6 pydantic==2.10.4

COPY admin_ui/app.py ./app.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
