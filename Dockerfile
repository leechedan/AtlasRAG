# Dockerfile
FROM python:3.12-slim

WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 用阿里云 PyPI 镜像加速依赖下载。
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

# torch install is its own layer, cached independently of requirements.txt --
# any future requirements.txt edit (e.g. adding a small pure-Python package)
# must not bust this layer and force a multi-GB torch+cudnn re-download.
# --index-url 指向 PyTorch 官方源以拿到 cu130 专属 wheel（阿里云镜像不一定同步这个
# 自定义 index）；--extra-index-url 让 torch 的普通依赖（sympy/jinja2/filelock 等）
# 走阿里云镜像加速。
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://mirrors.aliyun.com/pypi/simple/

# paddleocr 依赖的 opencv 需要 libgl1/libglib2.0-0，上面 apt 安装步骤已包含，
# 若未来升级 paddleocr 版本发现新的系统依赖缺失，在这里补充 apt 包。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
