# Dockerfile
FROM python:3.12-slim

WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# torch install is its own layer, cached independently of requirements.txt --
# any future requirements.txt edit (e.g. adding a small pure-Python package)
# must not bust this layer and force a multi-GB torch+cudnn re-download.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
