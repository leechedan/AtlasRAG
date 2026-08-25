# admin_ui/Dockerfile
FROM python:3.12-slim

WORKDIR /srv

RUN pip install --no-cache-dir gradio==5.9.1 httpx==0.28.1 pillow==11.0.0 requests==2.34.2 \
    fastapi==0.115.6 pydantic==2.10.4

COPY admin_ui/app.py ./app.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
