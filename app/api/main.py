from fastapi import FastAPI

from app.api.routes import health

app = FastAPI(title="Multimodal Retrieval Service")
app.include_router(health.router)
