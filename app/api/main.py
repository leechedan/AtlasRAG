from fastapi import FastAPI

from app.api.routes import health, search

app = FastAPI(title="Multimodal Retrieval Service")
app.include_router(health.router)
app.include_router(search.router)
