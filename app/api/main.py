from fastapi import FastAPI

from app.api.routes import documents, health, search

app = FastAPI(title="Multimodal Retrieval Service")
app.include_router(health.router)
app.include_router(search.router)
app.include_router(documents.router)
