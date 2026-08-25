from fastapi import FastAPI

from app.api.routes import collections, directories, documents, health, jobs, search

app = FastAPI(title="Multimodal Retrieval Service")
app.include_router(health.router)
app.include_router(search.router)
app.include_router(documents.router)
app.include_router(directories.router)
app.include_router(jobs.router)
app.include_router(collections.router)
