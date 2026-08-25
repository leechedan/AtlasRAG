from fastapi import APIRouter, Depends

from app.api.deps import get_store_client
from app.api.schemas import CollectionsStatsResponse
from app.store.qdrant_client import get_stats

router = APIRouter()


@router.get("/collections/stats", response_model=CollectionsStatsResponse)
def collections_stats(store=Depends(get_store_client)):
    stats = get_stats(store)
    return CollectionsStatsResponse(**stats)
