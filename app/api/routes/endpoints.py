import typing

from fastapi import APIRouter, Request
from fastapi.routing import APIRoute
from pydantic import BaseModel

from app.api.schemas import EndpointInfo

router = APIRouter()


def _response_fields(response_model: type | None) -> list[str] | None:
    """Best-effort field names of a route's declared response_model.

    Unwraps a single level of generic container (list[X]/set[X]/tuple[X])
    since most list-returning routes here declare response_model=list[X]
    rather than a dedicated page wrapper. Returns None (not []) when there's
    nothing to introspect -- a route with no response_model (e.g. /health's
    plain dict, or a FileResponse endpoint) genuinely has no declared shape,
    which is a different fact than "a model with zero fields".
    """
    if response_model is None:
        return None
    origin = typing.get_origin(response_model)
    model = typing.get_args(response_model)[0] if origin in (list, set, tuple) else response_model
    if isinstance(model, type) and issubclass(model, BaseModel):
        return list(model.model_fields.keys())
    return None


@router.get("/endpoints", response_model=list[EndpointInfo])
def list_endpoints(request: Request):
    # 自省 FastAPI 自己的路由表，而不是手写一份静态清单——手写清单会在新增/
    # 修改接口时悄悄过期(架构设计文档 docs/superpowers/specs/... 里那份手写
    # 的接口表格就曾经因为没跟着代码更新而过期)。isinstance 过滤掉 /docs、
    # /redoc、/openapi.json 这些 FastAPI 用 Starlette 原生 Route(不是
    # APIRoute)注册的内部路由，只保留业务路由。
    api_routes = [route for route in request.app.routes if isinstance(route, APIRoute)]
    return sorted(
        (
            EndpointInfo(
                path=route.path,
                methods=sorted(route.methods - {"HEAD"}),
                name=route.name,
                # route.summary is only ever non-None if a route explicitly
                # passed summary=... -- none of this codebase's routes do.
                # FastAPI's own /docs page falls back to a title-cased
                # function name in that case (see fastapi/openapi/utils.py),
                # but that fallback lives in the OpenAPI-generation code path
                # and is never written back onto the route object itself, so
                # reading route.summary directly (as an earlier version of
                # this endpoint did) returned null for every route. Mirror
                # the same fallback here so the two surfaces agree.
                summary=route.summary or route.name.replace("_", " ").title(),
                description=route.description or None,
                tags=list(route.tags),
                response_fields=_response_fields(route.response_model),
            )
            for route in api_routes
        ),
        key=lambda e: e.path,
    )
