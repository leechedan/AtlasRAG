from fastapi import APIRouter, Request
from fastapi.routing import APIRoute

from app.api.schemas import EndpointInfo

router = APIRouter()


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
                summary=route.summary,
                description=route.description or None,
                tags=list(route.tags),
            )
            for route in api_routes
        ),
        key=lambda e: e.path,
    )
