import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from discovery_api.agent_search import agent_search_stream, agent_search_unary
from discovery_api.auth import get_gcp_access_token
from discovery_api.config import settings
from discovery_api.handlers import vertex_stream, vertex_unary
from discovery_api.schemas import MODEL_ROUTES, ChatCompletionRequest, convert_to_gcp_contents

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("discovery-api-proxy")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=60.0)
    yield
    await app.state.http.aclose()


app = FastAPI(title="GCP Discovery & Vertex AI OpenAI Proxy", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, token: str = Depends(get_gcp_access_token)):
    logger.info("Request model=%s stream=%s", req.model, req.stream)

    route = MODEL_ROUTES.get(req.model)
    if route is None:
        supported = ", ".join(MODEL_ROUTES)
        raise HTTPException(status_code=400, detail=f"Unsupported model: {req.model}. Supported: {supported}")

    contents, system = convert_to_gcp_contents(req.messages)
    client: httpx.AsyncClient = app.state.http

    if route.backend == "agent_search":
        if req.stream:
            return StreamingResponse(
                agent_search_stream(client, req.model, contents, system, token, route.with_search),
                media_type="text/event-stream",
            )
        return await agent_search_unary(client, req.model, contents, system, token, route.with_search)

    if req.stream:
        return StreamingResponse(
            vertex_stream(
                client, req.model, contents, system, req.temperature,
                req.max_tokens, token, route.with_search,
            ),
            media_type="text/event-stream",
        )
    return await vertex_unary(
        client, req.model, contents, system, req.temperature,
        req.max_tokens, token, route.with_search,
    )


@app.get("/health")
async def health_check():
    try:
        get_gcp_access_token()
        return {
            "status": "healthy",
            "gcp_project_id": settings.gcp_project_id,
            "gcp_project_number": settings.gcp_project_number,
            "credentials_resolved": True,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "gcp_project_id": settings.gcp_project_id,
            "credentials_resolved": False,
            "error": str(e),
        }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting OpenAI Proxy on %s:%s", settings.proxy_host, settings.proxy_port)
    uvicorn.run("app:app", host=settings.proxy_host, port=settings.proxy_port, reload=True)
