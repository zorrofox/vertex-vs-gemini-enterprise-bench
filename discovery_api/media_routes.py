"""OpenAI 形状的媒体生成路由，后端为 Gemini Enterprise :streamAssist。

- POST /v1/images/generations      同步，兼容 OpenAI Images API（b64_json / url）
- POST /v1/videos                  异步 job，兼容 OpenAI Videos API
- GET  /v1/videos[/{id}[/content]] 轮询 / 下载
- GET  /v1/media/{session_id}/{file_id}  代理下载 GE session 文件（images url 模式使用）
"""

import asyncio
import base64
import logging
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from .auth import get_gcp_access_token
from .media import download_session_file, generate_media, session_resource_name
from .schemas import IMAGE_MODELS, VIDEO_MODELS, ImageGenerationRequest, VideoGenerationRequest
from .video_jobs import VideoJobStore

logger = logging.getLogger(__name__)
router = APIRouter()
video_jobs = VideoJobStore()
_background: set[asyncio.Task] = set()


def _http(request: Request) -> httpx.AsyncClient:
    return request.app.state.http


def _check_model(model: str, allowed: dict[str, str], what: str) -> None:
    if model not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported {what} model: {model}. Supported: {', '.join(allowed)}",
        )


# ---------- images ----------


@router.post("/v1/images/generations")
async def images_generations(
    req: ImageGenerationRequest, request: Request, token: str = Depends(get_gcp_access_token)
):
    _check_model(req.model, IMAGE_MODELS, "image")
    if req.n != 1:
        raise HTTPException(
            status_code=400, detail="Gemini Enterprise generates one image per request (n=1)."
        )
    logger.info("Image generation model=%s format=%s", req.model, req.response_format)

    client = _http(request)
    result = await generate_media(client, req.prompt, "image", token)
    data = []
    for f in result.files:
        item: dict = {"revised_prompt": result.text}
        if req.response_format == "url":
            item["url"] = str(
                request.url_for("media_download", session_id=result.session_id, file_id=f.file_id)
            )
        else:
            content, _ = await download_session_file(client, result.session, f.file_id, token)
            item["b64_json"] = base64.b64encode(content).decode()
        data.append(item)
    return {"created": int(time.time()), "data": data}


@router.get("/v1/media/{session_id}/{file_id}", name="media_download")
async def media_download(
    session_id: str, file_id: str, request: Request, token: str = Depends(get_gcp_access_token)
):
    content, mime = await download_session_file(
        _http(request), session_resource_name(session_id), file_id, token
    )
    return Response(content=content, media_type=mime)


# ---------- videos ----------


async def _run_video_job(client: httpx.AsyncClient, job_id: str, prompt: str, token: str) -> None:
    video_jobs.update(job_id, status="in_progress", progress=10)
    try:
        result = await generate_media(client, prompt, "video", token)
        f = result.files[0]
        video_jobs.update(
            job_id,
            status="completed",
            progress=100,
            completed_at=int(time.time()),
            session=result.session,
            file_id=f.file_id,
            mime_type=f.mime_type,
            note=result.text,
        )
    except HTTPException as e:
        video_jobs.update(
            job_id,
            status="failed",
            completed_at=int(time.time()),
            error={"code": str(e.status_code), "message": str(e.detail)},
        )
    except Exception as e:  # noqa: BLE001 - 任务失败必须落到 job 状态，不能静默
        logger.exception("Video job %s crashed", job_id)
        video_jobs.update(
            job_id,
            status="failed",
            completed_at=int(time.time()),
            error={"code": "internal_error", "message": str(e)},
        )


@router.post("/v1/videos", status_code=status.HTTP_202_ACCEPTED)
async def videos_create(
    req: VideoGenerationRequest, request: Request, token: str = Depends(get_gcp_access_token)
):
    _check_model(req.model, VIDEO_MODELS, "video")
    job = video_jobs.create(req.model, req.prompt)
    logger.info("Video job %s created model=%s", job.id, req.model)
    task = asyncio.create_task(_run_video_job(_http(request), job.id, req.prompt, token))
    _background.add(task)
    task.add_done_callback(_background.discard)
    return job.to_openai()


@router.get("/v1/videos")
async def videos_list():
    return {"object": "list", "data": [j.to_openai() for j in video_jobs.list()]}


@router.get("/v1/videos/{video_id}")
async def videos_get(video_id: str):
    job = video_jobs.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    return job.to_openai()


@router.get("/v1/videos/{video_id}/content")
async def videos_content(
    video_id: str, request: Request, token: str = Depends(get_gcp_access_token)
):
    job = video_jobs.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    if job.status != "completed" or not job.session or not job.file_id:
        raise HTTPException(
            status_code=409,
            detail=f"Video {video_id} is {job.status}; content is only available once completed.",
        )
    content, mime = await download_session_file(_http(request), job.session, job.file_id, token)
    return Response(content=content, media_type=mime or job.mime_type)
