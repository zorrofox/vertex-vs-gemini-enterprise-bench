"""Gemini Enterprise (Agentspace) Assistant API handler — :streamAssist / :assist."""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import HTTPException

from .citations import Citation, _dedup, format_citations_markdown
from .config import settings
from .sse import ChunkFactory

logger = logging.getLogger(__name__)


def _assist_url(stream: bool) -> str:
    if not settings.agent_search_engine:
        raise HTTPException(
            status_code=500,
            detail="AGENT_SEARCH_ENGINE is not configured. Set it in .env "
            "(run `python scripts/list_engines.py` to discover available engines).",
        )
    method = "streamAssist" if stream else "assist"
    return (
        f"https://discoveryengine.googleapis.com/v1alpha/projects/{settings.gcp_project_id}"
        f"/locations/{settings.discovery_location}/collections/default_collection"
        f"/engines/{settings.agent_search_engine}"
        f"/assistants/{settings.agent_search_assistant}:{method}"
    )


def _extract_last_user_text(contents: list[dict]) -> str:
    for c in reversed(contents):
        if c.get("role") == "user":
            parts = c.get("parts") or []
            if parts:
                return parts[0].get("text", "")
    return ""


def _build_payload(contents: list[dict], system: str | None, with_search: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": {"text": _extract_last_user_text(contents)}}
    tools: dict[str, Any] = {"webGroundingSpec": {}} if with_search else {}
    if tools:
        payload["toolsSpec"] = tools
    if not with_search:
        payload["googleSearchGroundingEnabled"] = False
    if settings.agent_search_model_version:
        payload["generationSpec"] = {"modelId": settings.agent_search_model_version}
    if system:
        payload.setdefault("generationSpec", {})["systemInstruction"] = {
            "parts": [{"text": system}]
        }
    return payload


def _extract_reply(answer: dict[str, Any]) -> tuple[str, list[Citation]]:
    """从单个 streamAssist answer 对象提取增量文本与引用。"""
    text_parts: list[str] = []
    cites: list[Citation] = []
    for reply in answer.get("replies") or []:
        gc = reply.get("groundedContent") or {}
        content = gc.get("content") or {}
        if t := content.get("text"):
            text_parts.append(t)
        meta = gc.get("groundingMetadata") or gc.get("textGroundingMetadata") or {}
        for ref in meta.get("references") or []:
            doc = ref.get("documentMetadata") or {}
            if uri := (doc.get("uri") or doc.get("url")):
                cites.append((doc.get("title", "Source"), uri))
        for chunk in meta.get("groundingChunks") or []:
            web = chunk.get("web") or {}
            if uri := web.get("uri"):
                cites.append((web.get("title", "Source"), uri))
    return "".join(text_parts), cites


async def _iter_json_objects(r: httpx.Response) -> AsyncGenerator[dict, None]:
    """流式响应是裸 JSON array：按花括号深度切分顶层对象。"""
    buf: list[str] = []
    depth = 0
    in_str = False
    esc = False
    async for piece in r.aiter_text():
        for ch in piece:
            if esc:
                buf.append(ch)
                esc = False
                continue
            if ch == "\\":
                buf.append(ch)
                esc = True
                continue
            if ch == '"':
                buf.append(ch)
                in_str = not in_str
                continue
            if in_str:
                buf.append(ch)
                continue
            if ch == "{":
                if depth == 0:
                    buf = []
                buf.append(ch)
                depth += 1
            elif ch == "}":
                buf.append(ch)
                depth -= 1
                if depth == 0:
                    snippet = "".join(buf)
                    try:
                        yield json.loads(snippet)
                    except json.JSONDecodeError as e:
                        logger.warning("Drop unparsable streamAssist chunk: %s", e)
                    buf = []
            elif depth > 0:
                buf.append(ch)


async def agent_search_stream(
    client: httpx.AsyncClient,
    model: str,
    contents: list[dict],
    system: str | None,
    token: str,
    with_search: bool,
) -> AsyncGenerator[str, None]:
    chunk = ChunkFactory(model)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Goog-User-Project": settings.gcp_project_id,
    }
    payload = _build_payload(contents, system, with_search)
    citations: list[Citation] = []
    emitted = False

    try:
        async with client.stream("POST", _assist_url(stream=True), headers=headers, json=payload) as r:
            if r.status_code != 200:
                err = (await r.aread()).decode(errors="replace")
                logger.error("GE Assist API %s: %s", r.status_code, err)
                yield chunk.error(f"GE Assist API {r.status_code}: {err}")
                yield chunk.done()
                return

            async for obj in _iter_json_objects(r):
                answer = obj.get("answer") or {}
                text, found = _extract_reply(answer)
                if found:
                    citations.extend(found)
                if text:
                    emitted = True
                    yield chunk.text(text)

        if not emitted:
            yield chunk.text("(Gemini Enterprise Assistant returned no content.)")
        if citations:
            yield chunk.text(format_citations_markdown(_dedup(citations)))
        yield chunk.stop()
        yield chunk.done()

    except Exception as e:
        logger.exception("Exception in agent_search_stream")
        yield chunk.error(f"Internal Proxy Error: {e}")
        yield chunk.done()


async def agent_search_unary(
    client: httpx.AsyncClient,
    model: str,
    contents: list[dict],
    system: str | None,
    token: str,
    with_search: bool,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Goog-User-Project": settings.gcp_project_id,
    }
    res = await client.post(
        _assist_url(stream=False), headers=headers, json=_build_payload(contents, system, with_search)
    )
    if res.status_code != 200:
        raise HTTPException(status_code=res.status_code, detail=f"GE Assist API error: {res.text}")

    answer = (res.json().get("answer") or {})
    text, cites = _extract_reply(answer)
    if cites:
        text += format_citations_markdown(_dedup(cites))

    factory = ChunkFactory(model)
    return {
        "id": factory.req_id,
        "object": "chat.completion",
        "created": factory.created,
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
    }
