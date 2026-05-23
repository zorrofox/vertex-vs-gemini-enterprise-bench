import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import HTTPException

from .citations import extract_grounding_citations, format_citations_markdown
from .config import settings
from .sse import ChunkFactory, parse_sse

logger = logging.getLogger(__name__)


def _vertex_url(stream: bool) -> str:
    method = "streamGenerateContent?alt=sse" if stream else "generateContent"
    return (
        f"https://{settings.vertex_location}-aiplatform.googleapis.com/v1/"
        f"projects/{settings.gcp_project_id}/locations/{settings.vertex_location}/"
        f"publishers/google/models/{settings.gemini_model}:{method}"
    )


def _build_payload(
    contents: list[dict],
    system: str | None,
    temperature: float,
    max_tokens: int | None,
    with_search: bool,
) -> dict[str, Any]:
    gen_config: dict[str, Any] = {"temperature": temperature}
    if max_tokens:
        gen_config["maxOutputTokens"] = max_tokens
    payload: dict[str, Any] = {"contents": contents, "generationConfig": gen_config}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if with_search:
        payload["tools"] = [{"google_search": {}}]
    return payload


def _extract_text(candidate: dict) -> str:
    parts = (candidate.get("content") or {}).get("parts") or []
    return parts[0].get("text", "") if parts else ""


async def vertex_stream(
    client: httpx.AsyncClient,
    model: str,
    contents: list[dict],
    system: str | None,
    temperature: float,
    max_tokens: int | None,
    token: str,
    with_search: bool,
) -> AsyncGenerator[str, None]:
    chunk = ChunkFactory(model)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = _build_payload(contents, system, temperature, max_tokens, with_search)
    citations = []

    try:
        async with client.stream("POST", _vertex_url(stream=True), headers=headers, json=payload) as r:
            if r.status_code != 200:
                err = (await r.aread()).decode(errors="replace")
                logger.error("Vertex API %s: %s", r.status_code, err)
                yield chunk.error(f"GCP Vertex API returned {r.status_code}: {err}")
                yield chunk.done()
                return

            async for data in parse_sse(r):
                cands = data.get("candidates") or []
                if not cands:
                    continue
                cand = cands[0]
                if meta := cand.get("groundingMetadata"):
                    if found := extract_grounding_citations(meta):
                        citations = found
                if text := _extract_text(cand):
                    yield chunk.text(text)

        if citations:
            yield chunk.text(format_citations_markdown(citations))
        yield chunk.stop()
        yield chunk.done()

    except Exception as e:
        logger.exception("Exception in vertex_stream")
        yield chunk.error(f"Internal Proxy Error: {e}")
        yield chunk.done()


async def vertex_unary(
    client: httpx.AsyncClient,
    model: str,
    contents: list[dict],
    system: str | None,
    temperature: float,
    max_tokens: int | None,
    token: str,
    with_search: bool,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = _build_payload(contents, system, temperature, max_tokens, with_search)

    res = await client.post(_vertex_url(stream=False), headers=headers, json=payload)
    if res.status_code != 200:
        raise HTTPException(status_code=res.status_code, detail=f"Vertex AI API error: {res.text}")

    data = res.json()
    cands = data.get("candidates") or []
    if not cands:
        raise HTTPException(status_code=502, detail="Empty response from GCP Vertex AI")

    cand = cands[0]
    text = _extract_text(cand)
    if cites := extract_grounding_citations(cand.get("groundingMetadata")):
        text += format_citations_markdown(cites)

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
