import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

logger = logging.getLogger(__name__)

_MAX_PENDING_BYTES = 1 << 20


async def parse_sse(response) -> AsyncGenerator[dict[str, Any], None]:
    """解析上游 SSE，丢弃无法解析的行而非塞回缓冲区，避免死循环。"""
    pending = ""
    async for raw in response.aiter_lines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            return

        candidate = pending + line
        try:
            yield json.loads(candidate)
            pending = ""
        except json.JSONDecodeError:
            pending = candidate
            if len(pending) > _MAX_PENDING_BYTES:
                logger.warning("Dropping unparsable SSE buffer (%d bytes)", len(pending))
                pending = ""


class ChunkFactory:
    """统一构造 OpenAI chat.completion.chunk SSE 帧。"""

    def __init__(self, model: str) -> None:
        self.req_id = f"chatcmpl-{uuid.uuid4()}"
        self.created = int(time.time())
        self.model = model

    def _frame(self, delta: dict, finish_reason: str | None) -> str:
        body = {
            "id": self.req_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    def text(self, content: str) -> str:
        return self._frame({"content": content}, None)

    def error(self, message: str) -> str:
        return self._frame({"content": f"Error: {message}"}, "stop")

    def stop(self) -> str:
        return self._frame({}, "stop")

    @staticmethod
    def done() -> str:
        return "data: [DONE]\n\n"
