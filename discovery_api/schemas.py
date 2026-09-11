from dataclasses import dataclass

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = 1.0
    max_tokens: int | None = None
    top_p: float | None = None


@dataclass(frozen=True)
class ModelRoute:
    backend: str  # "vertex" | "agent_search"
    with_search: bool


MODEL_ROUTES: dict[str, ModelRoute] = {
    "gemini-standard": ModelRoute(backend="vertex", with_search=False),
    "gemini-2.5-flash": ModelRoute(backend="vertex", with_search=False),
    "vertex-grounded-search": ModelRoute(backend="vertex", with_search=True),
    "discovery-standard": ModelRoute(backend="agent_search", with_search=False),
    "discovery-grounded-search": ModelRoute(backend="agent_search", with_search=True),
}


def convert_to_gcp_contents(
    messages: list[ChatMessage],
) -> tuple[list[dict], str | None]:
    """OpenAI messages -> GCP contents，分离 system 指令。"""
    contents: list[dict] = []
    system: str | None = None
    for msg in messages:
        if msg.role == "system":
            system = msg.content
        elif msg.role == "user":
            contents.append({"role": "user", "parts": [{"text": msg.content}]})
        elif msg.role in ("assistant", "model"):
            contents.append({"role": "model", "parts": [{"text": msg.content}]})
    return contents, system


# ---------- 媒体生成（Gemini Enterprise 生图 / 生视频） ----------


class ImageGenerationRequest(BaseModel):
    """OpenAI Images API 请求子集。GE 侧无 size/quality 等参数，仅透传 prompt。"""

    model: str
    prompt: str = Field(min_length=1)
    n: int = 1
    response_format: str = "b64_json"  # "b64_json" | "url"
    size: str | None = None  # 接受但忽略：GE ImageGenerationSpec 无参数


class VideoGenerationRequest(BaseModel):
    """OpenAI Videos API 请求子集。GE 侧时长/尺寸不可控。"""

    model: str
    prompt: str = Field(min_length=1)
    seconds: int | None = None  # 接受但忽略
    size: str | None = None  # 接受但忽略


IMAGE_MODELS: dict[str, str] = {"discovery-image": "agent_search"}
VIDEO_MODELS: dict[str, str] = {"discovery-video": "agent_search"}
