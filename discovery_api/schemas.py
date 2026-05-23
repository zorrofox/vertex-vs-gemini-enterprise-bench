from dataclasses import dataclass

from pydantic import BaseModel


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
