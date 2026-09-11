"""进程内视频生成任务表（OpenAI /v1/videos 异步语义）。单实例代理够用；多实例需换外部存储。"""

import time
import uuid
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

VideoStatus = Literal["queued", "in_progress", "completed", "failed"]


@dataclass(frozen=True)
class VideoJob:
    id: str
    model: str
    prompt: str
    created_at: int
    status: VideoStatus = "queued"
    progress: int = 0
    completed_at: int | None = None
    error: dict[str, str] | None = None
    session: str | None = None
    file_id: str | None = None
    mime_type: str | None = None
    note: str = ""

    def to_openai(self) -> dict[str, Any]:
        d = asdict(self)
        # session/file_id 是内部下载凭据，不暴露给客户端
        for k in ("session", "file_id", "mime_type"):
            d.pop(k)
        d["object"] = "video"
        return d


class VideoJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, VideoJob] = {}

    def create(self, model: str, prompt: str) -> VideoJob:
        job = VideoJob(
            id=f"video_{uuid.uuid4().hex}", model=model, prompt=prompt, created_at=int(time.time())
        )
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> VideoJob | None:
        return self._jobs.get(job_id)

    def list(self) -> list[VideoJob]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def update(self, job_id: str, **changes: Any) -> VideoJob:
        job = replace(self._jobs[job_id], **changes)
        self._jobs[job_id] = job
        return job

    def clear(self) -> None:
        self._jobs.clear()
