"""OpenAI 形状的 images / videos 路由测试（mock 掉 GE 调用）。"""

import base64
import time

import pytest
from fastapi.testclient import TestClient

import app as app_module
from discovery_api import media, media_routes
from discovery_api.auth import get_gcp_access_token

SESSION = media.session_resource_name("777")
PNG = b"\x89PNG\r\n\x1a\nfake"
MP4 = b"\x00\x00\x00\x18ftypmp42fake"


@pytest.fixture
def client(monkeypatch):
    async def fake_generate(_client, prompt, kind, _token):
        mime = "image/png" if kind == "image" else "video/mp4"
        return media.MediaResult(
            session=SESSION,
            files=[media.MediaFile(file_id=f"{kind}-1", mime_type=mime)],
            text=f"generated {kind} for {prompt}",
        )

    async def fake_download(_client, session, file_id, _token):
        assert session == SESSION
        return (
            PNG if file_id.startswith("image") else MP4,
            "image/png" if file_id.startswith("image") else "video/mp4",
        )

    monkeypatch.setattr(media_routes, "generate_media", fake_generate)
    monkeypatch.setattr(media_routes, "download_session_file", fake_download)
    media_routes.video_jobs.clear()
    app_module.app.dependency_overrides[get_gcp_access_token] = lambda: "tok"
    with TestClient(app_module.app) as c:
        yield c
    app_module.app.dependency_overrides.clear()


# ---------- images ----------


def test_images_b64_default(client):
    r = client.post(
        "/v1/images/generations", json={"model": "discovery-image", "prompt": "a red bicycle"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["created"], int)
    assert len(body["data"]) == 1
    assert base64.b64decode(body["data"][0]["b64_json"]) == PNG
    assert body["data"][0]["revised_prompt"] == "generated image for a red bicycle"


def test_images_url_points_to_proxy_download(client):
    r = client.post(
        "/v1/images/generations",
        json={"model": "discovery-image", "prompt": "a red bicycle", "response_format": "url"},
    )
    assert r.status_code == 200, r.text
    url = r.json()["data"][0]["url"]
    assert url.endswith("/v1/media/777/image-1")
    dl = client.get(url)
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("image/png")
    assert dl.content == PNG


def test_images_unsupported_model(client):
    r = client.post("/v1/images/generations", json={"model": "dall-e-3", "prompt": "x"})
    assert r.status_code == 400
    assert "discovery-image" in r.json()["detail"]


def test_images_rejects_n_gt_1(client):
    r = client.post(
        "/v1/images/generations", json={"model": "discovery-image", "prompt": "x", "n": 2}
    )
    assert r.status_code == 400


def test_images_upstream_error_passthrough(client, monkeypatch):
    from fastapi import HTTPException

    async def boom(*_a, **_k):
        raise HTTPException(status_code=403, detail="GE Assist API error: denied")

    monkeypatch.setattr(media_routes, "generate_media", boom)
    r = client.post("/v1/images/generations", json={"model": "discovery-image", "prompt": "x"})
    assert r.status_code == 403
    assert "denied" in r.json()["detail"]


# ---------- videos ----------


def _wait_terminal(client, vid: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/v1/videos/{vid}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"video {vid} did not finish: {body}")


def test_videos_job_lifecycle(client):
    r = client.post("/v1/videos", json={"model": "discovery-video", "prompt": "ocean waves"})
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["object"] == "video"
    assert job["id"].startswith("video_")
    assert job["status"] in ("queued", "in_progress")
    assert job["model"] == "discovery-video"

    done = _wait_terminal(client, job["id"])
    assert done["status"] == "completed", done
    assert done["progress"] == 100
    assert done["error"] is None

    dl = client.get(f"/v1/videos/{job['id']}/content")
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("video/mp4")
    assert dl.content == MP4


def test_videos_list_and_get_unknown(client):
    assert client.get("/v1/videos/video_nope").status_code == 404
    assert client.get("/v1/videos/video_nope/content").status_code == 404
    client.post("/v1/videos", json={"model": "discovery-video", "prompt": "a"})
    lst = client.get("/v1/videos").json()
    assert lst["object"] == "list" and len(lst["data"]) == 1


def test_videos_content_before_completion_is_409(client, monkeypatch):
    import asyncio

    async def slow(*_a, **_k):
        await asyncio.sleep(10)

    monkeypatch.setattr(media_routes, "generate_media", slow)
    job = client.post("/v1/videos", json={"model": "discovery-video", "prompt": "a"}).json()
    r = client.get(f"/v1/videos/{job['id']}/content")
    assert r.status_code == 409


def test_videos_failure_recorded(client, monkeypatch):
    from fastapi import HTTPException

    async def boom(*_a, **_k):
        raise HTTPException(status_code=502, detail="no file")

    monkeypatch.setattr(media_routes, "generate_media", boom)
    job = client.post("/v1/videos", json={"model": "discovery-video", "prompt": "a"}).json()
    done = _wait_terminal(client, job["id"])
    assert done["status"] == "failed"
    assert done["error"]["message"] == "no file"
    assert client.get(f"/v1/videos/{job['id']}/content").status_code == 409


def test_videos_unsupported_model(client):
    r = client.post("/v1/videos", json={"model": "sora-2", "prompt": "x"})
    assert r.status_code == 400
