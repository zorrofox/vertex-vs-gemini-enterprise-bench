import json
import logging
import os
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass
class Settings:
    gcp_project_id: str = field(default_factory=lambda: os.getenv("GCP_PROJECT_ID", ""))
    gcp_project_number: str = field(default_factory=lambda: os.getenv("GCP_PROJECT_NUMBER", ""))
    vertex_location: str = field(default_factory=lambda: os.getenv("VERTEX_LOCATION", "us-central1"))
    discovery_location: str = field(default_factory=lambda: os.getenv("DISCOVERY_LOCATION", "global"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    # Gemini Enterprise (Agentspace) App
    agent_search_engine: str = field(
        default_factory=lambda: os.getenv("AGENT_SEARCH_ENGINE", "")
    )
    agent_search_assistant: str = field(
        default_factory=lambda: os.getenv("AGENT_SEARCH_ASSISTANT", "default_assistant")
    )
    # streamAssist generationSpec.modelId，留空用 App 默认；需与 GEMINI_MODEL 同代
    agent_search_model_version: str = field(
        default_factory=lambda: os.getenv("AGENT_SEARCH_MODEL_VERSION", "")
    )
    proxy_host: str = field(default_factory=lambda: os.getenv("PROXY_HOST", "0.0.0.0"))
    proxy_port: int = field(default_factory=lambda: int(os.getenv("PROXY_PORT", "8000")))
    cors_origins: str = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "http://localhost:8501"))

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def _autodetect_gcp(s: Settings) -> None:
    if not s.gcp_project_id:
        try:
            import google.auth

            _, default_project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            if default_project:
                s.gcp_project_id = default_project
                logger.info("Detected GCP Project ID from ADC: %s", s.gcp_project_id)
        except Exception as e:
            logger.warning("ADC project detection failed: %s", e)

    if s.gcp_project_id and not s.gcp_project_number:
        try:
            res = subprocess.run(
                ["gcloud", "projects", "describe", s.gcp_project_id, "--format=json"],
                capture_output=True, text=True, check=True, timeout=10,
            )
            s.gcp_project_number = json.loads(res.stdout).get("projectNumber", "")
            logger.info("Resolved GCP Project Number: %s", s.gcp_project_number)
        except Exception as e:
            logger.warning("gcloud project number resolution failed: %s", e)


settings = Settings()
_autodetect_gcp(settings)
