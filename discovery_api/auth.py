import logging
import os
import subprocess
import threading
import time

import google.auth
from fastapi import HTTPException
from google.auth.transport.requests import Request as AuthRequest
from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_REFRESH_MARGIN_SEC = 300
_GCLOUD_TOKEN_TTL_SEC = 3000  # gcloud token 有效期 1h，缓存 50min


class TokenCache:
    """进程级 GCP token 缓存。

    优先级：
    1. gcloud CLI active account（带缓存，避免每请求 fork） — 当 ADC 与 CLI 账号不一致时使用
    2. Application Default Credentials
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._creds: Credentials | None = None
        self._creds_expiry: float = 0.0
        self._prefer_cli = os.getenv("GCP_AUTH_SOURCE", "gcloud") == "gcloud"

    def _from_gcloud_cli(self) -> Credentials | None:
        try:
            res = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True, text=True, check=True, timeout=15,
            )
            token = res.stdout.strip()
            if token:
                logger.info("Obtained access token from gcloud CLI active account")
                return Credentials(token)
        except Exception as e:
            logger.warning("gcloud CLI token fetch failed: %s", e)
        return None

    def _from_adc(self) -> Credentials | None:
        try:
            creds, _ = google.auth.default(scopes=_SCOPES)
            creds.refresh(AuthRequest())
            logger.info("Obtained access token from Application Default Credentials")
            return creds
        except Exception as e:
            logger.warning("ADC token fetch failed: %s", e)
        return None

    def _refresh(self) -> None:
        sources = (
            [self._from_gcloud_cli, self._from_adc]
            if self._prefer_cli
            else [self._from_adc, self._from_gcloud_cli]
        )
        for fn in sources:
            if creds := fn():
                self._creds = creds
                expiry = getattr(creds, "expiry", None)
                self._creds_expiry = (
                    expiry.timestamp() if expiry else time.time() + _GCLOUD_TOKEN_TTL_SEC
                )
                return
        raise HTTPException(
            status_code=401,
            detail="GCP Authentication failed. Run `gcloud auth login` or "
            "`gcloud auth application-default login`.",
        )

    def get_credentials(self) -> Credentials:
        with self._lock:
            if self._creds is None or time.time() > self._creds_expiry - _REFRESH_MARGIN_SEC:
                self._refresh()
            return self._creds

    def get_token(self) -> str:
        return self.get_credentials().token


token_cache = TokenCache()


def get_gcp_access_token() -> str:
    return token_cache.get_token()
