"""Configuration loading and local-only safety controls for the NRL pipeline."""

import ipaddress
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_TRUE_VALUES = {"1", "true", "t", "yes", "on"}


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE settings without adding a dotenv dependency."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the committed pipeline configuration."""
    if not path.is_file():
        raise RuntimeError(f"Configuration file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Configuration must be a YAML mapping: {path}")
    return payload


def _as_env_bool(value: Any) -> str:
    return "1" if str(value).strip().lower() in _TRUE_VALUES else "0"


def configure_curator(config: dict[str, Any]) -> None:
    """Apply configurable Curator privacy settings before Curator is imported."""
    curator_config = config.get("curator", {})
    cache_dir = (
        PROJECT_ROOT / str(curator_config.get("cache_dir", ".curator_working"))
    ).resolve()
    try:
        cache_dir.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise RuntimeError("curator.cache_dir must stay inside the project") from exc
    if cache_dir != PROJECT_ROOT / ".curator_working":
        raise RuntimeError("curator.cache_dir must resolve to .curator_working")
    defaults = {
        "CURATOR_LOCAL_ONLY": curator_config.get("local_only", True),
        "CURATOR_VIEWER": curator_config.get("viewer_enabled", False),
        "TELEMETRY_ENABLED": curator_config.get("telemetry_enabled", False),
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, _as_env_bool(value))
    # This pipeline owns its cache location; do not inherit a user-global cache.
    os.environ["CURATOR_CACHE_DIR"] = str(cache_dir)

    if os.environ["CURATOR_LOCAL_ONLY"].lower() in _TRUE_VALUES:
        if os.environ["CURATOR_VIEWER"].lower() in _TRUE_VALUES:
            raise RuntimeError(
                "CURATOR_VIEWER cannot be enabled while CURATOR_LOCAL_ONLY is enabled"
            )
        if os.environ["TELEMETRY_ENABLED"].lower() in _TRUE_VALUES:
            raise RuntimeError(
                "TELEMETRY_ENABLED cannot be enabled while CURATOR_LOCAL_ONLY is enabled"
            )


def require_setting(name: str) -> str:
    """Return a required non-empty environment setting."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Set {name} in {PROJECT_ROOT / '.env'}")
    return value


def is_private_host(host: str | None) -> bool:
    """Return whether `host` is localhost or a private/loopback/link-local IP."""
    if not host:
        return False
    if host.rstrip(".").lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(address.is_private or address.is_loopback or address.is_link_local)


def validate_endpoint_url(
    url: str,
    name: str,
    *,
    allow_public_https: bool = False,
) -> str:
    """Enforce the pipeline's single outbound-endpoint privacy policy.

    Always rejects a non-absolute http(s) URL, embedded credentials, and
    query parameters/fragments (which could carry credentials). When
    `allow_public_https` is False (the default — used by every generation,
    judge, and OCR endpoint), the host must resolve to localhost or a
    private/loopback/link-local IP address regardless of scheme. When
    `allow_public_https` is True (used only by the optional
    semantic-diversity embedding endpoint, which sends generated question
    text only — never source or answer text), a public host is permitted but
    only over HTTPS.

    Raises `ValueError` naming `name` on any violation; callers translate
    this into their own exception convention (e.g. `SystemExit` for CLI
    startup checks, `RuntimeError` for config loading).
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host:
        raise ValueError(f"{name} must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{name} must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{name} must not contain query parameters or a fragment")
    private = is_private_host(host)
    if not allow_public_https:
        if not private:
            raise ValueError(f"{name} must be private; refusing to send data to {host}")
        return url
    if parsed.scheme != "https" and not private:
        raise ValueError(f"Public {name} endpoints must use HTTPS")
    return url


def require_private_endpoint(name: str) -> str:
    """Return an endpoint only when it targets localhost or a private address."""
    value = require_setting(name)
    try:
        return validate_endpoint_url(value, name, allow_public_https=False)
    except ValueError as exc:
        raise SystemExit(f"{exc} ({PROJECT_ROOT / '.env'})") from exc


load_dotenv(PROJECT_ROOT / ".env")
CONFIG = load_config()
configure_curator(CONFIG)
