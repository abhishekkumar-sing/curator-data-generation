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
    defaults = {
        "CURATOR_LOCAL_ONLY": curator_config.get("local_only", True),
        "CURATOR_VIEWER": curator_config.get("viewer_enabled", False),
        "TELEMETRY_ENABLED": curator_config.get("telemetry_enabled", False),
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, _as_env_bool(value))

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


def require_private_endpoint(name: str) -> str:
    """Return an endpoint only when it targets localhost or a private address."""
    value = require_setting(name)
    parsed = urlparse(value)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host:
        raise SystemExit(f"{name} must be an http(s) URL in {PROJECT_ROOT / '.env'}")
    if parsed.username or parsed.password:
        raise SystemExit(f"{name} must not contain embedded credentials")
    if host.rstrip(".").lower() == "localhost":
        return value
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise SystemExit(
            f"{name} must use localhost or a private IP address; refusing endpoint {host!r}"
        ) from exc
    if not (address.is_private or address.is_loopback or address.is_link_local):
        raise SystemExit(f"{name} must be private; refusing to send data to {host}")
    return value


load_dotenv(PROJECT_ROOT / ".env")
CONFIG = load_config()
configure_curator(CONFIG)
