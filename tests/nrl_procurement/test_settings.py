"""Tests for the shared outbound-endpoint privacy policy."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[2] / "pipelines" / "nrl_procurement"
sys.path.insert(0, str(PIPELINE))

from settings import (  # noqa: E402
    is_private_host,
    require_private_endpoint,
    validate_endpoint_url,
)


def test_is_private_host_accepts_localhost_and_private_ips() -> None:
    assert is_private_host("localhost") is True
    assert is_private_host("LOCALHOST.") is True
    assert is_private_host("127.0.0.1") is True
    assert is_private_host("::1") is True
    assert is_private_host("10.180.148.183") is True
    assert is_private_host("192.168.1.5") is True


def test_is_private_host_rejects_public_ip_and_hostname() -> None:
    assert is_private_host("integrate.api.nvidia.com") is False
    assert is_private_host("8.8.8.8") is False
    assert is_private_host(None) is False
    assert is_private_host("") is False


def test_validate_endpoint_url_default_policy_rejects_public_host() -> None:
    """A new call site that doesn't pass allow_public_https defaults to private-only."""
    with pytest.raises(ValueError, match="must be private"):
        validate_endpoint_url("https://integrate.api.nvidia.com/v1", "SOME_NEW_ENDPOINT")


def test_validate_endpoint_url_default_policy_accepts_private_host_any_scheme() -> None:
    assert (
        validate_endpoint_url("http://10.180.148.183:8010/v1", "SOME_NEW_ENDPOINT")
        == "http://10.180.148.183:8010/v1"
    )


def test_validate_endpoint_url_allow_public_https_permits_https_public_host() -> None:
    endpoint = "https://integrate.api.nvidia.com/v1/embeddings"
    assert (
        validate_endpoint_url(endpoint, "EMBEDDING_BASE_URL", allow_public_https=True)
        == endpoint
    )


def test_validate_endpoint_url_allow_public_https_still_rejects_public_http() -> None:
    with pytest.raises(ValueError, match="must use HTTPS"):
        validate_endpoint_url(
            "http://integrate.api.nvidia.com/v1/embeddings",
            "EMBEDDING_BASE_URL",
            allow_public_https=True,
        )


def test_validate_endpoint_url_rejects_embedded_credentials() -> None:
    with pytest.raises(ValueError, match="embedded credentials"):
        validate_endpoint_url(
            "https://user:pass@integrate.api.nvidia.com/v1",
            "EMBEDDING_BASE_URL",
            allow_public_https=True,
        )


def test_validate_endpoint_url_rejects_query_and_fragment() -> None:
    with pytest.raises(ValueError, match="query parameters"):
        validate_endpoint_url(
            "https://integrate.api.nvidia.com/v1?api_key=leak",
            "EMBEDDING_BASE_URL",
            allow_public_https=True,
        )


def test_require_private_endpoint_raises_system_exit_for_public_host(monkeypatch) -> None:
    monkeypatch.setenv("SOME_TEST_ENDPOINT", "https://integrate.api.nvidia.com/v1")
    with pytest.raises(SystemExit, match="must be private"):
        require_private_endpoint("SOME_TEST_ENDPOINT")


def test_require_private_endpoint_accepts_private_ip(monkeypatch) -> None:
    monkeypatch.setenv("SOME_TEST_ENDPOINT", "http://10.180.148.183:8010/v1")
    assert require_private_endpoint("SOME_TEST_ENDPOINT") == "http://10.180.148.183:8010/v1"
