import pytest

from bespokelabs.curator.client import Client


def test_local_only_disables_viewer_from_environment(monkeypatch):
    monkeypatch.setenv("CURATOR_LOCAL_ONLY", "1")
    monkeypatch.setenv("CURATOR_VIEWER", "1")

    with pytest.raises(RuntimeError, match="CURATOR_LOCAL_ONLY"):
        Client()


def test_local_only_disables_explicit_hosted_client(monkeypatch):
    monkeypatch.setenv("CURATOR_LOCAL_ONLY", "true")
    monkeypatch.setenv("CURATOR_VIEWER", "0")

    with pytest.raises(RuntimeError, match="CURATOR_LOCAL_ONLY"):
        Client(hosted=True)


def test_local_client_remains_available(monkeypatch):
    monkeypatch.setenv("CURATOR_LOCAL_ONLY", "1")
    monkeypatch.setenv("CURATOR_VIEWER", "0")

    assert Client().hosted is False
