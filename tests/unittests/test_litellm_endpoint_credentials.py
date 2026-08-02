"""Request-local credential tests for independent hosted-vLLM endpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

import litellm

from bespokelabs.curator.request_processor.online.litellm_online_request_processor import (
    APIRequest,
    LiteLLMOnlineRequestProcessor,
)
from bespokelabs.curator.types.generic_request import GenericRequest


class _Completion:
    """Minimal LiteLLM response used by credential propagation tests."""

    def __init__(self) -> None:
        self._hidden_params = {"additional_headers": {"x-test": "ok"}}
        self.usage = SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )
        self.choices = [
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="ok"),
            )
        ]

    def model_dump(self) -> dict:
        """Return the safe provider response shape needed by Curator."""
        return {
            "model": "hosted_vllm/test",
            "choices": [
                {"finish_reason": "stop", "message": {"content": "ok"}}
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    def __getitem__(self, key: str):
        """Provide LiteLLM's mapping-style response access."""
        return self.model_dump()[key]


def _processor() -> LiteLLMOnlineRequestProcessor:
    processor = object.__new__(LiteLLMOnlineRequestProcessor)
    processor.config = SimpleNamespace(
        model="hosted_vllm/test",
        base_url="http://generator.test/v1",
        api_key="role-specific-secret",
        generation_params={"max_tokens": 16},
        request_timeout=30,
        return_completions_object=False,
        structured_output_mode="tools_auto",
    )
    processor.completion_cost = lambda response: 0.0
    return processor


def _request(*, structured: bool = False) -> tuple[APIRequest, dict]:
    source_request = {
        "model": "hosted_vllm/test",
        "messages": [{"role": "user", "content": "hello"}],
        "api_base": "http://generator.test/v1",
    }
    generic_request = GenericRequest(
        model="hosted_vllm/test",
        messages=source_request["messages"],
        response_format={"type": "object"} if structured else None,
        original_row={"id": "row-1"},
        original_row_idx=0,
    )
    return (
        APIRequest(
            task_id=0,
            generic_request=generic_request,
            api_specific_request=source_request,
            attempts_left=1,
            prompt_formatter=SimpleNamespace(
                response_format=object() if structured else None
            ),
            created_at=datetime.now(),
        ),
        source_request,
    )


def _tracker() -> SimpleNamespace:
    return SimpleNamespace(
        time_of_last_rate_limit_error=None,
        num_rate_limit_errors=0,
        num_api_errors=0,
    )


def test_rate_limit_probe_passes_processor_api_key(monkeypatch) -> None:
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return _Completion()

    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **kwargs: 0.0)
    processor = _processor()
    assert processor.test_call() == {"x-test": "ok"}
    assert captured["api_key"] == "role-specific-secret"
    assert captured["api_base"] == "http://generator.test/v1"


def test_completion_uses_transient_key_but_persists_safe_request(
    monkeypatch,
) -> None:
    captured = {}

    async def completion(**kwargs):
        captured.update(kwargs)
        return _Completion()

    monkeypatch.setattr(litellm, "acompletion", completion)
    processor = _processor()
    request, source_request = _request()
    response = asyncio.run(
        processor.call_single_request(
            request=request,
            session=None,
            status_tracker=_tracker(),
        )
    )
    assert captured["api_key"] == "role-specific-secret"
    assert "api_key" not in source_request
    assert "api_key" not in response.raw_request


def test_tools_auto_completion_receives_transient_key(monkeypatch) -> None:
    captured = {}

    async def completion(**kwargs):
        captured.update(kwargs)
        return _Completion()

    monkeypatch.setattr(litellm, "acompletion", completion)
    processor = _processor()
    processor._auto_tool_request = lambda request, response_model: request
    processor._parse_auto_tool_completion = lambda completion, response_model: (
        SimpleNamespace(model_dump=lambda: {"answer": "ok"})
    )
    request, source_request = _request(structured=True)

    response = asyncio.run(
        processor.call_single_request(request, None, _tracker())
    )

    assert captured["api_key"] == "role-specific-secret"
    assert "api_key" not in source_request
    assert "api_key" not in response.raw_request


def test_instructor_completion_receives_transient_key() -> None:
    captured = {}

    async def create_with_completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model_dump=lambda: {"answer": "ok"}), _Completion()

    processor = _processor()
    processor.config.structured_output_mode = "json_schema"
    processor.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create_with_completion=create_with_completion
            )
        )
    )
    request, source_request = _request(structured=True)

    response = asyncio.run(
        processor.call_single_request(request, None, _tracker())
    )

    assert captured["api_key"] == "role-specific-secret"
    assert "api_key" not in source_request
    assert "api_key" not in response.raw_request
