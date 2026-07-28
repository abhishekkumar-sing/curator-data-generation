"""Tests for work-conserving online request retries."""

import asyncio
import json
from types import SimpleNamespace

from bespokelabs.curator.request_processor.base_request_processor import (
    BaseRequestProcessor,
)
from bespokelabs.curator.request_processor.online.base_online_request_processor import (
    APIRequest,
    BaseOnlineRequestProcessor,
)
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.generic_response import GenericResponse
from bespokelabs.curator.types.token_usage import _TokenUsage


class _RetryStatus:
    def __init__(self) -> None:
        self.max_tokens_per_minute = 1_000_000
        self.num_other_errors = 0
        self.num_tasks_in_progress = 1
        self.num_tasks_succeeded = 0
        self.num_tasks_failed = 0
        self.capacity_consumptions = 0

    def update_cost_projection(self, *args, **kwargs) -> None:
        pass

    def update_stats(self, *args, **kwargs) -> None:
        pass

    def has_capacity(self, token_estimate) -> bool:
        return True

    def consume_capacity(self, token_estimate) -> None:
        self.capacity_consumptions += 1


class _CountingSemaphore:
    def __init__(self) -> None:
        self.releases = 0

    def release(self) -> None:
        self.releases += 1


def _request(*, attempts_left: int) -> APIRequest:
    generic_request = GenericRequest(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        original_row={"id": "row-1"},
        original_row_idx=0,
    )
    return APIRequest(
        task_id=0,
        generic_request=generic_request,
        api_specific_request={"model": "test-model"},
        attempts_left=attempts_left,
        prompt_formatter=None,
    )


def _processor(*, max_retries: int, call_single_request):
    async def no_cooldown(status_tracker) -> None:
        pass

    async def no_viewer_log(status_tracker) -> None:
        pass

    appended = []

    async def append_response(status_tracker, response, response_file) -> None:
        appended.append(response)

    semaphore = _CountingSemaphore()
    processor = SimpleNamespace(
        config=SimpleNamespace(
            invalid_finish_reasons=["length"],
            max_retries=max_retries,
        ),
        prompt_formatter=SimpleNamespace(
            response_to_response_format=lambda response: response
        ),
        _viewer_client=SimpleNamespace(log_cost_projection=no_viewer_log),
        _semaphore=semaphore,
        call_single_request=call_single_request,
        append_generic_response=append_response,
        estimate_total_tokens=lambda messages: _TokenUsage(input=10, output=20),
        cool_down_if_rate_limit_error=no_cooldown,
        _add_output_token_moving_window=lambda tokens: None,
        _free_capacity=lambda status, used, blocked: None,
    )
    return processor, semaphore, appended


def test_transient_failures_retry_in_active_task_and_release_once() -> None:
    request = _request(attempts_left=2)
    attempts = 0

    async def call_single_request(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError(f"transient-{attempts}")
        return GenericResponse(
            response_message={"answer": "ok"},
            raw_response={"id": "response-1"},
            raw_request={"model": "test-model"},
            generic_request=request.generic_request,
            created_at=request.created_at,
            finished_at=request.created_at,
            token_usage=_TokenUsage(input=10, output=5),
            response_cost=0.0,
            finish_reason="stop",
        )

    processor, semaphore, appended = _processor(
        max_retries=2,
        call_single_request=call_single_request,
    )
    status = _RetryStatus()

    asyncio.run(
        BaseOnlineRequestProcessor.handle_single_request_with_retries(
            processor,
            request=request,
            session=None,
            response_file="unused.jsonl",
            status_tracker=status,
            blocked_capacity=_TokenUsage(input=10, output=20),
        )
    )

    assert attempts == 3
    assert semaphore.releases == 1
    assert status.capacity_consumptions == 2
    assert status.num_tasks_succeeded == 1
    assert status.num_tasks_failed == 0
    assert status.num_tasks_in_progress == 0
    assert len(appended) == 1
    assert appended[0].response_message == {"answer": "ok"}


def test_permanent_failure_is_recorded_and_releases_once() -> None:
    request = _request(attempts_left=1)
    attempts = 0

    async def call_single_request(**kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("still unavailable")

    processor, semaphore, appended = _processor(
        max_retries=1,
        call_single_request=call_single_request,
    )
    status = _RetryStatus()

    asyncio.run(
        BaseOnlineRequestProcessor.handle_single_request_with_retries(
            processor,
            request=request,
            session=None,
            response_file="unused.jsonl",
            status_tracker=status,
            blocked_capacity=_TokenUsage(input=10, output=20),
        )
    )

    assert attempts == 2
    assert semaphore.releases == 1
    assert status.capacity_consumptions == 1
    assert status.num_tasks_succeeded == 0
    assert status.num_tasks_failed == 1
    assert status.num_tasks_in_progress == 0
    assert len(appended) == 1
    assert appended[0].response_message is None
    assert appended[0].response_errors == ["still unavailable(x2)"]


def test_filtered_parse_response_is_still_persisted(monkeypatch) -> None:
    """A successful provider response must survive parse-time filtering."""

    request = _request(attempts_left=0)
    response = GenericResponse(
        response_message={"answer": "provider output"},
        raw_response={"id": "response-1"},
        raw_request={"model": "test-model"},
        generic_request=request.generic_request,
        created_at=request.created_at,
        finished_at=request.created_at,
        token_usage=_TokenUsage(input=10, output=5),
        response_cost=0.0,
        finish_reason="stop",
    )

    async def stream_response(*args, **kwargs) -> None:
        pass

    processor = SimpleNamespace(
        _process_response=lambda data: [],
        viewer_client=SimpleNamespace(stream_response=stream_response),
    )
    status = SimpleNamespace(num_parsed_responses=0)
    writes = []

    class _AsyncFile:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def write(self, value):
            writes.append(value)

    monkeypatch.setattr(
        "bespokelabs.curator.request_processor.online.base_online_request_processor.aiofiles.open",
        lambda *args, **kwargs: _AsyncFile(),
    )

    asyncio.run(
        BaseOnlineRequestProcessor.append_generic_response(
            processor,
            status,
            response,
            "responses_0.jsonl",
        )
    )

    persisted = json.loads(writes[0])
    assert persisted["response_message"] == {"answer": "provider output"}
    assert persisted["parsed_response_message"] is None
    assert status.num_parsed_responses == 0


def test_all_filtered_provider_responses_return_empty_dataset(tmp_path) -> None:
    """Filtering every valid response is not equivalent to provider failure."""

    request = _request(attempts_left=0)
    response = GenericResponse(
        response_message={"answer": "provider output"},
        raw_response={"id": "response-1"},
        raw_request={"model": "test-model"},
        generic_request=request.generic_request,
        created_at=request.created_at,
        finished_at=request.created_at,
        token_usage=_TokenUsage(input=10, output=5),
        response_cost=0.0,
        finish_reason="stop",
    )
    (tmp_path / "responses_0.jsonl").write_text(
        response.model_dump_json() + "\n",
        encoding="utf-8",
    )
    processor = SimpleNamespace(
        working_dir=str(tmp_path),
        _process_response=lambda data: [],
        prompt_formatter=SimpleNamespace(
            response_to_response_format=lambda value: value,
            parse_func=lambda row, value: [],
        ),
        config=SimpleNamespace(require_all_responses=False),
    )

    dataset = BaseRequestProcessor.create_dataset_files(processor, "filtered")

    assert dataset.to_list() == []
    assert not (tmp_path / "filtered.arrow").exists()
