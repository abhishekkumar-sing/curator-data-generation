import litellm
import pytest
from pydantic import BaseModel

from bespokelabs.curator.request_processor.config import OnlineRequestProcessorConfig
from bespokelabs.curator.request_processor.online import (
    litellm_online_request_processor,
)
from bespokelabs.curator.request_processor.online.base_online_request_processor import APIRequest, TokenLimitStrategy
from bespokelabs.curator.request_processor.online.litellm_online_request_processor import (
    LiteLLMOnlineRequestProcessor,
    dereference_json_schema,
    normalize_tool_arguments,
)
from bespokelabs.curator.status_tracker.online_status_tracker import OnlineStatusTracker
from bespokelabs.curator.types.generic_request import GenericRequest
from bespokelabs.curator.types.prompt import File, _MultiModalPrompt
from bespokelabs.curator.types.token_usage import _TokenUsage


def test_explicit_structured_output_mode_overrides_static_lookup(monkeypatch):
    calls = []

    def unsupported(**kwargs):
        calls.append(kwargs)
        return False

    monkeypatch.setattr(
        litellm_online_request_processor, "supports_response_schema", unsupported
    )
    processor = LiteLLMOnlineRequestProcessor(
        OnlineRequestProcessorConfig(
            model="hosted_vllm/private-model",
            structured_output_mode="md_json",
        )
    )

    assert processor.check_structured_output_support()
    assert calls == []


def test_auto_structured_output_mode_uses_static_lookup(monkeypatch):
    calls = []

    def unsupported(**kwargs):
        calls.append(kwargs)
        return False

    monkeypatch.setattr(
        litellm_online_request_processor, "supports_response_schema", unsupported
    )
    processor = LiteLLMOnlineRequestProcessor(
        OnlineRequestProcessorConfig(model="hosted_vllm/private-model")
    )

    assert not processor.check_structured_output_support()
    assert calls == [{"model": "hosted_vllm/private-model"}]


def test_auto_tool_mode_builds_and_validates_one_schema_tool():
    class NestedProbe(BaseModel):
        label: str
        notes: list[str] = []

    class ProbeResponse(BaseModel):
        value: str
        count: int
        nested: NestedProbe = NestedProbe(label="default")

    api_request = {
        "model": "hosted_vllm/private-model",
        "messages": [{"role": "user", "content": "return a probe"}],
        "temperature": 1.0,
    }
    processor = LiteLLMOnlineRequestProcessor(
        OnlineRequestProcessorConfig(
            model="hosted_vllm/private-model",
            structured_output_mode="tools_auto",
            dereference_tool_schema=True,
        )
    )
    request = processor._auto_tool_request(
        api_request,
        ProbeResponse,
    )

    assert request["tool_choice"] == "auto"
    function = request["tools"][0]["function"]
    parameters = function["parameters"]
    assert function["name"] == "ProbeResponse"
    assert function["strict"] is True
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["value", "count", "nested"]
    assert "$defs" not in parameters
    nested = parameters["properties"]["nested"]
    assert "$ref" not in nested
    assert nested["additionalProperties"] is False
    assert nested["required"] == ["label", "notes"]
    assert request["messages"][0]["role"] == "system"
    assert api_request["messages"] == [
        {"role": "user", "content": "return a probe"}
    ]

    completion = type(
        "Completion",
        (),
        {
            "choices": [
                type(
                    "Choice",
                    (),
                    {
                        "message": type(
                            "Message",
                            (),
                            {
                                "tool_calls": [
                                    type(
                                        "ToolCall",
                                        (),
                                        {
                                            "function": type(
                                                "Function",
                                                (),
                                                {
                                                    "name": "ProbeResponse",
                                                    "arguments": (
                                                        '{"value":"alpha","count":7}'
                                                    ),
                                                },
                                            )()
                                        },
                                    )()
                                ]
                            },
                        )()
                    },
                )()
            ]
        },
    )()
    parsed = LiteLLMOnlineRequestProcessor._parse_auto_tool_completion(
        completion,
        ProbeResponse,
    )
    assert parsed == ProbeResponse(value="alpha", count=7)

    completion.choices[0].message.tool_calls = []
    with pytest.raises(ValueError, match="exactly one ProbeResponse"):
        LiteLLMOnlineRequestProcessor._parse_auto_tool_completion(
            completion,
            ProbeResponse,
        )


def test_auto_tool_schema_dereference_is_opt_in():
    class NestedProbe(BaseModel):
        label: str

    class ProbeResponse(BaseModel):
        nested: NestedProbe

    processor = LiteLLMOnlineRequestProcessor(
        OnlineRequestProcessorConfig(
            model="hosted_vllm/private-model",
            structured_output_mode="tools_auto",
        )
    )
    request = processor._auto_tool_request(
        {"messages": [{"role": "user", "content": "probe"}]},
        ProbeResponse,
    )

    parameters = request["tools"][0]["function"]["parameters"]
    assert "$defs" in parameters
    assert parameters["properties"]["nested"]["$ref"] == "#/$defs/NestedProbe"


def test_dereference_json_schema_rejects_recursive_refs():
    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
            }
        },
        "$ref": "#/$defs/Node",
    }

    with pytest.raises(ValueError, match="Recursive schema"):
        dereference_json_schema(schema)


def test_normalize_tool_arguments_repairs_nested_json_strings():
    schema = {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string"},
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "quote": {"type": "string"},
                                },
                                "required": ["quote"],
                            },
                        },
                    },
                },
            },
        },
    }
    value = {
        "claims": (
            '[{"statement":"Supported statement","evidence":'
            '["Exact source quotation."]}]'
        )
    }

    normalized = normalize_tool_arguments(value, schema)

    assert normalized["claims"][0]["evidence"] == [
        {"quote": "Exact source quotation."}
    ]


def test_normalize_tool_arguments_does_not_decode_ordinary_strings():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }

    assert normalize_tool_arguments({"answer": "[not JSON]"}, schema) == {
        "answer": "[not JSON]"
    }


def test_multimodal_support_falls_back_for_claude_aliases(monkeypatch):
    monkeypatch.setattr(litellm, "supports_vision", lambda model: False)

    processor = LiteLLMOnlineRequestProcessor(OnlineRequestProcessorConfig(model="anthropic/claude-sonnet-4-6"))

    assert processor._multimodal_prompt_supported is True


def test_multimodal_support_does_not_fallback_for_non_claude_models(monkeypatch):
    monkeypatch.setattr(litellm, "supports_vision", lambda model: False)

    processor = LiteLLMOnlineRequestProcessor(OnlineRequestProcessorConfig(model="openai/gpt-4o-mini"))

    assert processor._multimodal_prompt_supported is False


def test_requests_to_responses_uses_resolved_token_limit_strategy(monkeypatch):
    processor = LiteLLMOnlineRequestProcessor(OnlineRequestProcessorConfig(model="anthropic/claude-sonnet-4-6"))
    processor.prompt_formatter = type("PromptFormatter", (), {"model_name": "anthropic/claude-sonnet-4-6"})()
    processor.total_requests = 0

    def fake_rate_limits():
        processor.token_limit_strategy = TokenLimitStrategy.seperate
        return 20000, _TokenUsage(input=10000000, output=2000000)

    monkeypatch.setattr(processor, "get_header_based_rate_limits", fake_rate_limits)

    processor.requests_to_responses([])

    assert processor.tracker.token_limit_strategy == TokenLimitStrategy.seperate
    assert processor.tracker.max_tokens_per_minute == _TokenUsage(input=10000000, output=2000000)


def test_explicit_rate_limits_skip_provider_bootstrap_call(monkeypatch):
    processor = LiteLLMOnlineRequestProcessor(
        OnlineRequestProcessorConfig(
            model="hosted_vllm/test",
            max_requests_per_minute=100,
            max_tokens_per_minute=1000,
        )
    )

    def unexpected_probe():
        raise AssertionError("explicit limits must not make a provider call")

    monkeypatch.setattr(
        processor,
        "get_header_based_rate_limits",
        unexpected_probe,
    )

    assert processor.max_requests_per_minute == 100
    assert processor.max_tokens_per_minute == 1000
    assert processor._rate_limits_initialized is True


def test_anthropic_file_prompts_use_document_blocks_and_pdf_beta_header():
    processor = LiteLLMOnlineRequestProcessor(OnlineRequestProcessorConfig(model="anthropic/claude-sonnet-4-6"))
    request = GenericRequest(
        model="anthropic/claude-sonnet-4-6",
        messages=[
            {
                "role": "user",
                "content": _MultiModalPrompt(texts=["Describe the pdf"], files=[File(url="https://example.com/sample.pdf")]).model_dump(),
            }
        ],
        original_row={"pdf": "https://example.com/sample.pdf", "text": "Describe the pdf"},
        original_row_idx=0,
        is_multimodal_prompt=True,
    )

    request = processor._unpack_multimodal(request)
    api_request = processor.create_api_specific_request_online(request)

    assert api_request["messages"][0]["content"][1]["type"] == "document"
    assert api_request["messages"][0]["content"][1]["source"] == {
        "type": "url",
        "url": "https://example.com/sample.pdf",
    }
    assert api_request["extra_headers"]["anthropic-beta"] == "pdfs-2024-09-25"


def test_estimate_total_tokens_falls_back_for_anthropic_document_prompts():
    processor = LiteLLMOnlineRequestProcessor(OnlineRequestProcessorConfig(model="anthropic/claude-sonnet-4-6"))

    tokens = processor.estimate_total_tokens(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the pdf"},
                    {"type": "document", "source": {"type": "url", "url": "https://example.com/sample.pdf"}},
                ],
            }
        ]
    )

    assert tokens.input >= 2048


@pytest.mark.asyncio
async def test_rate_limit_error_never_drives_num_api_errors_negative(monkeypatch):
    """T23 regression test.

    litellm_online_request_processor.py's RateLimitError branch never increments
    num_api_errors (unlike the openai/anthropic siblings, which parse a generic
    {"error": ...} body and bump it before learning it's specifically a rate limit),
    so decrementing it had nothing to offset and drove it negative on every rate-limit
    error. The fix redirects the compensating decrement to num_other_errors, which is
    what handle_single_request_with_retries's outer except block actually double-counts.
    """
    processor = LiteLLMOnlineRequestProcessor(
        OnlineRequestProcessorConfig(
            model="hosted_vllm/test",
            max_requests_per_minute=100,
            max_tokens_per_minute=1000,
        )
    )

    async def raise_rate_limit(**kwargs):
        raise litellm.RateLimitError(message="rate limited", llm_provider="hosted_vllm", model="test")

    monkeypatch.setattr(litellm, "acompletion", raise_rate_limit)

    generic_request = GenericRequest(
        model="hosted_vllm/test",
        messages=[{"role": "user", "content": "hi"}],
        original_row={},
        original_row_idx=0,
    )
    api_request = APIRequest(
        task_id=0,
        generic_request=generic_request,
        api_specific_request={"model": "hosted_vllm/test", "messages": [{"role": "user", "content": "hi"}]},
        attempts_left=1,
    )
    status_tracker = OnlineStatusTracker()

    with pytest.raises(litellm.RateLimitError):
        await processor.call_single_request(api_request, session=None, status_tracker=status_tracker)

    assert status_tracker.num_api_errors == 0
    assert status_tracker.num_rate_limit_errors == 1
    assert status_tracker.num_other_errors == -1

    # Mirrors what handle_single_request_with_retries's outer except block does for
    # every exception (including this re-raised RateLimitError), to prove the two
    # nets out to zero double-counting instead of a phantom negative num_api_errors.
    status_tracker.num_other_errors += 1
    assert status_tracker.num_api_errors == 0
    assert status_tracker.num_other_errors == 0
    assert status_tracker.num_rate_limit_errors == 1
