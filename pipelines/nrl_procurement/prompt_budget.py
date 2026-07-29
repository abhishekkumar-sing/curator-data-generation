"""Local-only rendered request token measurement with an audited fallback."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


def configured_context_window(model_profile: dict[str, Any]) -> int:
    """Return an explicit positive serving-context limit for one model profile."""
    value = model_profile.get("context_window")
    if isinstance(value, bool):
        raise ValueError("Selected model profile must define a positive context_window")
    try:
        context_window = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Selected model profile must define a positive context_window"
        ) from exc
    if context_window <= 0:
        raise ValueError("Selected model profile must define a positive context_window")
    return context_window


def measure_rendered_request(
    messages: list[dict[str, str]],
    response_schema: dict[str, Any],
    *,
    context_window: int,
    reserved_completion_tokens: int,
    safety_margin_tokens: int,
    conservative_chars_per_token: float,
    tokenizer: Any | None = None,
    tokenizer_identity: str = "",
    tokenizer_revision: str = "",
    require_exact: bool = False,
    include_response_schema: bool = True,
    exact_prompt_tokens: int | None = None,
    server_context_window: int | None = None,
    exact_method: str = "vllm_tokenize_endpoint",
) -> dict[str, Any]:
    """Measure a complete chat plus response schema without network access."""
    if context_window < 1 or conservative_chars_per_token <= 0:
        raise ValueError("Invalid prompt-budget configuration")
    schema_text = json.dumps(
        response_schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    template_hash = None
    if exact_prompt_tokens is not None:
        if exact_prompt_tokens < 0:
            raise ValueError("Exact prompt token count cannot be negative")
        prompt_tokens = exact_prompt_tokens
        method = exact_method
    elif tokenizer is not None:
        template = str(getattr(tokenizer, "chat_template", "") or "")
        if not template:
            if require_exact:
                raise ValueError("Exact prompt counting requires a chat template")
        else:
            chat_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            schema_tokens = (
                len(tokenizer.encode(schema_text, add_special_tokens=False))
                if include_response_schema
                else 0
            )
            prompt_tokens = len(chat_ids) + schema_tokens
            method = "tokenizer_chat_template"
            template_hash = hashlib.sha256(template.encode()).hexdigest()
    if exact_prompt_tokens is None and (tokenizer is None or template_hash is None):
        if require_exact:
            raise ValueError("Exact prompt counting requires a local tokenizer")
        rendered_fallback = "\n".join(f"{message['role']}:{message['content']}" for message in messages)
        schema_chars = len(schema_text) if include_response_schema else 0
        prompt_tokens = int(
            (len(rendered_fallback) + schema_chars)
            / conservative_chars_per_token
        ) + 1
        method = "conservative_character_estimate"
    effective_context_window = context_window
    if server_context_window is not None:
        if server_context_window < 1:
            raise ValueError("Server context window must be positive")
        effective_context_window = min(context_window, server_context_window)
    total_reserved = prompt_tokens + reserved_completion_tokens + safety_margin_tokens
    return {
        "method": method,
        "prompt_tokens": prompt_tokens,
        "reserved_completion_tokens": reserved_completion_tokens,
        "safety_margin_tokens": safety_margin_tokens,
        "context_window": effective_context_window,
        "configured_context_window": context_window,
        "server_context_window": server_context_window,
        "passed": total_reserved <= effective_context_window,
        "tokenizer_identity": tokenizer_identity or None,
        "tokenizer_revision": tokenizer_revision or None,
        "chat_template_sha256": template_hash,
    }


def vllm_tokenize_chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    chat_template_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, int]:
    """Ask the selected private vLLM endpoint to render and count a chat."""
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    tokenize_url = urlunsplit(
        (parsed.scheme, parsed.netloc, f"{path}/tokenize", "", "")
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "add_generation_prompt": True,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    if tools:
        payload["tools"] = tools
    request = Request(
        tokenize_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        result = json.loads(response.read().decode("utf-8"))
    count = result.get("count")
    max_model_len = result.get("max_model_len")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or isinstance(max_model_len, bool)
        or not isinstance(max_model_len, int)
        or max_model_len < 1
    ):
        raise ValueError("vLLM /tokenize returned invalid count or max_model_len")
    return {"count": count, "max_model_len": max_model_len}
