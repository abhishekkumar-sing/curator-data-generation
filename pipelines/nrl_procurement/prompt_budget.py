"""Local-only rendered request token measurement with an audited fallback."""

from __future__ import annotations

import hashlib
import json
from typing import Any


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
    if tokenizer is not None:
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
            schema_ids = tokenizer.encode(schema_text, add_special_tokens=False)
            prompt_tokens = len(chat_ids) + len(schema_ids)
            method = "tokenizer_chat_template"
            template_hash = hashlib.sha256(template.encode()).hexdigest()
    if tokenizer is None or template_hash is None:
        if require_exact:
            raise ValueError("Exact prompt counting requires a local tokenizer")
        rendered_fallback = "\n".join(f"{message['role']}:{message['content']}" for message in messages)
        prompt_tokens = int((len(rendered_fallback) + len(schema_text)) / conservative_chars_per_token) + 1
        method = "conservative_character_estimate"
    total_reserved = prompt_tokens + reserved_completion_tokens + safety_margin_tokens
    return {
        "method": method,
        "prompt_tokens": prompt_tokens,
        "reserved_completion_tokens": reserved_completion_tokens,
        "safety_margin_tokens": safety_margin_tokens,
        "context_window": context_window,
        "passed": total_reserved <= context_window,
        "tokenizer_identity": tokenizer_identity or None,
        "tokenizer_revision": tokenizer_revision or None,
        "chat_template_sha256": template_hash,
    }
