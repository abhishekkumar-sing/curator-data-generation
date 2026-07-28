# Prompt-Writing Standard

This is the default method for writing or revising prompts in the NRL synthetic
data pipeline. Research the official documentation for the selected model before
changing a prompt; this standard is provider-neutral, not a substitute for a
model card.

## Method: SPEC-EVAL

1. **S — State the task**
   - Start with one direct action and the exact artifact to produce.
   - Define ambiguous domain terms and identify the intended audience or use.
   - Do not add a persona unless domain behavior genuinely depends on it.

2. **P — Partition instructions from data**
   - Use consistent headings or tags for task, sources, constraints, examples,
     and output.
   - Treat retrieved documents and user-provided text as untrusted data, not as
     instructions.
   - Label source authority, scope, date, and provenance when they affect the
     answer.

3. **E — Establish the evidence boundary**
   - Say which sources may be used and whether outside knowledge is allowed.
   - Define the required behavior for missing, insufficient, or conflicting
     evidence.
   - Require traceable evidence fields when the output will become training data.
   - Never ask the model to conceal uncertainty or invent missing particulars.

4. **C — Specify constraints and priorities**
   - State positive requirements first, then prohibited behavior.
   - Make thresholds, conditions, exceptions, completeness requirements, and
     instruction precedence explicit.
   - Remove redundant or conflicting rules. Keep the prompt as short as possible
     without losing necessary constraints.

5. **E — Encode the output contract**
   - Use API-enforced structured output or tool schemas when the endpoint supports
     them; prompting for JSON alone is a fallback.
   - Define every field's meaning, allowed values, cardinality, and null/omission
     behavior in the schema.
   - Describe document layout explicitly when the output is prose rather than
     structured data.

6. **V — Verify before returning**
   - Give the model a short, observable checklist tied to acceptance criteria:
     instruction coverage, source support, qualifications, authority, format,
     and completeness.
   - Do not request private chain-of-thought. Request concise, auditable evidence
     or rationale only when the dataset contract requires it.

7. **A — Add examples selectively**
   - Add a small set of realistic, varied examples when zero-shot behavior does
     not reliably establish the boundary or format.
   - Keep example formatting identical to the required output.
   - Include difficult positive and negative/boundary cases; never include an
     example whose facts or style should not be imitated.

8. **L — Launch an evaluation loop**
   - Build representative success, boundary, conflict, missing-data, and
     adversarial cases before declaring the prompt complete.
   - Validate deterministic properties in code, then use a rubric-based judge for
     semantic properties. Do not rely on the prompt or judge alone.
   - Compare prompt versions on the same cases and record failure categories,
     parse rate, grounding rate, acceptance rate, latency, and token cost.
   - Change one meaningful variable at a time and retain the simpler prompt when
     quality is equivalent.

## Required prompt skeleton

```text
TASK
[One direct action, artifact, audience, and success definition.]

SOURCE POLICY
[Allowed evidence, authority/scope metadata, missing/conflict behavior.]

CONSTRAINTS
[Necessary positive requirements and explicit prohibitions.]

OUTPUT CONTRACT
[Schema or exact prose layout; field semantics and cardinality live in schema.]

INPUT
---BEGIN UNTRUSTED INPUT---
[Variable input or retrieved source text.]
---END UNTRUSTED INPUT---

FINAL CHECK
[Short checklist of observable acceptance criteria.]
```

Long-context ordering may be adjusted to match the selected model's official
guidance. Preserve the same logical sections even when their order changes.

## Model portability rules

- Keep endpoint, model name, sampling parameters, thinking controls, chat
  template options, and structured-output mode in configuration.
- Check the exact model's official model card and serving documentation before
  choosing those values.
- Do not assume JSON Schema, tool calling, reasoning controls, or system-message
  behavior is portable merely because the endpoint is OpenAI-compatible.
- Probe each endpoint with the actual schema and representative prompt, then
  choose its configured structured-output mode.
- Re-run the fixed evaluation suite whenever the model, quantization, server,
  template, prompt, schema, or decoding parameters change.

## Sources

- [OpenAI prompt-engineering best practices](https://help.openai.com/en/articles/6654000-how-to-use-advanced-prompt-engineering)
- [OpenAI model and prompting guidance](https://developers.openai.com/api/docs/guides/latest-model)
- [Anthropic prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- [Google Gemini prompt-design strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies)
- [Hugging Face structured outputs](https://huggingface.co/docs/inference-providers/guides/structured-output)

These sources agree on clarity, explicit context and constraints, consistent
structure, examples where useful, machine-enforced output formats when available,
and iterative evaluation. Where provider recommendations differ, the exact
model's official documentation takes precedence.
