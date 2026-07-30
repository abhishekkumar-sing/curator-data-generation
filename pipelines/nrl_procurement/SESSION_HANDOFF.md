# NRL Procurement Pipeline — Session Handoff

Last updated: 2026-07-30

## Current objective

Produce grounded, independently validated procurement training data from the
NRL manuals. The core release must contain non-empty, valid exports for:

1. single-document QA;
2. single-document QA-CoT;
3. cross-document QA;
4. cross-document QA-CoT.

Do not scale to a full generation run until the smoke-test release gates below
pass. Drafting and temporal exports are useful extensions, but they must not
hide or block the status of the four core exports.

## Authoritative inputs

- Dynamic registry: `/home/abhishek/curator/data/source/manuals.yaml`
- Original NRL PDFs:
  - `data/source/nrlManual_Procurement_of_Goods_Rev1.pdf`
  - `data/source/nrlManual_Procurement_of_Works_Rev1.pdf`
  - `data/source/nrlManual_Procurement_of_Consultancy_Other_Services_Rev1.pdf`
- Pipeline text input: Chandra OCR artifacts under `data/interim/ocr`

`manuals.yaml` will grow as documents arrive. Never hard-code the current
three-manual set. Preserve both identities:

- `source_sha256`: original PDF identity;
- `content_sha256`: OCR content identity.

A changed PDF or OCR artifact must change the relevant fingerprint and
invalidate dependent artifacts.

## Model contract

Read live values from `config.yaml` and `.env`; never copy API keys into logs,
manifests, fingerprints, or this document.

At this handoff:

- Generation: `hosted_vllm/nvidia/nemotron-3-super`, tool-call structured
  output, deployment identity
  `nemotron-3-super-fp8-vllm-0.25-strict-bypass-v1`.
- Judge: the active `gemma` profile uses JSON-schema structured output with
  `temperature=1.0`, `top_k=64`, and `top_p=0.95`. Its local endpoint is
  configured through ignored `.env` variables rather than tracked files. The
  supplied endpoint was updated on 2026-07-30; verify `/models` before a live
  pilot because its initial probe was reset by the server.
- OCR: Chandra uses the private vLLM endpoint configured through ignored
  `OCR_MODEL`, `OCR_BASE_URL`, and `OCR_API_KEY` variables. The OCR command is
  `chandra`; source PDFs are written to `data/interim/ocr` and that OCR output,
  not direct PDF extraction, is what the generation pipeline consumes.

Endpoint relocation may reuse a checkpoint only when the explicit deployment
identity and semantic stage contract are unchanged. A model, prompt, schema,
validator, or stage-contract change must invalidate the dependent checkpoint.

## Fixes completed after pilot-016

- Normalized common Nemotron structured-output defects, including nested JSON
  arrays returned as strings, before Pydantic validation.
- Added safe judge quote recovery. Recovery is allowed only when the judge says
  the answer is supported and the answer text is found verbatim inside evidence
  that already passed deterministic validation.
- Materialized every exhausted or omitted model request as an explicit terminal
  audit row with `terminal_state=model_failure_after_retries`. Missing records
  may no longer disappear between planned inputs and audits.
- Added contract versions and strict contract-hash checks to completed
  checkpoint reuse. Semantically stale checkpoints are no longer silently
  reused.
- Restricted validator failure statistics to stages belonging to the latest
  manifest attempt. Old working-directory failures no longer inflate the
  current attempt.
- Tightened cross-document QA-CoT validation: at least one synthesis step must
  use exact evidence from both `source_a` and `source_b`.
- Removed generic terms such as `consultant`, `supplier`, `payment`, and
  `procurement` from bridge anchors. Generic vocabulary alone cannot establish
  a connected reasoning path.
- Improved temporal-pair pilot sampling and made drafting failure non-fatal to
  the core manifest.

Focused verification at handoff:

```text
ruff: passed
pytest: 96 passed, 3 dependency deprecation warnings
```

The LiteLLM pricing warning and PyArrow `null_placement` warning are
non-blocking. Structural validation errors, permanent request failures, missing
lineage, or empty core exports are blocking.

## Non-negotiable generation rules

### Evidence and propositions

- Extract atomic propositions independently from each source window.
- Store exact evidence text and exact offsets.
- Preserve authority, modality, polarity, conditions, exceptions, thresholds,
  and temporal scope.
- Reject ambiguous duplicate occurrences; do not guess offsets.
- Never convert NRL policy into Government authority, or Government guidance
  into NRL policy.
- Never claim currentness, supersession, or an amendment without verified
  source metadata and a recorded verification cutoff.

### Reasoning paths

- Build accepted propositions before questions.
- A path must declare `path_id`, `relationship_type`,
  `required_source_ids`, `input_claim_ids`, `operations`, and
  `output_claim_id`.
- Supported relationships include comparison, bridge, temporal transition,
  complementary procedure, exception-condition interaction, and cross-domain
  comparison.
- A cross-document record must express a real relationship. Two unrelated
  lookups joined into one answer are not multi-hop reasoning.
- For cross-document QA-CoT, a synthesis operation must cite exact evidence
  from both sources. Merely using source A in one lookup and source B in another
  lookup is insufficient.

### Validation and export

- Run deterministic checks before the independent judge.
- Reject changed numbers, changed modality, reversed dates, identical temporal
  states, unrelated subjects, unsupported currentness, missing temporal labels,
  and authority leakage.
- Every claim and reasoning step must remain bound to exact source evidence.
- No unjudged or deterministically failed record may enter a trainer export.
- Every planned request must end in an accepted, rejected, prompt-budget
  rejected, or model-failure audit state.
- Keep each temporal change lineage in one split. Hold out separate rule
  families for evaluation.
- Curator exports a curriculum manifest; provider training and model training
  remain outside this pipeline.

## Pilot-016 diagnosis

The last inspected attempt was partial:

- 44 single-document QA;
- 4 single-document QA-CoT;
- 0 cross-document QA;
- 1 cross-document QA-CoT;
- 49 canonical records.

The sole cross-document CoT was an unrelated compound lookup and is now
rejected by the stricter connectivity rule. The attempt also had invisible
post-retry omissions and reused stale cross-stage checkpoints. Both defects are
fixed in code, but a new smoke run is required to validate them against the
live model servers.

## Next commands

Load the existing secret-bearing environment locally:

```bash
set -a
source .env
set +a
```

Run a fresh bounded smoke test. Use a new run ID so the result is easy to audit:

```bash
.curator/bin/python pipelines/nrl_procurement/generate.py \
  --run-id pilot-017 \
  --limit 50 \
  --cross-document-limit 50 \
  --drafting-limit 2
```

Validate it:

```bash
.curator/bin/python pipelines/nrl_procurement/validate_run.py \
  --run-id pilot-017
```

Do not refresh or delete old pilot directories merely to make metrics look
clean. Checkpoint incompatibility should be handled by fingerprints and
contract versions.

After the smoke gates pass, run the user-controlled full data pilot with a new,
descriptive run ID. Use `--skip-drafting` if the immediate goal is only the
four core datasets; drafting can be run as a separate controlled stage.

## Smoke release gates

- Manifest status is `complete` for the requested core scope.
- All four core trainer exports are non-empty.
- Planned-request coverage has zero missing IDs.
- Every rejected record has explicit deterministic and/or judge reasons.
- No accepted record lacks a passing independent judge.
- Cross-document CoT includes genuine two-source synthesis.
- Exact-evidence and authority-leakage validation pass.
- Temporal outputs, when requested, expose historical/transition/target scope
  and lineage.
- Manually inspect at least:
  - 100 accepted records for a full pilot, or all records if fewer;
  - representative deterministic rejections;
  - every cross-document accepted record in a small smoke run;
  - all currentness and temporal records.
- Only after a clean data pilot should a separate controlled training
  experiment be used to compare curriculum schedules. Do not claim schedule
  benefits from generation statistics.

## Files to inspect first in a new session

- `pipelines/nrl_procurement/TASKS.md`
- `pipelines/nrl_procurement/PROMPTING_STANDARD.md`
- `pipelines/nrl_procurement/generate.py`
- `pipelines/nrl_procurement/validation.py`
- `pipelines/nrl_procurement/reasoning_paths.py`
- `pipelines/nrl_procurement/resume.py`
- `pipelines/nrl_procurement/validate_run.py`
- `outputs/<run-id>/files/manifest.json`
- `.curator_working/<run-id>/<stage>/<fingerprint>/<attempt>/failed_requests.jsonl`

Run the focused regression suite before further edits:

```bash
.curator/bin/ruff check \
  pipelines/nrl_procurement \
  tests/nrl_procurement \
  tests/unittests/test_litellm_online_request_processor.py

.curator/bin/pytest -q \
  tests/nrl_procurement/test_pipeline.py \
  tests/nrl_procurement/test_temporal.py \
  tests/unittests/test_litellm_online_request_processor.py
```

The current working tree contains intentional, uncommitted pipeline and test
changes. Inspect `git diff` before changing or committing them; do not discard
unrelated user work.
