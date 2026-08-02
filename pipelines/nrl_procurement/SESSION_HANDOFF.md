# NRL Procurement Pipeline — Session Handoff

Last updated: 2026-07-31

## Current objective

Scope is currently narrowed, deliberately, to **single-document QA and
QA-CoT only**. In `config.yaml`, `path_qa.enabled`, `temporal.enabled`,
`propositions.enabled`, and `reasoning_paths.enabled` are all `false`, and
`quality.required_task_types` is `[qa, qa_cot]`. Generation runs should also
pass `--skip-cross-document --skip-drafting` at the CLI. This is a scope
decision to ship one clean two-task-type dataset first, not a regression:
cross-document QA/QA-CoT, drafting, temporal, and path-derived
cross-document records are all still implemented and can be brought back by
flipping those config flags and dropping the CLI skip flags.

## Authoritative inputs

- Dynamic registry: `/home/abhishek/curator/data/source/manuals.yaml` — 16
  manuals registered as of this session (`goods_2017`, `goods_2022`,
  `goods_2024`, `goods_om_2022_l1_withdrawal`, `nrl_goods_rev1`,
  `nrl_consultancy_other_services_rev1`, `nrl_works_rev1`, `services_2017`,
  `services_2022`, `services_consultancy_2025`,
  `services_non_consultancy_2025`, `services_om_2021_startup_definition`,
  `works_2019`, `works_2022`, `works_2025`,
  `works_om_2022_para763_certification`), spanning 3,006 total corpus
  chunks.
- Original NRL PDFs:
  - `data/source/nrlManual_Procurement_of_Goods_Rev1.pdf`
  - `data/source/nrlManual_Procurement_of_Works_Rev1.pdf`
  - `data/source/nrlManual_Procurement_of_Consultancy_Other_Services_Rev1.pdf`
  Most of the 16 registered manuals are separately registered Government/PSU
  edition documents, not derived from these 3 base PDFs.
- Pipeline text input: Chandra OCR artifacts under `data/interim/ocr`

`manuals.yaml` will grow as documents arrive. Never hard-code the current
manual set. Preserve both identities:

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
  `generation_params.max_tokens` was raised from 4096 to **8192** this
  session after live path-answer generation hit `IncompleteOutputException`
  at the prior ceiling — Nemotron's context window is 131,072, so there is
  ample headroom. Path-answer generation itself is currently disabled by
  the qa/qa_cot-only scope above, but the raised ceiling also benefits plain
  `qa`/`qa_cot` generation.
- Judge: the active `gemma` profile uses `google/gemma-4-31B` with JSON-schema
  structured output, `temperature=1.0`, `top_k=64`, `top_p=0.95`, and
  `max_tokens=1024`. That ceiling is deliberately tight — the endpoint has an
  8,192-token *combined* prompt+completion limit — do not raise it without
  first checking judge prompt sizes.
- OCR: Chandra uses the private vLLM endpoint configured through ignored
  `OCR_MODEL`, `OCR_BASE_URL`, and `OCR_API_KEY` variables. The OCR command is
  `chandra`; source PDFs are written to `data/interim/ocr` and that OCR output,
  not direct PDF extraction, is what the generation pipeline consumes.

Endpoint relocation may reuse a checkpoint only when the explicit deployment
identity and semantic stage contract are unchanged. A model, prompt, schema,
validator, or stage-contract change must invalidate the dependent checkpoint.

## Fixes completed this session (2026-07-31)

Found via a manual audit of `pilot-020` plus two live full-scope runs
(`pilot-021`, `pilot-022`, `qa-qacot-full-002`). Full research records with
sources and rejected alternatives are in `TASKS.md` under matching dated
headers — read those before changing any of this code further.

1. **Citation offsets resolved against the wrong text.**
   `ProcurementGenerator.parse` (`generate.py`) computed `start_char`/
   `end_char` against `generation_passage` (image-line-stripped,
   whitespace-collapsed) instead of the registered `source_passage`, so
   citations did not reconstruct `source_chunk[start:end] == quote`. Fixed
   to match the pattern already used in `propositions.py`; a quote that
   cannot be located in the source chunk is now rejected
   (`citation_offset_unresolvable`) instead of exported with a wrong offset.
   Verified live: all citation spans in a 2,493-record run reconstruct
   exactly against the registered corpus.

2. **Leakage audit silently skipped single-document records.**
   `provenance.py::leakage_audit` only inspected `source_documents`
   (cross-document only) for its `manual`/`section` fields. A run with 35
   single-document records and 1 cross-document record reported only 2
   unique manuals and a false `passed: true`. Fixed with the same
   `manual_id` fallback already used throughout `export.py`.

3. **Manifest reporting bugs.** `pending_independent_judge` double-counted
   judge-rejected records as still pending instead of subtracting the
   terminal-complete `ablation_judged` count. `judged` request-coverage
   compared against every planned request instead of the judge-eligible
   subset, so deterministic rejections and dedup removals read as "missing
   judge responses." Both fixed; a shared `judge_eligible_planned` helper
   now serves both single- and cross-document coverage.

4. **Export crash on any reasoning-graph-invalid record.**
   `export.py::export_records` aborted the *entire* export — discarding
   every accepted record, not just the bad ones — if even one accepted
   record failed its post-hoc reasoning-graph connectivity check
   (`disconnected_claims`/`unused_source_claim`, usually a `qa_cot` record
   whose declared claim was never actually cited by a reasoning step). It
   now excludes only the invalid records (still logged to
   `reasoning_graph_rejected.jsonl`) and continues. This crashed a real
   full-corpus run once this session before the fix landed; the checkpoint/
   resume system meant the run resumed cleanly afterward with zero repeated
   model calls.

5. **Train/test split leakage in exported files.** `qa_sft.jsonl`,
   `qa_cot_sft.jsonl`, and `rag.jsonl` contained all three splits mixed
   together, and `eval.jsonl` was a full reformat of the same accepted
   pool — 100% of `qa_sft.jsonl`'s record_ids also appeared in
   `eval.jsonl`. The `split` field existed per-row but nothing partitioned
   the files themselves on it. Fixed: `*_sft.jsonl`/`rag.jsonl` now contain
   `train`-split records only; `eval.jsonl` contains `validation`+`test`
   only. `canonical.jsonl` still holds every split — it is the full audit
   record, not a training artifact. New per-file record counts in manifest
   statistics (`qa_sft_records`, `eval_records`, etc.) make a future
   mismatch visible without manually diffing record_ids by hand.

6. **Question-opener template collapse.** 84.7% of accepted questions in a
   live run began with the literal phrase "According to..." —
   `ProcurementGenerator.prompt`'s authority-disambiguation constraint gives
   the model no phrasing guidance across ~2,938 independent, stateless
   generation calls, so it converges on one default construction. Fixed at
   the corpus level: `validation.py::enforce_question_opener_diversity`
   caps any single normalized 4-word opening n-gram to
   `quality.max_question_opener_share` (default 0.15) of the deduplicated
   pool, applied right after `deduplicate()` and before judging. Grounded in
   Self-Instruct's (ACL 2023) pool-similarity filtering, generalized from
   full-text near-duplicates to shared opening templates. A first attempt
   (a within-one-response check) was tried and explicitly reverted as
   inadequate — it can only ever catch a same-call collision, a negligible
   fraction of a defect spanning thousands of independent calls. The prompt
   was also given a complementary source-side instruction (persona-voice
   phrasing; the identifying detail need not open the sentence).

Focused verification after all six fixes:

```text
ruff: passed
pytest: 105 passed, 3 dependency deprecation warnings
```

The LiteLLM pricing warning and PyArrow `null_placement` warning are
non-blocking. Structural validation errors, permanent request failures, missing
lineage, or empty core exports are blocking.

## Latest validated run: `qa-qacot-full-002`

Full 3,006-chunk corpus, `--skip-cross-document --skip-drafting`,
`path_qa`/`temporal`/`propositions`/`reasoning_paths` all disabled:

- **2,493 accepted**: 2,163 `qa` + 330 `qa_cot` (~13% CoT rate).
- Splits: 1,930 train / 305 validation / 258 test — independently verified
  zero record_id overlap between `qa_sft.jsonl`+`qa_cot_sft.jsonl` and
  `eval.jsonl`.
- All 3,567 citation spans across accepted records reconstruct exactly
  against the registered source chunks (re-verified independently, not just
  via the pipeline's own audit).
- `leakage_audit_passed: true`, all 16 manuals correctly represented.
- `status: "partial"` only because of `missing_judge_responses: 27` — real
  permanent Gemma judge failures (`IncompleteOutputException` truncations
  that exhausted their one retry), honestly recorded in `qa_rejected.jsonl`.
  Not a defect in the fixes above.
- **Human review: 0/100** — the single largest remaining release blocker,
  untouched by anything in this session. Generate the sampling template
  with `review.py prepare <files_dir> <output.jsonl>`, then
  `review.py validate <reviewed.jsonl>` once real reviewer labels exist.
- Fix #6 (opener diversity) and the raised `max_tokens=8192` were **not**
  yet in effect for this specific run — both landed after it started. The
  next full run will additionally reflect those two.

Earlier runs this session (`pilot-021` n=5, `pilot-022` n=50,
`qa-qacot-full-001` n=5) were smaller smoke tests validating individual
fixes as they landed; `qa-qacot-full-002` is the only full-corpus,
all-fixes-applied result so far — and even it predates the last two fixes.
Do not treat `outputs/qa-qacot-full-002`'s files as final without both a
fresh full run and the pending human review.

## Next commands

Load the existing secret-bearing environment locally:

```bash
set -a
source .env
set +a
```

Re-run at full scope with every fix (including the two that landed after
`qa-qacot-full-002` started) in effect — use a new, descriptive run ID:

```bash
.curator/bin/python pipelines/nrl_procurement/generate.py \
  --run-id <new-descriptive-id> \
  --skip-cross-document \
  --skip-drafting
```

This is a live, uncapped, full-corpus run (3,006 chunks) — expect roughly
the same order of magnitude as `qa-qacot-full-002` (~6,800 live model calls
across generation and judging, about 90 minutes wall time). Only the user
runs model-backed pilots; the agent must not run one unprompted.

After that run:

1. Verify citation offsets, leakage audit, and `missing_judge_responses`
   the same way as documented for `qa-qacot-full-002` above.
2. Check `question_opener_diversity` in manifest statistics — this is the
   first run where the corpus-level cap is actually active; confirm the
   top opener's share is at or below the configured 0.15 ceiling.
3. Generate and complete the human-review sample (`review.py`) — this gate
   is independent of code correctness and nothing this session touched it.
4. Only after that: decide whether to re-enable `path_qa`/`temporal`/
   cross-document generation for a second scope, or proceed to SLM
   fine-tuning / RAG work on the qa/qa_cot-only dataset as-is.

Do not refresh or delete old run directories merely to make metrics look
clean. Checkpoint incompatibility should be handled by fingerprints and
contract versions.

## Release gates (current qa/qa_cot-only scope)

- Manifest status `complete` requires zero `missing_judge_responses` in
  addition to the gates below. A real, if infrequent, judge-side failure
  rate has kept every run this session at `"partial"`. Whether that bar is
  right for this scope, or whether an occasional judge failure should
  become an accounted terminal state the way generation failures already
  are, is an open question this session did not resolve.
- Both `qa` and `qa_cot` non-empty (`required_task_types_complete: true`).
- Zero overlap between `qa_sft.jsonl`/`qa_cot_sft.jsonl` and `eval.jsonl`
  record_ids — mechanically fixed this session; still worth a spot-check
  on each new run.
- Every citation offset reconstructs exactly against the registered source
  chunk — fixed this session; spot-check a sample per run.
- `leakage_audit_passed: true` with the full manual count represented —
  fixed this session; the audit was previously a false pass.
- No single question-opener template should dominate — check
  `question_opener_diversity.top_opener_share` in manifest statistics; the
  configured cap is a ceiling on generation, not a guarantee about the
  final judged/deduplicated distribution.
- 100 accepted + 25 rejected records human-reviewed via `review.py`.
- Manually inspect a representative sample of deterministic rejections in
  addition to the structured review above.

Cross-document/temporal/drafting-specific gates from the prior four-core-
export objective still apply whenever those tracks are re-enabled; see
`TASKS.md` for that fuller gate list.

## Files to inspect first in a new session

- `pipelines/nrl_procurement/TASKS.md` — full research record for every fix
  above (with primary sources and rejected alternatives), plus the
  still-open broader backlog (unified CLI, best-of-N selection, independent
  task-classification judge, RAG distractor variants, etc.)
- `pipelines/nrl_procurement/generate.py` — orchestration, prompts,
  deterministic checks, manifest assembly
- `pipelines/nrl_procurement/export.py` — split-safe export, the
  reasoning-graph gate, leakage audit call, opener-diversity manifest stat
- `pipelines/nrl_procurement/validation.py` — deterministic checks,
  `deduplicate`, `enforce_question_opener_diversity`
- `pipelines/nrl_procurement/provenance.py` — `leakage_audit`,
  `build_reasoning_graph`
- `pipelines/nrl_procurement/review.py` — human-review template
  prepare/validate
- `config.yaml` — `quality.max_question_opener_share`; `path_qa`/
  `temporal`/`propositions`/`reasoning_paths.enabled` (all `false` for the
  current scope)
- `outputs/qa-qacot-full-002/files/manifest.json` — latest full-scope
  result (predates the last two fixes; see above)
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

Current baseline: 105 tests passing, Ruff clean. The working tree should be
clean at this handoff (every fix above was committed individually); run
`git log --oneline -10` to confirm before assuming otherwise.
