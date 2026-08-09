# NRL Procurement Pipeline — Session Handoff

Last updated: 2026-08-08

## Current objective

Scope is **not** narrowed to single-document QA/QA-CoT any more. As of
2026-08-08, `config.yaml` has `path_qa.enabled`, `temporal.enabled`,
`propositions.enabled`, `reasoning_paths.enabled`, `drafting.enabled`, and
`cross_document.enabled` all set to `true` (`TASKS.md`'s "Comprehensive smoke
feature enablement" entry, same date) — every implemented task-type stage now
runs unless explicitly skipped at the CLI. `quality.required_task_types` is
still `[qa, qa_cot]`, so those two remain the only *release-gated* task types,
but every other stage generates and exports records if it isn't skipped.
`generate.py`'s argparse wiring currently only exposes `--skip-cross-document`
and `--skip-drafting` — there is still no CLI flag to independently disable
`path_qa`/`temporal`/`propositions`/`reasoning_paths` (open item; see
`--skip-propositions`/`--skip-temporal`/`--skip-path-qa`/`--skip-reasoning-paths`
in the audit remediation backlog). No full-corpus run has yet completed
successfully under this broadened config — the latest full-corpus validated
run (`qa-qacot-full-003`, see below) predates the 2026-08-08 re-broadening and
was produced with `--skip-cross-document --skip-drafting` under the earlier
narrower config. Read the live `config.yaml` `enabled:` flags before assuming
either scope; do not trust this paragraph's flag values without re-checking
`config.yaml`, since this is exactly the kind of claim that goes stale between
sessions.

## Authoritative inputs

- Dynamic registry: `/home/abhishek/curator/data/source/manuals.yaml` — 19
  manuals registered as of this update (`goods_2017`, `goods_2022`,
  `goods_2024`, `goods_om_2018_issuance`, `goods_om_2021_networth`,
  `goods_om_2022_l1_withdrawal`, `nrl_goods_rev1`,
  `nrl_consultancy_other_services_rev1`, `nrl_works_rev1`, `services_2017`,
  `services_2022`, `services_consultancy_2025`,
  `services_non_consultancy_2025`, `services_om_2021_startup_definition`,
  `works_2019`, `works_2022`, `works_2025`, `works_om_2022_corrigendum_pqb`,
  `works_om_2022_para763_certification`), spanning 3,006 total corpus chunks
  (`corpus_quality.json`'s `manuals`/`chunks` fields; verified unchanged
  across the two most recent runs). Six of these — the five one-to-two-page
  Office Memorandum manuals plus `services_om_2021_startup_definition` —
  currently produce **zero eligible generation chunks** each
  (`corpus.py::source_quality_issues`, independently reproduced against the
  live corpus this update: `goods_om_2018_issuance`, `goods_om_2021_networth`,
  `goods_om_2022_l1_withdrawal`, `services_om_2021_startup_definition`,
  `works_om_2022_corrigendum_pqb`, `works_om_2022_para763_certification` all
  show 0/N eligible chunks). This is a live front-matter-heuristic bug, not a
  stale doc claim — see the audit remediation backlog's data-corpus track for
  the fix; until it lands these six manuals are registered but silently
  contribute no records.
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

At this handoff (`GENERATION_PROFILE=glm`, `JUDGE_PROFILE=gemma_structured`
in `.env`; both route through the shared LiteLLM gateway at
`http://10.180.148.183:3005/v1` — Nemotron is **no longer** the active
generation profile and `gemma_thinking` is **no longer** the active judge
profile, both superseded on 2026-08-05/2026-08-07 per `TASKS.md`):

- Generation: `glm` profile, model `GLM-5.2-NVFP4-FP8`, JSON-schema structured
  output, deployment identity `glm-5.2-nvfp4-fp8-v1`.
  `generation_params.max_tokens` is **5000** (reduced from an earlier 8192 on
  2026-08-08 to avoid advertising a long tail to the shared scheduler); a
  separately checkpointed output-rescue pass retries truncations at a larger
  12,000-token ceiling. Thinking/reasoning is explicitly disabled
  (`chat_template_kwargs.enable_thinking: false`).
- Judge: the active `gemma_structured` profile uses `gemma-4-31b-it` with
  JSON-schema structured output, `temperature=1.0`, `top_k=64`, `top_p=0.95`,
  `max_tokens=2048`, thinking disabled, deployment identity
  `gemma-4-31b-it-10.180.148.183-8010-v1`. A `gemma_thinking` profile (same
  deployment, thinking enabled) also exists in `config.yaml` and was the prior
  default; `TASKS.md`'s 2026-08-08 same-deployment A/B found `gemma_structured`
  reproduces the same accept/reject decisions on a small sample far faster and
  without needing rescue, but this has not been validated on a
  human-labeled set (still open — see judge-calibration gate below). Judge
  output rescue retries missing decisions once at 4,096 tokens after a
  context-window preflight.
- Both profiles' request/judge concurrency is set to 45 for the current
  shared-gateway smoke; do not raise it without first reviewing measured
  timeout/yield evidence in `TASKS.md`.
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
   caps any repeated normalized 4-word opening n-gram to
   `quality.max_question_opener_share` (default 0.15) of the resulting pool,
   applied after `deduplicate()`, before judging, and again after judge
   attrition. Grounded in
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

## Fixes added 2026-08-02 after auditing `qa-qacot-full-002`

The historical output was independently measured at 84.68% literal
`According to` openers, 52.59% substantial normalized answer-in-evidence span
copying, 13.24% accepted QA-CoT, and 100% record-ID overlap between its old SFT
and eval files. The split fix above is present in current code, but the artifact
predates it. Additional safeguards now:

- calculate opener quotas against the resulting pool and re-apply them after
  judge attrition;
- preserve short exact labels/values while capping substantial extractive
  answers of four or more words at 35%, before and after judging;
- plan 40% QA-CoT requests and require at least 20% accepted QA-CoT for a
  `complete` portfolio;
- report answer-style distribution and portfolio removals, with removed rows
  retained in `qa_rejected.jsonl`;
- independently validate train-only SFT, non-train eval, zero ID overlap,
  split-safe cardinality reconciliation, and portfolio-quality status.

The 35% extractive ceiling, 40% planning allocation, and 20% accepted QA-CoT
floor are project policies for the next experiment, not literature-derived
universal optima. See the dated research record in `TASKS.md`.

Focused verification after these additions:

```text
ruff: passed
pytest: 108 passed, 3 dependency deprecation warnings
```

The LiteLLM pricing warning and PyArrow `null_placement` warning are
non-blocking. Structural validation errors, permanent request failures, missing
lineage, or empty core exports are blocking.

## Latest validated run: `qa-qacot-full-003`

`qa-qacot-full-002` (previously documented here) has been superseded.
`qa-qacot-full-003` (2026-08-02) is the most recent *substantial,
non-`failed`* run — full 3,006-chunk corpus, `--skip-cross-document
--skip-drafting`, run under the pre-2026-08-08 config where `path_qa`/
`temporal`/`propositions`/`reasoning_paths` were still disabled. Every
`outputs/*` directory timestamped after it (`quality-smoke-001`,
`quality-smoke-gemma-001`/`-002`, `saturation-500-001`,
`run-20260808T122041-362263Z`, `smoke-fast-20260808T131737Z`) is either a
`status: "failed"` run or a 2-to-30-record smoke fixture, not a full-corpus
result — check `outputs/*/files/manifest.json`'s `status`/`statistics.records`
directly before trusting any claim here about "the latest run," this changes
often:

- **1,572 accepted**: 1,048 `qa` + 524 `qa_cot` (33.3% CoT share).
- Splits (canonical): 1,159 train / 210 validation / 203 test.
- `reasoning_graphs_valid: 1572`, `reasoning_graphs_rejected: 11`.
- `leakage_audit_passed: true`; 15 of the 19 registered manuals are
  represented in accepted records (`leakage_audit.json`'s
  `unique_values.manual: 15`). The 4 unrepresented registered manuals are
  `goods_om_2018_issuance`, `goods_om_2021_networth`,
  `goods_om_2022_l1_withdrawal`, and `works_om_2022_corrigendum_pqb` — all
  four currently produce 0 eligible chunks (see Authoritative inputs above).
  Note: `services_om_2021_startup_definition` and
  `works_om_2022_para763_certification` *did* each contribute 1 accepted
  record in this run, but independently re-running today's corpus-eligibility
  check against the unchanged underlying corpus shows 0 eligible chunks for
  both as of this update — worth a closer look by whoever picks up the
  front-matter-heuristic fix, since it suggests eligibility for those two may
  have regressed since 2026-08-02 rather than being a static, unchanged bug.
- `status: "partial"`: `terminal_request_completeness.missing_judge_responses: 10`,
  and `quality_acceptance.portfolio_quality_complete: false` because
  `question_opener_share_complete: false` (`top_opener_share: 0.1508`,
  `top_opener: "according to the manual"` — above the now-configured 0.08
  ceiling; `max_question_opener_share` was lowered from 0.15 to 0.08 the same
  day, so this run may predate that specific tightening taking effect, the
  same caveat pattern documented for `qa-qacot-full-002`'s opener fix below).
- **Human review: 0/100 accepted, 0/25 rejected** — still the single largest
  remaining release blocker, unchanged since `qa-qacot-full-002`. Generate the
  sampling template with `review.py prepare <files_dir> <output.jsonl>`, then
  `review.py validate <reviewed.jsonl>` once real reviewer labels exist.
- Generated under the `GLM`/`gemma_structured` model contract described above
  is **not** accurate for this specific run — `qa-qacot-full-003` predates the
  2026-08-05/08-07 migration off Nemotron/`gemma_thinking`; check
  `outputs/qa-qacot-full-003/files/manifest.json`'s own `models` block for the
  models actually used for that run's records, don't assume today's contract
  applied retroactively.

Earlier runs (`pilot-021` n=5, `pilot-022` n=50, `qa-qacot-full-001` n=5,
`qa-qacot-full-002`) were smaller smoke tests or an earlier full-corpus
iteration validating fixes as they landed. Do not treat
`outputs/qa-qacot-full-003`'s files as final without both a fresh full run
under the current (broadened, GLM/Gemma) config and the still-pending human
review.

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
3. Check `answer_style_diversity.extractive_answer_share <= 0.35` and
   `quality_acceptance.qa_cot_share >= 0.20`.
4. Run `validate_run.py` and confirm zero training/eval ID overlap.
5. Generate and complete the human-review sample (`review.py`) — this gate
   is independent of code correctness and nothing this session touched it.
6. Only after that: decide whether to re-enable `path_qa`/`temporal`/
   cross-document generation for a second scope, or proceed to SLM
   fine-tuning / RAG work on the qa/qa_cot-only dataset as-is.

Do not refresh or delete old run directories merely to make metrics look
clean. Checkpoint incompatibility should be handled by fingerprints and
contract versions.

## Release gates

`qa` and `qa_cot` remain the only *required* task types
(`quality.required_task_types`) — see "Current objective" above for why the
other task-type stages are no longer disabled by default.

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
  configured cap is re-applied after judging.
- Substantial extractive answers remain at or below 35%, and accepted QA-CoT
  remains at or above 20%; both are explicit manifest release gates.
- 100 accepted + 25 rejected records human-reviewed via `review.py`.
- Manually inspect a representative sample of deterministic rejections in
  addition to the structured review above.

Cross-document/temporal/drafting-specific gates from the prior four-core-
export objective apply now, not just "whenever re-enabled" — those tracks
are currently `enabled: true` in `config.yaml` (see "Current objective"
above). See `TASKS.md` for that fuller gate list. Note also that drafting
records bypass the export-time leakage-audit/split-assignment gates entirely
as of this update (`drafting_accepted` is written straight to
`drafting.jsonl` without going through `assign_splits`/`leakage_audit`) —
this is an open, tracked gap, not a documentation error.

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
