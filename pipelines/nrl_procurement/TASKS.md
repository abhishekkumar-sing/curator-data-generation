# NRL procurement synthetic-data tasks

This document tracks the work required to make the procurement QA pipeline
production-ready. Tasks are ordered by priority. Do not begin a large
generation run until every P0 item and its acceptance criteria are complete.

The design combines the reference project's strongest operational features
with stricter multi-source necessity checks:

- bounded, page-preserving source windows;
- complete provenance and forward lineage;
- deterministic and model-based verification;
- actual counterfactual source-ablation runs;
- separate QA and QA-with-auditable-rationale exports;
- explicit authority and temporal boundaries;
- leakage-safe evaluation.

## Mandatory research-first gate

This gate applies to **every capability, feature, fix, refactor, integration,
model, dependency, configuration behavior, and data-processing stage**. It is
not limited to large changes or tasks already listed in this roadmap.

Do not touch or edit production code for a capability until its research gate
has been completed and recorded. Documentation and task files may be edited
only to capture that research, its sources, conclusions, and the proposed
implementation. Similar code in the reference project is never sufficient
authorization to begin implementation.

For each individual capability, before reaching a design conclusion or
changing code:

- [ ] Inspect the relevant implementation in both
  `/home/abhishek/curator` and
  `/home/abhishek/nrl_curator_native_glm52`.
- [ ] Treat the reference project as an untrusted design input that may be
  incomplete, outdated, corrupt, or buggy.
- [ ] Research the capability in depth on the internet, even when a local
  implementation or reference implementation already exists.
- [ ] Check current official documentation and upstream source code for
  Curator and every dependency, external model, protocol, file format, API,
  and tool involved in that capability.
- [ ] Review relevant primary research papers and official benchmark
  specifications.
- [ ] Prefer primary sources such as papers, official repositories, and
  official documentation over blogs or unsourced summaries.
- [ ] Verify research claims against the actual source code and tests rather
  than assuming the published method matches a local implementation.
- [ ] Identify assumptions, conflicting evidence, unresolved questions, and
  facts that require empirical validation.
- [ ] Separate verified facts, research-supported conclusions, code-derived
  inferences, and hypotheses in the research report.
- [ ] Compare viable alternatives using explicit criteria such as grounding,
  faithfulness, coverage, diversity, leakage, cost, reproducibility, and
  operational risk.
- [ ] Present the findings and recommended design before making material code
  changes.
- [ ] When evidence is inconclusive, run a small controlled experiment or
  pilot instead of choosing an approach by intuition.
- [ ] Record the capability name, research questions, dated source links,
  findings, alternatives, risks, decision rationale, and proposed validation
  in the task or design document before editing production code.

Research-gate acceptance criteria:

- A separate research record exists for every capability being changed.
- Internet research is complete before the first production-code edit for
  that capability.
- The recommendation cites primary or official sources.
- Statements about either codebase are verified against concrete files,
  behavior, or tests.
- Known weaknesses in the chosen approach are recorded.
- Rejected alternatives and their tradeoffs are recorded.
- Any conclusion that depends on generated-data quality is labelled
  provisional until evaluated through a controlled pilot.
- No large generation run begins solely on the basis of schema validation or
  passing unit tests.

## P1 — Unified `nrl-curate` command-line interface

- [x] Complete and record the mandatory research-first gate for CLI design,
  saturation stopping, resumability, configuration precedence, and packaging.
  Compare the reference application's CLI with the current pipeline, current
  Curator APIs, Python packaging guidance, and established CLI conventions
  before changing production code.
- [ ] Register an `nrl-curate` project script in `pyproject.toml` and implement
  an `all` command that orchestrates preprocessing when requested, ordinary QA,
  QA with auditable rationale, cross-document QA, cross-document QA with
  auditable rationale, drafting, independent judging, validation, and export.
- [ ] Support a convenient command shape such as:

  ```bash
  nrl-curate all \
    --generation-profile glm \
    --judge-profile nemotron \
    --limit 200 \
    --cross-document-limit 200 \
    --drafting-limit 2 \
    --max-passes 0
  ```

- [ ] Keep model identities, endpoints, credentials, generation parameters,
  structured-output modes, concurrency, retry counts, and rate limits in
  `.env`/`config.yaml`. CLI model/profile switches select configuration; they
  must not require code changes or expose API keys in manifests or logs.
- [ ] Preserve independent generation and judge selection. If a shorthand such
  as `--model nemotron120b` is supported, define whether it selects generation
  only or a named paired profile; never silently use the same endpoint as its
  own judge when independent judging is required.
- [ ] Define `--max-passes` unambiguously: `0` means a full saturation run with
  no numeric pass cap; a positive integer caps the number of passes. A
  saturation run must stop only on a researched, configured convergence rule,
  not merely when one pass produces fewer records than requested.
- [ ] Make saturation termination safe and auditable: persist per-stage/pass
  state; measure novel accepted records after deterministic validation,
  deduplication, and independent judging; require the configured consecutive
  no-progress/low-yield condition; stop when the eligible source/planning space
  is exhausted; and record the exact stopping reason and pass metrics in the
  manifest. Resume must continue from persisted saturation state rather than
  reset the evidence window.
- [ ] Generate a safe dynamic run ID when `--run-id` is omitted, while allowing
  an explicit run ID for reproducible pilots and resumptions.
- [ ] Preserve local-only operation: Curator Viewer and telemetry remain off,
  source content goes only to configured private endpoints, caches remain under
  `.curator_working/<run-id>/<stage>`, and outputs remain under
  `outputs/<run-id>/files`.
- [ ] Make every substage resumable and make `all` fail with an auditable
  terminal manifest rather than silently omitting failed requests.
- [ ] Retain the existing direct Python entry point during migration or provide
  a documented compatibility path; CLI orchestration must call shared pipeline
  functions rather than duplicate generation logic.
- [ ] Add unit tests for argument/config precedence, profile switching,
  independent-judge enforcement, dynamic and explicit run IDs, pass bounds,
  resume behavior, local-only paths, exit codes, and secret redaction.
- [ ] Validate with a user-run bounded pilot before approving the CLI for a
  full `--limit 200` execution. Codex must not run model-backed pilots.

### Unified CLI research record (2026-07-28)

Status: research gate complete; implementation has not started. Each unchecked
CLI item above remains a separate implementation task and must be committed
after its own verification.

Questions researched:

- How should `nrl-curate` be packaged in this Poetry-based Curator repository?
- How can it preserve the existing direct Python command while calling shared
  pipeline logic rather than duplicating it?
- What precedence should apply among CLI flags, environment variables, and
  YAML configuration, including model profiles and secrets?
- What does Curator itself resume, and what additional state must a
  saturation controller persist?
- What evidence supports a convergence rule for synthetic-data novelty?
- Which parts of the reference CLI/saturation implementation are reusable,
  and which are coupled to its different pipeline?

Verified local findings:

- `pyproject.toml` uses Poetry and currently packages only
  `src/bespokelabs`. The NRL pipeline under `pipelines/nrl_procurement` is not
  an installed Python package. Adding a console-script string alone would
  therefore be unreliable in a built wheel.
- `generate.py` owns argument parsing and orchestration inside `main()`, while
  `settings.py` loads `.env`, `config.yaml`, and Curator privacy settings at
  import time. A CLI must first expose a callable orchestration boundary and
  explicit profile/config resolution; copying the current orchestration into a
  second file would create divergent behavior.
- The existing command already provides safe dynamic/explicit run IDs,
  project-local cache/output paths, private-endpoint enforcement, independent
  judge enforcement, and direct Python compatibility. These are invariants,
  not features to reimplement differently.
- Curator cache identity includes the input dataset, prompt function,
  response schema, model, batch mode, and generation parameters. A stable
  per-run/per-stage/per-pass `working_dir` enables request recovery, but
  Curator does not persist the procurement pipeline's accepted-novel set,
  judged outcome, pass counters, source exhaustion, or saturation reason.
- The reference CLI is an `argparse` wrapper around a shared `run()` function.
  Its saturation module atomically persists accepted rows, normalized novelty,
  empty counts, completed/quarantined parents, next pass, and stage
  statistics. It correctly distinguishes missing/invalid generations from
  genuine empty novelty. However, its stages and parent semantics differ from
  this pipeline, and its fuzzy text novelty alone is insufficient for
  procurement evidence/path saturation.

Research-supported conclusions and design:

- Use a standard installed console entry point and `argparse` subcommands. The
  `all` subcommand should produce an immutable invocation/options object and
  call the same orchestration function used by the legacy direct entry point.
- Make packaging real before registering the script: the CLI target and its
  imported pipeline modules must be included in both editable installs and
  wheels. Preserve `python pipelines/nrl_procurement/generate.py ...` during
  migration with a thin compatibility `main()`.
- Use explicit precedence `CLI flag > environment override > YAML profile >
  documented default`. API keys remain environment-only. A profile flag
  selects a named configuration block; it never accepts or prints a key.
- Treat generation and judge profiles as separate options. Reject identical
  resolved model/endpoint pairs when independent judging is required.
- Define `max_passes=0` as no numeric cap, never as one pass. A positive value
  caps total passes represented by persisted state, including resumed runs.
- Count novelty only after deterministic validation, near-duplicate removal,
  and independent judging. Require a configured number of consecutive
  zero/low-novelty observations per eligible planning unit. Missing,
  malformed, invalid-only, or judge-missing outputs are failures/quarantine,
  not saturation evidence.
- Persist state atomically after every pass with schema/code/config/source
  fingerprints, accepted IDs and novelty keys, planned/terminal request IDs,
  completed/exhausted/quarantined units, next pass, per-stage Curator
  statistics, and the exact stop reason. Resume must reject incompatible state
  rather than silently reset it.
- Source/planning-space exhaustion and statistical low yield are distinct stop
  reasons. The convergence thresholds remain provisional until a bounded
  user-run pilot measures yield and false saturation.

Alternatives rejected:

- Registering `pipelines.nrl_procurement.cli` without packaging the pipeline:
  may work from a checkout but is not a reliable console script in a wheel.
- Calling `generate.py` through a subprocess: preserves the old command but
  prevents clean typed configuration, status propagation, and shared
  orchestration tests.
- Copying the reference package wholesale: it contains valuable patterns but
  different stages, schemas, parent identities, and validity assumptions.
- Treating one empty or low-yield pass as saturation: model failure and strict
  filtering are observationally confounded with true exhaustion.
- Using Curator cache presence as pipeline completion: cached provider
  responses do not prove deterministic validation, judging, export, or
  terminal lineage completed.
- Allowing CLI-supplied endpoints/API keys: increases accidental disclosure
  through process listings, shell history, logs, and manifests.

Known risks and empirical questions:

- Packaging the current bare intra-pipeline imports requires a compatibility
  migration with tests for both module import and direct-script execution.
- A stable state fingerprint must be broad enough to prevent unsafe resume but
  narrow enough to preserve valid Curator cache reuse after unrelated changes.
- Similarity thresholds can collapse legitimate questions about different
  conditions; novelty should include source/path identity and not text alone.
- No paper or official Curator feature establishes the correct NRL saturation
  threshold. It must remain configurable and provisional until human-reviewed
  pilot evidence exists.

Planned validation:

- Build/install metadata inspection plus console-entry smoke tests without
  invoking a model.
- Unit tests for subcommands, precedence, profile resolution, secret
  redaction, independent-judge rejection, run IDs, pass bounds, atomic state,
  compatible/incompatible resume, failure exit codes, and legacy entry-point
  equivalence.
- Deterministic saturation-state tests covering new yield, duplicates,
  invalid-only output, missing output, consecutive zero novelty, source
  exhaustion, interruption, and resume.
- Only the user runs the bounded model-backed pilot.

Primary and official sources:

- [Poetry `pyproject.toml` scripts](https://python-poetry.org/docs/pyproject/#scripts)
  documents installed console scripts and requires reinstalling after script
  changes.
- [PyPA entry-point specification](https://packaging.python.org/en/latest/specifications/entry-points/)
  defines the ecosystem console-script object reference.
- [Python `argparse` documentation](https://docs.python.org/3/library/argparse.html#sub-commands)
  defines subparser-based command dispatch and parser exit behavior.
- [Curator automatic recovery and caching](https://docs.bespokelabs.ai/bespoke-curator/getting-started/automatic-recovery-and-caching)
  documents custom `working_dir`, partial recovery, and cache fingerprint
  inputs.
- [Curator API reference](https://docs.bespokelabs.ai/bespoke-curator/api-reference)
  defines the `CuratorResponse` dataset/statistics interface available to
  orchestration.
- [Self-Instruct](https://aclanthology.org/2023.acl-long.754/) uses iterative
  generation with invalid/similarity filtering, supporting novelty filtering
  but not a one-pass saturation claim.
- Reference code inspected:
  `/home/abhishek/nrl_curator_native_glm52/src/nrl_curator_native/cli.py` and
  `saturation.py`; both remain untrusted implementation inputs.

## Capability research — switch independent judge to Gemma 4

Status: researched and approved for configuration on 2026-07-28. Endpoint
behavior remains provisional until a user-run judge pilot completes.

- The configured endpoint identifies its served model as
  `/models/gemma4-12b`; it remains independent from the GLM generation
  endpoint.
- A read-only authenticated `/models` query on 2026-07-28 confirms that vLLM
  serves this endpoint with `max_model_len: 8192`. The official checkpoint's
  256K capability does not override the server's much smaller runtime limit.
- The official `google/gemma-4-12B` model card and generation configuration
  recommend `temperature=1.0`, `top_p=0.95`, and `top_k=64` across use cases.
- The official Gemma 4 Hugging Face model card demonstrates
  `apply_chat_template(..., enable_thinking=False)`. For an OpenAI-compatible
  vLLM endpoint, this template argument is carried in
  `extra_body.chat_template_kwargs`. Gemma therefore needs an explicit
  `enable_thinking: false` in its own profile; it must not inherit the setting
  accidentally from another model.
- The current configuration merges a selected model profile over role
  defaults. Supplying profile-specific `generation_params` therefore replaces
  the complete Nemotron-shaped payload without adding model-name conditionals
  to pipeline code.
- Keep the existing JSON-schema mode because that is the configured endpoint
  contract, but treat support as unverified until the exact private endpoint
  completes a controlled structured judge request. Do not silently fall back
  or expose unjudged records if it fails.
- Pilot-004's six-record Nemotron judge batch used 9,115 input tokens and would
  not fit this Gemma server before any output. Its one-record cross judge used
  5,483 input tokens, while individual drafting judges used 2,982 and 3,518.
  Configure one record per judge request and a 2,048-token output ceiling so
  the observed largest one-record shape fits below 8,192 with limited
  headroom.
- [x] Configure Gemma-specific sampling and explicit non-thinking template
  behavior from the official model card.
- [x] Select Gemma as the default and active independent judge.
- [x] Record the endpoint's verified 8,192-token runtime limit, reduce the
  output ceiling to 2,048, and use one record per judge request.
- [ ] Add a model-aware preflight that estimates the complete rendered prompt,
  response schema, and output reserve against the selected profile's actual
  server context. A fixed batch size is conservative but cannot prove every
  future source record fits.
- [ ] Validate schema compliance, exact witness behavior, latency, and judge
  calibration with a user-run pilot before production-scale generation.

## Capability research — pilot-003 quality recovery

Status: researched and approved for implementation on 2026-07-28. Conclusions
about generated-data quality remain provisional until pilot-004 is manually
reviewed. No production-scale run is approved.

Research questions:

- How should false-unanswerable examples be prevented?
- How should task/persona judgments avoid acquiescing to generator labels?
- How should cross-document records prove that both documents are necessary?
- How should ready-to-use drafting layout survive model and serving changes?
- How should every exported answer remain traceable to its source?

Sources consulted:

- [SQuAD 2.0 paper](https://arxiv.org/abs/1806.03822) (accessed
  2026-07-28): plausible unanswerable questions were created adversarially and
  systems must distinguish supported answers from abstention. Merely assigning
  an arbitrary paragraph an `answerable=false` target does not establish that
  the resulting question is unanswerable.
- [Challenges in Information-Seeking QA](https://arxiv.org/abs/2010.11915)
  (accessed 2026-07-28): paragraph selection and answerability prediction are
  separate sources of error; answerability remains difficult even with strong
  readers.
- [HotpotQA paper and official dataset site](https://hotpotqa.github.io/)
  (accessed 2026-07-28): genuine multi-hop questions are written from two
  related documents and carry sentence-level supporting facts. Both answer and
  supporting-fact quality are evaluated.
- [Pydantic JSON Schema documentation](https://docs.pydantic.dev/latest/concepts/json_schema/)
  (accessed 2026-07-28): nested typed lists are represented as JSON Schema
  arrays with model-defined items.
- [vLLM structured-output documentation](https://docs.vllm.ai/en/v0.15.0/features/structured_outputs/)
  (accessed 2026-07-28): the OpenAI-compatible server supports schema-guided
  JSON generation. A list of document blocks is therefore a portable structured
  contract; relying on a model to embed newline escapes inside one string is not.
- [Curator official repository](https://github.com/bespokelabsai/curator)
  (accessed 2026-07-28): `response_format` is the supported typed-output
  mechanism, and `parse` is responsible for converting that response into
  dataset rows.
- [W3C PROV-O Recommendation](https://www.w3.org/TR/prov-o/) (accessed
  2026-07-28): derivation and quotation provenance should link a derived entity
  to the entity it used or quoted.
- [HotpotQA official JSON format](https://github.com/hotpotqa/hotpot)
  (accessed 2026-07-28): supporting facts retain both a document identifier and
  a within-document sentence location rather than only an answer string.

Code and pilot verification:

- Pilot-003 assigned one randomly selected answer-bearing chunk an
  unanswerable contract. It produced two false abstentions. The exact source
  explicitly answers both questions at lines 8795–8801 of the 2025 consultancy
  manual, yet the judge marked both `supported=true` with score 5.
- The same judge accepted `task=nit_filling` for a comparison of negotiated
  offer validity and EMD forfeiture. A boolean `task_correct` lets the judge
  echo the proposed label instead of demonstrating classification.
- Four of five cross-document requests returned no parsed examples; the only
  accepted record used both exact source passages but had the wrong task. The
  existing source-ablation fields are model assertions, not independent
  executions.
- Both drafting generations contained the required facts but placed the entire
  document in a single string line and failed
  `draft_has_no_document_line_structure`.
- The reference project contains a separate `TaskClassificationJudge`, but it
  is treated only as an untrusted design input. Its useful principle is making
  the reviewer emit a label; its extra stage and rescue complexity are not
  copied.

Decision:

- [x] Stop forcing unanswerable generation from arbitrary source chunks.
  Disable unanswerable records until a dedicated adversarial construction and
  answerability-verification stage exists. This prefers missing coverage over
  mislabeled supervision.
- [x] Make each judge emit `recommended_task` and `recommended_persona` from
  the full canonical taxonomies. Acceptance requires exact agreement with the
  record, rather than trusting self-reported `task_correct` or
  `persona_correct`.
- [x] For unanswerable candidates, require the judge to return
  `answer_found_in_source` and an exact `answer_quote` when it finds one.
  Reject every abstention unless the judge explicitly finds no answer. This is
  defense in depth for imported or future adversarial records, not proof that
  absence has been exhaustively established.
- [x] Preserve exact evidence per material cross-document claim and require
  both source IDs. Generate exactly one record per bundle to reduce output
  ambiguity and prompt cost. Do not rescue an unrelated pair into a fake
  multi-hop item; zero output remains a tracked coverage failure.
- [x] Replace drafting's monolithic response string with typed non-empty
  `document_blocks`, then render blocks with deterministic newlines in `parse`.
  Keep evidence and tender-fact fields unchanged.
- [x] Add deterministic consistency checks for judge-selected taxonomy labels,
  answerability findings, exact found-answer quotes, block structure, and
  one-record cross-document cardinality.
- [x] Preserve existing seed citation IDs and add a stable structured
  `citations` array to every canonical and training-facing record. Each source
  citation must include manual/document identity, source file, page, section,
  chunk ID, and exact supporting quote where applicable. Tender-instance
  citations retain their tender ID and are distinguishable from manual-source
  citations.
- [x] Add regression tests reproducing pilot-003 failures.

Rejected alternatives and risks:

- Lexical overlap alone for answerability was rejected: paraphrases, negation,
  tables, and numeric equivalence make it unsafe.
- Keeping forced unanswerables and strengthening only the prompt was rejected:
  pilot-003 demonstrates that both generator and judge can agree on the same
  false label.
- Silently converting a false-unanswerable record to answerable was rejected:
  the abstention contains no supported answer and must be regenerated.
- Post-processing punctuation into drafting newlines was rejected: sentence
  boundaries do not reliably identify headings, fields, clauses, or signatures.
- A dedicated extra task-classification model stage is deferred because the
  same independent judge can emit an explicit selection in one call. Pilot-004
  must test whether this is sufficient.
- Source-removal judgments remain model-based and can overstate necessity.
  Exact two-source claim evidence and human pilot review remain mandatory.

## Capability research — prompt specification refactor

Status: researched and approved for implementation on 2026-07-28; output
quality remains provisional until fixed-case prompt evaluations and a manually
reviewed local-model pilot are complete.

Research question:

- How should all existing generation and judge prompts be rewritten as clear,
  grounded, model-portable specifications without changing their dataset
  schemas or weakening deterministic validation?

Sources consulted:

- [OpenAI prompt-engineering best practices](https://help.openai.com/en/articles/6654000-how-to-use-advanced-prompt-engineering)
  (accessed 2026-07-28): put clear instructions first, separate instructions
  from context with delimiters, specify the desired output, and use examples
  when zero-shot instructions are insufficient.
- [OpenAI model guidance](https://developers.openai.com/api/docs/guides/latest-model)
  (accessed 2026-07-28): leaner prompts can outperform accumulated instruction
  blocks; prompt changes should be evaluated rather than assumed beneficial.
- [Anthropic prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
  (accessed 2026-07-28): use clear, direct instructions, consistent structure,
  explicit context, and relevant examples for difficult formats.
- [Google Gemini prompt-design strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies)
  (accessed 2026-07-28): define inputs and constraints, partition complex
  prompts, keep few-shot formatting consistent, and iteratively test prompt
  variants. The exact model documentation controls long-context ordering and
  decoding recommendations.
- [Hugging Face structured-output guide](https://huggingface.co/docs/inference-providers/guides/structured-output)
  (accessed 2026-07-28): API-enforced JSON Schema is more reliable than asking
  for JSON only; prompt-only JSON remains a compatibility fallback.
- The repository's
  [`PROMPTING_STANDARD.md`](PROMPTING_STANDARD.md): the resulting SPEC-EVAL
  method requires a direct task, partitioned untrusted input, evidence policy,
  prioritized constraints, schema-backed output contract, observable final
  checks, selective examples, and a repeatable evaluation loop.

Code verification:

- Six pipeline prompts exist: single-document generation and judging in
  `generate.py`, cross-document generation and judging in `cross_stage.py`,
  and drafting generation and judging in `drafting.py`.
- All six already use typed Pydantic response formats, but their prompts do not
  consistently label source text as untrusted input, define every field's
  semantics, state cardinality requirements, define missing/conflicting-source
  behavior, or provide an observable final checklist.
- Deterministic validators enforce exact quotations, numeric support,
  qualification preservation, QA/rationale separation, two-source evidence,
  and drafting-specific number/email support. Prompt requirements should align
  exactly with those checks rather than introduce a second, inconsistent
  contract.
- Judge acceptance is computed in code. The prompts must define boolean and
  scoring semantics, but must not claim that a model judgment replaces
  deterministic checks.

Decision:

- [x] Rewrite all six prompts using consistent TASK, SOURCE POLICY,
  CONSTRAINTS, OUTPUT CONTRACT, UNTRUSTED INPUT, and FINAL CHECK sections.
- [x] Keep output shape in Pydantic schemas and structured-output transport;
  prompts explain semantics and relationships rather than embedding raw JSON
  examples.
- [x] Require exact evidence, explicit source authority, safe missing/conflict
  behavior, and complete one-to-one judge coverage.
- [x] Request concise auditable rationales only for rationale dataset variants;
  never request private hidden chain-of-thought.
- [x] Keep sampling, thinking controls, model names, endpoints, and structured
  output modes in configuration.
- [ ] Evaluate old and new prompts on a fixed representative set before a
  production-scale generation run.

Rejected alternatives:

- Adding a long persona to every prompt: rejected because it does not define
  observable correctness and adds tokens without resolving a domain ambiguity.
- Embedding hand-written JSON examples in every prompt: rejected because the
  response is already schema-constrained and examples can leak content or
  create false format conflicts. Add examples only in response to measured
  failures.
- Asking for unrestricted chain-of-thought: rejected because the training
  contract needs short, evidence-linked teaching rationales, not hidden model
  reasoning.
- Relying on prompt instructions alone: rejected because deterministic checks,
  schema validation, judge review, and human pilot review cover different
  failure modes.

## Capability research — Curator cache and per-run artifact layout

Status: researched and approved for implementation on 2026-07-28.

Research question:

- How should Curator recovery/cache files and generated artifacts be separated
  so cache state always remains under the repository while every execution has
  an auditable `outputs/<run-id>/files` directory?

Sources consulted:

- [Curator automatic recovery and caching](https://docs.bespokelabs.ai/bespoke-curator/getting-started/automatic-recovery-and-caching)
  (accessed 2026-07-28): Curator defaults to `~/.cache/curator`; callers can
  select a cache root through `CURATOR_CACHE_DIR` or pass `working_dir` when
  applying an LLM. The root contains `metadata.db` and fingerprinted run
  directories.
- [Curator upstream repository](https://github.com/bespokelabsai/curator)
  (accessed 2026-07-28): `CURATOR_CACHE_DIR` is the documented environment
  control; cache fingerprints cover the input dataset, prompt function, batch
  mode, response format, model, and generation parameters.

Code verification:

- `curator.LLM.__call__` accepts `working_dir`; Curator creates a fingerprinted
  child directory below it and stores request, response, metadata, and Arrow
  recovery files there.
- The pipeline currently passes
  `<output-dir>/.cache/{generation,judge,cross_generation,cross_judge,drafting_generation,drafting_judge}`.
  Therefore cache state is mixed with exported artifacts and changes location
  whenever the output directory changes.
- The current CLI writes directly to one `--output-dir` and has no run ID,
  allowing later runs to overwrite files from earlier runs.

Decision:

- [x] Configure the repository-relative cache root as `.curator_working`
  without embedding an absolute checkout path.
- [x] Give each run and pipeline stage an isolated child working directory,
  matching the cautious structure used by the reference project while leaving
  Curator's own fingerprinting intact below it.
- [x] Write every run only to `outputs/<run-id>/files`.
- [x] Accept an explicit safe `--run-id` and otherwise create a UTC run ID.
- [x] Reject path traversal, absolute run IDs, and reuse of a non-empty artifact
  directory.
- [x] Ignore both local cache state and generated run artifacts in Git.
- [ ] Add retention/cleanup policy only after storage requirements are known;
  never delete caches or outputs implicitly.

Rejected alternatives:

- Keeping cache below each output run: rejected because it duplicates recovery
  state, prevents stable reuse, and violates the requested separation.
- Using only `CURATOR_CACHE_DIR` while continuing to pass output-local
  `working_dir` values: rejected because the explicit argument takes precedence.
- Writing directly to `outputs/` without a run directory: rejected because files
  can be overwritten and provenance between executions becomes ambiguous.
- Putting cache inside the virtual environment `.curator`: rejected because
  environment replacement would destroy recovery state and the requested cache
  name is `.curator_working`.

## Capability research — work-conserving Curator online retries

Status: researched and approved for implementation on 2026-07-28.

Research question:

- Does the reference project's Curator retry monkey patch address a real
  throughput problem in the installed Curator 0.1.29, and if so, what is the
  safest way to apply the fix here?

Sources consulted:

- [Official Curator repository](https://github.com/bespokelabsai/curator)
  at tagged commit `461b4170` (`v0.1.29`, fetched 2026-07-28): the online
  processor still gathers every initial request before consuming its deferred
  retry queue.
- Official Curator source,
  `base_online_request_processor.py`: the retry-only loop acquires the
  concurrency semaphore before checking whether the queue contains a request,
  and retries do not begin until `asyncio.gather(*pending_requests)` completes.
- Reference project commit `d9364846`
  (`fix(slm-data/curator): retry a request inside its own task, not on a
  deferred queue`, inspected 2026-07-28): replaces deferred retries with an
  in-task retry loop and adds success/permanent-failure accounting tests.

Code and history verification:

- This checkout and the official `v0.1.29` source have the same deferred retry
  architecture. This is not merely a workaround for the reference project's
  older Curator 0.1.27 pin.
- A fast failure cannot retry while any unrelated first attempt is still
  running. The resulting tail is largest when request latency is variable.
- While only retry tasks remain, the dispatcher can acquire permits during
  queue-empty iterations before observing task completion. This temporarily
  removes otherwise available permits and can serialize the retry tail.
- The reference checkout is heavily dirty and its current compatibility file
  contains uncommitted changes. Only the committed patch and tests were treated
  as evidence.
- A controlled asynchronous simulation with 16 requests, one 0.40-second slow
  first attempt, 15 fast failures, and 0.10-second retries measured:
  deferred retries starting at 0.401 seconds and finishing at 0.502 seconds;
  in-task retries starting at 0.010 seconds and finishing at 0.401 seconds.
  This is a synthetic scheduler measurement, not an endpoint throughput claim.

Decision:

- [x] Use work-conserving in-task retries for online Curator requests.
- [x] Hold exactly one concurrency permit for a request across all its attempts
  and release it once on success or permanent failure.
- [x] Re-check request/token capacity and rate-limit cooldown before every
  additional attempt.
- [x] Preserve response caching, structured-response validation, permanent
  failure records, viewer cost projection, and status counters.
- [x] Return unused reserved token capacity after failed responses that report
  actual token use.
- [x] Implement the correction directly in this editable Curator source rather
  than monkey-patching private methods at runtime.
- [ ] Submit or track an upstream fix before rebasing onto a later Curator
  release; remove the local change once an equivalent tested fix is upstream.

Risks and validation:

- Immediate per-request retries can be less fair than putting failed requests
  behind all first attempts, and sustained endpoint failures can keep permits
  occupied. Curator's cooldown, RPM/TPM capacity checks, and retry bound remain
  mandatory.
- Unit tests must cover transient success, permanent failure, exact attempt
  counts, empty deferred queues, status counters, failure persistence, token
  capacity consumption, and exactly one semaphore release.
- The first local-model pilot should record retry counts and elapsed tail time;
  quality conclusions must not be inferred from scheduler tests.

Rejected alternatives:

- Copying the entire reference `compat.py`: rejected because it targets pinned
  Curator 0.1.27/LiteLLM 1.61.3 internals and bundles unrelated platform,
  serialization, credential, Arrow, and recovery patches.
- Runtime monkey-patching: rejected here because the Curator source is part of
  this editable repository; a normal source change is visible, type-checkable,
  and testable.
- Disabling retries: rejected because transient endpoint and schema failures
  are expected during large local-model runs.
- Increasing concurrency alone: rejected because it does not allow queued
  failures to retry before the initial barrier and does not correct permit
  acquisition when the retry queue is empty.

## Capability research — seed-driven grounded drafting

Status: researched and approved for implementation on 2026-07-28; generated
quality remains provisional until a controlled pilot is manually reviewed.

Research question:

- How should authored tender requests under `data/seeds` drive locally
  generated drafting records with the requested `id`, `tender_id`, `task`,
  `instruction`, `context`, `response`, and `citations` output contract?

Sources consulted:

- [Curator upstream repository](https://github.com/bespokelabsai/curator)
  (accessed 2026-07-28): structured Pydantic responses and explicit `prompt`
  and `parse` methods support a typed drafting stage.
- [vLLM structured-output documentation](https://docs.vllm.ai/en/latest/features/structured_outputs/)
  (accessed 2026-07-28): OpenAI-compatible vLLM servers can constrain output
  with a JSON schema.
- [LiteLLM upstream repository](https://github.com/BerriAI/litellm)
  (accessed 2026-07-28): `hosted_vllm` provides OpenAI-compatible completion
  transport, but capability lookup for a private served-model name can be
  absent from its static model registry.
- [NVIDIA Nemotron 3 Super NVFP4 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)
  and its
  [generation configuration](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/blob/main/generation_config.json)
  (accessed 2026-07-28): NVIDIA requires `temperature=1.0` and `top_p=0.95`
  for all tasks and serving backends.
- [OpenAI GPT-OSS model card](https://huggingface.co/openai/gpt-oss-120b),
  [OpenAI GPT-OSS announcement](https://openai.com/index/introducing-gpt-oss/),
  and [OpenAI GPT-OSS support guidance](https://help.openai.com/en/articles/11870455)
  (accessed 2026-07-28): GPT-OSS supports structured outputs and function
  calling, requires Harmony response formatting, and delegates exact feature
  support to the selected self-hosting runtime.
- [Google Gemma 4 function-calling guide](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4),
  [Gemma 4 prompt-format guide](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4),
  and [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
  (accessed 2026-07-28): Gemma 4 supports structured tool use through
  model-specific control tokens and chat templates. Google's Gemini JSON
  Schema API documentation is not evidence that every self-hosted Gemma
  generation supports OpenAI `response_format`.
- [Moonshot Kimi K3 model documentation](https://github.com/MoonshotAI/Kimi-K3/blob/main/README.md)
  and [Kimi Vendor Verifier](https://github.com/MoonshotAI/Kimi-Vendor-Verifier)
  (accessed 2026-07-28): Kimi K3 supports `response_format`, but always uses
  thinking, requires preserved reasoning history for multi-turn/tool use, and
  requires provider-specific feature verification.
- [Hugging Face Datasets JSON loading documentation](https://huggingface.co/docs/datasets/loading)
  (accessed 2026-07-28): JSON Lines is an appropriate one-record-per-line
  interchange format.
- [Self-Instruct](https://aclanthology.org/2023.acl-long.754/)
  (Wang et al., ACL 2023; accessed 2026-07-28): synthetic instruction
  pipelines require filtering of invalid and similar generations.
- [Source2Synth](https://openreview.net/forum?id=z9rR3btRFI)
  (Lupidi et al., ICLR 2025; accessed 2026-07-28): source-grounded synthetic
  generation benefits from answerability-based curation and explicit
  grounding in real source data.
- [Data Provenance Initiative](https://www.dataprovenance.org/)
  (accessed 2026-07-28): retain source and derivation metadata for post-training
  datasets.
- [Judging the Judges](https://arxiv.org/abs/2406.07791)
  (Shi et al., 2024; accessed 2026-07-28): LLM judges exhibit systematic
  biases, so a judge cannot replace deterministic checks or human pilot
  review.

Code verification:

- The reference seed file contains two authored requests with tender facts and
  `manual_chunk_ids`; it does not contain the final response.
- The reference `_build_drafting_inputs` resolves every chunk ID and fails on
  unknown IDs, while `TenderDraftingGenerator` uses Curator structured output.
- The reference compact exporter emits the requested seven core fields plus
  `evidence_quotes`.
- All five reference chunk IDs are absent from this repository's current
  corpus because the two projects use different chunk construction and ID
  formats. Copying the seed file unchanged would fail.
- A controlled request showed Curator rejects the private hosted-vLLM model
  before contacting it because LiteLLM's static `supports_response_schema`
  lookup returns false for the custom model name. This affects all existing
  structured QA stages as well as drafting.
- Direct endpoint probes showed GLM returns valid schema-constrained JSON with
  native `json_schema`; the current Nemotron server returns invalid `{"` for
  both `json_schema` and `json_object`, but valid JSON under prompt-only schema
  instructions. This is server behavior, not a claim that the underlying
  Nemotron model lacks structured-output training.
- A direct probe of the candidate Gemma 4 endpoint at the user-supplied private
  server verified its exact advertised model ID and successful responses for
  `json_schema`, `json_object`, prompt-only JSON, and forced tool calling.
  Native `json_schema` is preferred for that endpoint. Credentials are not
  recorded in this document.
- Official documentation confirms that model-family support is not enough to
  select a transport mode. GPT-OSS, Gemma, and Kimi have different chat
  templates, reasoning semantics, and runtime/provider requirements.
- The supplied Qwen2.5-Coder endpoint timed out at `/models` on 2026-07-28, so
  its deployment capabilities remain unverified. Official Qwen documentation
  confirms improved JSON generation and notes that vLLM tool calling requires
  launch-time `--enable-auto-tool-choice` and the Hermes tool parser.
- The example LD output is not fully grounded: the current NRL Goods source
  states a 5% cap on the value of **delayed goods**, not total goods, and the
  quoted example context does not support the added cancellation sentence.

Decision:

- [x] Add a separately configurable drafting track and seed path.
- [x] Preserve the requested seven-field compact JSONL contract.
- [x] Resolve seeds against this repository's own current chunk IDs and fail
  fast on missing or duplicate IDs.
- [x] Treat authored tender facts as instance-specific source material and
  manual chunks as governing policy; report conflicts instead of blending.
- [x] Require exact manual evidence quotes from the structured model response.
- [x] Reject unsupported numbers and email addresses deterministically before
  judging.
- [x] Keep drafting output separate from QA and QA-with-rationale exports.
- [x] Make the feature and seed path configurable without Python edits.
- [x] Add an explicit, opt-in hosted-model structured-output capability setting
  rather than globally monkey-patching Curator or assuming every endpoint
  supports JSON Schema.
- [x] Support configurable `auto`, `tools`, `json_schema`, `json`, and
  prompt-validated `md_json` modes without model-name conditionals.
- [ ] Add a mandatory per-endpoint structure probe before a newly configured
  model is allowed to generate production data; do not infer support from a
  model-family name.
- [ ] Run a small local-model pilot and manually review grounding, omissions,
  authority, privacy, and drafting usefulness before scaling.

Rejected alternatives:

- Copying the reference implementation verbatim: rejected because its IDs do
  not resolve here and its sample demonstrates unsupported clause expansion.
- Exporting the displayed examples as fixed gold responses: rejected because
  one example changes the LD basis and fixed rows would not constitute
  synthetic generation.
- Trusting only an LLM judge: rejected because judge bias is documented and
  exact numbers, emails, citations, and evidence membership are
  deterministically testable.
- Mixing drafting rows into QA exports: rejected because their schemas and
  intended training behavior differ.

### Drafting citation/detail integrity research addendum (2026-07-29)

Status: implemented and locally verified; next user-run pilot pending.

Observed failure and verified cause:

- Both accepted pilot-009 drafting records contain citation IDs with no
  corresponding `citation_details` entry. For example, the NIT row lists the
  page-72 chunk although its details resolve only page-94 evidence and the
  tender seed.
- `build_drafting_inputs()` currently labels every seed-provided manual chunk
  plus the tender ID as `citations`. `TenderDraftingGenerator.parse()` instead
  creates details only for exact evidence quotations the model declares.
  Candidate context and used/quoted provenance are therefore being conflated.
- The reference project also exports seed/input citation IDs without a
  one-to-one detail structure. It does not solve traceability and is not a safe
  implementation to copy.

Research-supported design:

- W3C PROV distinguishes general derivation/usage from quotation through
  `prov:wasQuotedFrom`; provenance relations should identify the actual source
  entity involved rather than imply derivation from every available entity:
  https://www.w3.org/TR/prov-o/
- Work on citation-generating QA evaluates citation correctness/precision
  separately from answer correctness. A citation is not justified merely
  because its document was available in context:
  https://aclanthology.org/2024.acl-long.641/
- Multi-source attribution research likewise treats claims and their supporting
  citations as explicit mappings:
  https://aclanthology.org/2024.naacl-long.216/

Decision:

- Treat `manual_chunk_ids` as candidate/source-input lineage and retain it in
  canonical audit rows, but do not automatically present every candidate as an
  answer citation.
- Derive exported `citations` in first-use order from resolved
  `citation_details`, followed by the tender-seed detail. Every exported
  citation ID must have at least one matching detail, and every detail ID must
  appear in `citations`.
- Permit repeated details for multiple distinct quotations from the same chunk
  while keeping the flat citation ID list unique and ordered.
- Fail deterministic validation on unresolved evidence, dangling citation IDs,
  detail IDs absent from the flat list, or missing tender-seed provenance.
  Never silently synthesize a quote/detail for an unused candidate chunk.
- Keep `citations` as the final compact-output field.

Risks and alternatives:

- Requiring every candidate chunk to be quoted would force irrelevant
  citations and reward the model for copying unused context.
- Dropping `manual_chunk_ids` entirely would lose input lineage needed for
  reproducibility; retain it in full audit/canonical records.
- One-to-one ID/detail integrity proves traceability, not that every material
  response clause is cited. Atomic block-to-evidence completeness remains a
  separate pending task.

Validation plan:

- Add tests for unused candidate chunks, repeated quotations from one chunk,
  unique stable citation ordering, missing tender detail, and dangling IDs.
- Re-audit pilot-009 locally and confirm its compact drafting rows would no
  longer expose unresolvable citation IDs.

Implementation result:

- [x] Renamed seed-derived citation lineage to
  `candidate_citation_ids`; it remains available in generation/canonical audit
  rows but is not automatically exported as claimed evidence.
- [x] Derive exported citation IDs from resolved manual evidence details in
  first-use order, followed by the tender-seed provenance detail.
- [x] Added bidirectional integrity checks for dangling flat IDs, unlisted
  details, duplicate flat IDs, unresolved evidence quotations, and missing or
  duplicate tender-seed provenance.
- [x] Preserve repeated quotation details from one chunk while deduplicating
  only the flat citation ID list.
- [x] Re-audited pilot-009 locally: the page-72 NIT citation and page-149 LD
  citation are correctly identified as candidate-only dangling citations and
  would not be exported after this fix.
- [x] Passed 50 focused procurement tests and Ruff checks.
- [ ] Validate regenerated drafting outputs and citation/detail completeness in
  the next bounded user-run pilot.

### Atomic drafting block attribution research addendum (2026-07-29)

Status: implemented and locally verified; user-run pilot validation pending.

Research and local findings:

- ALCE evaluates citation quality separately from correctness and includes
  citation completeness/recall: answer content must be attributable, not merely
  accompanied by a document-level citation:
  https://github.com/princeton-nlp/ALCE and
  https://aclanthology.org/2023.emnlp-main.398/
- MultiAttr represents multi-source attribution with citations associated with
  individual answer sentences and supports multiple citations per sentence:
  https://aclanthology.org/2024.naacl-long.216/
- Atomic claim generation is useful because complex sentences can mix
  supported and unsupported content that a document-level decision masks:
  https://aclanthology.org/2022.acl-long.175/
- The current drafting schema has ordered text blocks but only document-level
  evidence/fact lists. Pilot-009 demonstrates the masking failure: an unrelated
  `shall` in one supporting quotation can hide a `may`→`shall` change in a
  different block when all evidence is concatenated.
- The reference project likewise has document-level citations/evidence and
  offers no block-to-source binding to reuse.

Decision:

- Extend every `DraftingBlock` with exact, per-block
  `manual_evidence_quotes`, `tender_facts_used`, and
  `instruction_evidence_quotes`. Preserve the existing document-level lists as
  backward-compatible aggregates and require exact equality with the stable
  first-use union of block bindings.
- Every block must cite at least one exact support item. Layout/headings may
  cite an exact instruction substring; policy paragraphs must cite their
  governing manual quotation; instance fields/contacts must cite complete
  tender facts. A source type may be empty only when another type supports the
  whole block.
- Run number, email, authority, absence, and deontic-modality checks at block
  scope against that block's bound support. Retain document-level checks as a
  defense in depth.
- Resolve citation details from the aggregated exact manual quotes and tender
  facts after block validation. Do not expose instruction substrings as source
  citations: they establish task/layout intent, not external factual
  provenance.
- Bump the drafting structured-output contract through its schema change so
  Curator does not reuse incompatible parsed responses.

Rejected alternatives and risks:

- Lexical post-hoc matching cannot reliably map paraphrased legal text to the
  correct quotation and can recreate the masking bug.
- Sentence splitting after generation loses authored block structure and is
  brittle for numbered clauses and labelled fields.
- Removing the document-level lists would break existing audit consumers; keep
  them as validated aggregates during migration.
- Model-authored support bindings can still be wrong. Exact membership,
  aggregate equality, deterministic semantic checks, independent judging, and
  human pilot review remain required.

Validation plan:

- Add regressions for aggregate mismatch, unsupported blocks, wrong source
  type, block-local modality drift, exact instruction support, stable first-use
  aggregation, and preservation of the compact output contract.
- Do not accept the capability until a user-run bounded pilot shows that both
  drafting seeds produce complete block bindings without material quality
  regression.

Implementation result:

- [x] Extended `DraftingBlock` without removing its existing `text` field:
  block-local exact manual quotations, complete tender facts, and exact
  instruction substrings are now retained.
- [x] Production drafting inputs require block attribution. Every block must
  bind at least one source item; manual, tender, and instruction bindings are
  checked against their correct source namespace.
- [x] Require document-level manual evidence and tender-fact lists to equal the
  stable first-use union of block bindings, preserving the earlier fields
  without allowing an unrelated aggregate.
- [x] Apply absence and deontic-modality checks independently to each block and
  its own support while retaining document-level defense-in-depth checks.
- [x] Updated the structured prompt contract to explain source roles and forbid
  instruction text from substituting for policy or tender-fact evidence.
- [x] Preserved compact drafting field order and citation behavior.
- [x] Passed 50 focused procurement tests and Ruff checks, including a
  block-local `may`→`shall` regression.
- [ ] Validate block-attribution completeness, generation yield, and drafting
  usefulness with both seeds in the next bounded user-run pilot.

## Current baseline

Implemented:

- [x] Manifest-controlled ingestion from `data/source/manuals.yaml`
- [x] Chandra OCR output ingestion with page provenance
- [x] Configurable local generation, judge, and OCR models
- [x] Curator local-only enforcement
- [x] Single-document QA and QA-with-rationale generation
- [x] Explicit cross-document manual relationships
- [x] Cross-document QA and QA-with-rationale schemas
- [x] Exact, source-specific evidence checks
- [x] Model-reported source-ablation judgment
- [x] Near-duplicate filtering
- [x] Canonical, SFT, RAG, and evaluation exports

Known limitations:

- Cross-document candidates use one chunk per source.
- Lexical overlap may favor copied text or generic procurement language.
- The source-ablation result is currently predicted by the judge; separate
  A-only and B-only answer attempts are not executed.
- Reasoning steps are not explicitly linked to input and output claims.
- Rejected candidates are not written to stage-specific audit files.
- Cross-manual relationships may collapse most records into one split.
- No statistically meaningful generated pilot has been manually evaluated.

## P0 — required before full generation

## Parallel capability — temporal curriculum and dataset annealing

This capability is tracked separately from ordinary cross-document QA. It
produces temporal training artifacts and a trainer-readable curriculum; Curator
generates and validates the records but does not itself perform SFT sampling or
claim that a schedule improves a model without a controlled training
experiment.

### Dataset-annealing research record (2026-07-29)

Status: initial research gate complete; no production implementation has
started. The effectiveness of any schedule remains an empirical hypothesis.

Questions researched:

- Does published evidence support dynamically shifting historical,
  transition, and target examples during SFT?
- How should evolving procurement facts be represented without overwriting
  historical truth or projecting later rules backward?
- Which work belongs in Curator data generation versus the downstream trainer?
- Is the supplied manifest sufficient to infer amendment lineage and
  currentness?
- Which parts of the supplied/reference implementation are safe to reuse?

Verified findings:

- Data-mixture optimization for SFT is an active research area. Li et al.
  optimize static mixture weights against validation loss and explicitly
  describe SFT mixture optimization as underexplored; this does not validate a
  universal linear or polynomial decay schedule for old facts.
- “Annealing” in foundation-model training often refers to a final,
  lower-learning-rate phase using a curated high-quality mixture. That usage
  does not by itself show that a historical-to-current three-phase SFT
  curriculum prevents catastrophic forgetting.
- Continual Knowledge Learning evaluates three separate objectives: retain
  invariant/older knowledge, update outdated knowledge, and acquire new
  knowledge. Its results show that reliably achieving all three remains
  difficult.
- Temporal Knowledge Editing reports that direct updates can preserve new
  knowledge while catastrophically forgetting historical facts. Its METO
  method trains historical and new knowledge together with explicit time
  objectives. This supports time-scoped contrast records, not an unscoped
  “overwrite old with current” target.
- The official Department of Expenditure manuals page currently lists Goods
  Second Edition 2024, Consultancy and Non-Consultancy Services 2025, and Works
  Second Edition 2025. Official publication listing is evidence of document
  identity/date, but it does not establish that no later OM modifies a
  provision. Currentness must be evaluated at a declared cutoff using the
  complete registered amendment set.
- The supplied manifest preserves useful manual IDs, dates, authority, and some
  `amends` edges. It is not yet a complete amendment graph: several documents
  whose titles describe an amendment/corrigendum have no `amends` field.
  Titles and filename patterns cannot safely create those edges.
- The reference project contains valuable safeguards: configured historical
  and target pairs, bounded section alignment, exact evidence on both states,
  time-labelled questions/answers, phase-specific exports, a validated optional
  sampling schedule, and tests rejecting unrelated or identical states.
- The reference alignment is still heuristic. Fuzzy heading/content similarity
  proposes candidates but does not prove that two passages express different
  states of the same rule. One-to-one greedy alignment may also miss
  one-to-many amendments.

Corrections to the supplied proposal:

- Do not omit years/dates from the user query. Both visible question and answer
  must identify historical and target editions/as-of dates, or an exported
  record can teach temporal leakage.
- Do not automatically call the later side “current” or “active.” Use
  `historical_state` and `target_state` until official lineage and a declared
  cutoff establish current applicability.
- Do not generate an explanation of *why* a rule changed unless the source
  explicitly states the reason. A safe transition explanation describes what
  differs and the documented amendment relation.
- Do not use `gpt-4o-mini`, OpenAI Batch, or any public Curator service. All
  source passages stay on configured private endpoints, with Viewer and
  telemetry disabled.
- Curator `batch=True` is an inference-provider batching option, not a training
  data scheduler. This repository should export a schedule manifest; the
  downstream trainer must apply and log actual step-wise weights.
- Generation temperature is unrelated to “temperature-based” dataset sampling.
  Model decoding parameters remain profile-specific; sampling temperature or
  phase weights belong to the trainer curriculum.
- Do not rely on a paragraph-number regex over an entire converted manual.
  Numbered lists, OCR spacing, Markdown emphasis, tables, repeated headings,
  and cross-page provisions make that extraction brittle. Use registered
  page/chunk/section structures, exact evidence offsets, and bounded
  one-to-many windows.
- Never pass an entire OM merely because a paragraph lookup failed. Quarantine
  unresolved alignments or use bounded candidate windows with explicit
  provenance.

Research-supported architecture:

1. **Authoritative temporal graph**
   - Verify each edition and OM against the official source.
   - Record explicit `amends`, `supersedes`, `effective_from`,
     `effective_until`, and `verification_cutoff` only when documented.
   - Keep Government and NRL graphs separate; no Government-to-NRL adoption
     edge is inferred from similar language.
2. **Verified state alignment**
   - Reuse accepted atomic propositions and multi-chunk windows.
   - Align candidates using section/subject/action signatures, then require
     exact evidence and an independent same-provision/change judgment.
   - Support one-to-many and many-to-one changes; identical states are not
     transition examples.
3. **Three exported record families**
   - `historical_context`: historical-only QA with explicit historical
     edition/as-of scope.
   - `temporal_transition`: both dated states, exact evidence from each,
     explicit change type, and no unsupported causal explanation.
   - `target_context`: target-state QA explicitly scoped to its edition/as-of
     date; “current as of cutoff” is allowed only after verified lineage.
4. **Trainer-readable curriculum**
   - Export immutable phase tags, record IDs, source-group IDs, and a validated
     piecewise schedule whose non-negative weights sum to one at every anchor.
   - Do not invent a default decay exponent. Candidate schedules are
     experiment configurations, not truth.
   - Preserve a nonzero replay/retention component unless experiments show it
     is unnecessary.
5. **Evaluation before adoption**
   - Compare uniform mixing against at least one staged curriculum using the
     same base checkpoint, total examples/tokens, optimizer, learning-rate
     schedule, seed set, and evaluation suite.
   - Measure historical retention, target-state accuracy, temporal
     disambiguation, invariant procurement reasoning, authority isolation, and
     general capability regression.
   - Evaluate unscoped prompts separately from explicitly dated prompts.
     Report confidence intervals across seeds before claiming an advantage.

Implementation backlog:

- [ ] Verify and complete amendment/currentness metadata against official
  Department of Expenditure and NRL sources; record a verification cutoff.
- [ ] Add typed temporal-pair and sampling-schedule configuration with strict
  validation and secret-free fingerprints.
- [ ] Build bounded one-to-many temporal alignments from accepted propositions
  and section windows; write candidate and rejected alignment audits.
- [ ] Add a source-grounded change extractor that emits historical/target
  proposition IDs, exact evidence, change type, and explicit lineage basis.
- [ ] Generate separately validated historical, transition, and target QA/CoT
  exports with visible time and authority scope.
- [ ] Export a trainer curriculum manifest; do not implement provider batching
  or model training inside Curator.
- [ ] Add leakage-safe temporal splits that keep a change lineage together
  while holding out separate rule families for evaluation.
- [ ] Add deterministic and judge checks for identical states, unrelated
  subjects, reversed dates, unsupported currentness/supersession, missing
  temporal labels, changed numbers/modalities, and NRL/Government leakage.
- [ ] Run a user-controlled data pilot, followed by a separate controlled
  training experiment before selecting or claiming benefits from a schedule.

Primary and official sources:

- Li et al.,
  [Data Mixing Optimization for Supervised Fine-Tuning of Large Language
  Models](https://proceedings.mlr.press/v267/li25bh.html), treats SFT mixture
  selection as an optimization problem and reports that the area remains
  underexplored.
- Jang et al.,
  [Towards Continual Knowledge Learning of Language Models](https://openreview.net/forum?id=vfsRB5MImo9),
  evaluates retention, updating, and acquisition under temporal knowledge
  change.
- Yin et al.,
  [History Matters: Temporal Knowledge Editing in Large Language
  Model](https://arxiv.org/abs/2312.05497), documents historical forgetting and
  uses explicit time objectives for old and new knowledge.
- Dhingra et al.,
  [Time-Aware Language Models as Temporal Knowledge
  Bases](https://aclanthology.org/2022.tacl-1.15/), studies time-conditioned
  language models and facts that change over time.
- [Department of Expenditure manuals](https://www.doe.gov.in/manuals) is the
  primary publication index for the registered Government procurement manual
  editions.
- [Bespoke Curator](https://github.com/bespokelabsai/curator) documents local
  structured generation and batching boundaries; it does not provide a
  temporal SFT curriculum or evidence for a particular schedule.

### 1. Extract grounded atomic propositions

- [x] Add a proposition schema containing:
  - proposition ID;
  - subject;
  - authority;
  - action;
  - object;
  - modality;
  - conditions;
  - exceptions;
  - threshold/value/unit;
  - temporal scope;
  - exact evidence reference.
- [x] Extract propositions independently for each source window.
- [x] Reject propositions whose complete factual content is not supported by
  exact evidence.
- [x] Cache propositions by source hash, chunk ID, model profile, and schema
  version.

Acceptance criteria:

- Every proposition resolves to one registered source and exact offsets.
- Numbers, dates, names, and modalities match the evidence.
- Government propositions cannot be labelled as NRL policy.
- Re-running unchanged sources reuses the cached proposition set.

#### Grounded atomic-proposition research record (2026-07-29)

Status: research and deterministic implementation complete; model-backed
validation remains a user-run gate.

Questions researched:

- What should “atomic” mean for procurement rules containing modality,
  negation, conditions, exceptions, thresholds, and attributed authority?
- Should propositions be decontextualized, and how can that avoid adding facts?
- What evidence identity and validation are needed before propositions can
  support cross-document paths?
- Which inputs must invalidate a cached extraction?
- What does the reference project implement, and what remains missing?

Verified local and reference findings:

- The current pipeline generates QA claims inside `CrossDocumentGenerator`.
  Those claims have stable per-record IDs and exact source-specific evidence,
  but they are downstream of pair selection and question generation. They are
  not reusable source propositions and cannot establish a verified reasoning
  path before a question is written.
- Current corpus chunks already provide manual ID/title, issuer, policy scope,
  revision/as-of dates, source hash/file, chunk ID, page, section, and passage.
  Evidence locations can therefore be resolved deterministically without
  asking a model to invent provenance.
- The reference project similarly creates claim IDs while parsing generated
  cross-document QA. It validates source-specific quotes and links rationale
  steps to claims, but it has no independent proposition extraction stage or
  proposition cache. Copying it would preserve the same ordering problem.
- FActScore and SAFE decompose text into independently verifiable facts.
  FActScore notes assumptions about non-conflicting knowledge sources that do
  not hold across procurement authorities and editions. Authority and temporal
  scope must therefore be part of each proposition rather than inferred later.
- Graphene represents a core proposition plus linked contextual information;
  MinIE separately annotates polarity and modality. These are better fits than
  a bare triple for rules whose legal force changes with `shall`, `may`,
  negation, conditions, or exceptions.
- SAFE-style self-contained rewriting can reduce ambiguous references but may
  introduce or omit information. A model-authored self-contained statement
  cannot replace the exact evidence span as the canonical truth source.

Research-supported design:

- Extract propositions independently from one registered source window at a
  time. The model receives source metadata as immutable context and emits
  semantic fields plus one exact evidence quote; it never authors IDs, hashes,
  offsets, issuer, scope, or dates.
- Represent the core event as `subject`, `action`, and `object`, with separate
  `authority`, `modality`, `polarity`, `conditions`, `exceptions`,
  threshold/value/unit, and temporal scope. Use explicit enum values plus
  source-language text where normalization could lose meaning.
- Derive proposition IDs from a versioned canonical serialization of source
  identity, exact evidence location, and semantic fields. Resolve the quote to
  its registered chunk and offsets deterministically. Reject ambiguous
  duplicate quote locations unless the model supplies a valid occurrence or
  the location is otherwise uniquely resolvable.
- Treat the exact quote and registered metadata as authoritative. The concise
  proposition statement is an indexable representation, not evidence.
- Validate every numeric/date/name literal and every material modality,
  polarity, condition, and exception against the evidence. Deterministic checks
  handle exact membership, offsets, metadata authority, and literal integrity;
  an independent judge handles semantic completeness. Fail closed on
  disagreement.
- Cache a proposition batch under a fingerprint containing source SHA-256,
  chunk/window IDs and content hashes, extraction profile/model/endpoint
  identity excluding credentials, decoding parameters, structured-output mode,
  prompt hash, response-schema version/hash, and validator version. Write
  atomically and retain rejected/raw lineage separately.

Alternatives rejected:

- Generate propositions jointly from two manuals: risks authority leakage and
  makes source-specific validation ambiguous.
- Use only subject–predicate–object triples: loses conditions, exceptions,
  modality, and temporal applicability.
- Let the model emit provenance or proposition IDs: permits fabricated
  authority/offsets and makes cache identity unstable.
- Automatically rewrite evidence into self-contained text: decontextualization
  may introduce meaning not present in the quoted source.
- Cache only by chunk ID: fails to invalidate on source, schema, prompt,
  validator, model, or decoding changes.
- Use the existing downstream QA claim list as the proposition store: makes
  planning depend on a question that has already been generated and preserves
  circular verification.

Known risks and proposed validation:

- Coordinated clauses may need multiple propositions sharing one evidence
  span; atomicity cannot be proven by schema alone. Add fixtures for conjunction,
  exception, conditional, threshold, definition, cross-reference, and table
  rows.
- Exact duplicate sentences can occur within a window. Offset resolution must
  reject ambiguity rather than silently select the first occurrence.
- Model extraction may omit contextual qualifiers even when every returned
  field is individually grounded. Independent completeness judging and a
  user-reviewed pilot are required before using propositions for path planning.
- Cache hits prove deterministic input identity, not proposition quality.
  Cache metadata must preserve validator/judge status and code fingerprints.
- Codex will test schema, validation, cache invalidation, and local
  orchestration without model calls. Only the user runs the bounded extraction
  pilot.

Primary and official sources:

- Min et al.,
  [FActScore](https://aclanthology.org/2023.emnlp-main.741/), defines atomic
  facts as separately supportable units and explicitly records limitations
  involving conflicting or overlapping knowledge sources.
- Wei et al.,
  [Long-form factuality / SAFE](https://openreview.net/forum?id=4M9f8VMt2C),
  decomposes responses into individual facts and verifies each independently.
- Cetto et al.,
  [Graphene](https://aclanthology.org/C18-1321/), represents core relational
  propositions with semantically linked contextual information.
- Gashteovski et al.,
  [MinIE](https://aclanthology.org/D17-1278/), retains factuality through
  explicit polarity and modality annotations.
- Morante and Sporleder,
  [Modality and Negation](https://doi.org/10.1162/COLI_a_00095), identifies
  source, time, conditionality, modality type, actuality, polarity, and focus
  as distinct meaning-bearing dimensions.
- [Bespoke Curator](https://github.com/bespokelabsai/curator) provides the
  structured generation, parse, cache, and recovery boundary but does not
  establish domain semantic correctness.
- [Pydantic validators](https://docs.pydantic.dev/latest/concepts/validators/)
  support typed boundary validation; semantic source support remains an
  application responsibility.

Implementation result:

- [x] Added a typed, model-portable `PropositionBatch` schema with atomic
  subject/action/object fields and separate modality, polarity, conditions,
  exceptions, threshold, temporal scope, and exact evidence.
- [x] Added a source-isolated Curator extraction stage. Models emit semantic
  drafts only; stable proposition IDs, authority, source identity, citations,
  and offsets are derived by the application.
- [x] Added fail-closed deterministic checks for non-verbatim semantic fields,
  unsupported or lost modality/polarity, unsupported condition/exception
  markers, missing evidence, and ambiguous duplicate evidence occurrences.
- [x] Resolve evidence offsets against the original registered chunk, even
  when image markup was removed from the generation passage.
- [x] Added a reusable cache under `.curator_working/proposition_cache` keyed
  by source and passage hashes, chunk ID, secret-free model/endpoint and
  decoding configuration, prompt hash, response-schema hash/version, and
  validator version. Explicit empty extractions are cached as terminal results.
- [x] Added per-run `propositions.jsonl`,
  `propositions_generated_audit.jsonl`, and
  `propositions_rejected.jsonl`, plus cache/yield statistics in the run
  manifest. Existing QA and cross-document selection do not consume the new
  propositions yet; P0.2 introduces that dependency after verified paths
  exist.
- [x] Added deterministic tests covering authority isolation, exact offsets,
  cleaned-vs-original source text, modality drift, ambiguous repeated evidence,
  cache invalidation, cache reuse, and empty-result caching.
- [x] Passed 27 local procurement tests, Ruff checks, Python compilation, and
  YAML configuration validation. The existing Pydantic class-config
  deprecation warning is outside this task and remains recorded.
- [ ] Validate extraction completeness, proposition atomicity, cache reuse, and
  latency through a bounded user-run pilot before allowing propositions to
  drive question/path planning.

#### Stable proposition-row schema research record (2026-07-29)

Status: implemented and locally verified after pilot-009; user-run resume
validation pending.

Observed failure:

- Pilot-009 completed all 15 Nemotron proposition requests successfully, then
  failed locally in `datasets.arrow_writer.ArrowWriter.finalize()` with
  `KeyError: 'empty_extraction'`.
- `PropositionExtractor.parse()` currently mixes full proposition dictionaries
  with a smaller empty-extraction sentinel. The failure therefore occurs after
  provider completion and is independent of endpoint availability, Curator
  throttling, or model generation speed.

Official evidence and verified code findings:

- Hugging Face Datasets documents that its Arrow-backed datasets require every
  example to have the same keys and compatible value/subvalue types:
  https://huggingface.co/docs/datasets/package_reference/main_classes
- Hugging Face maintainers likewise explain that Arrow enforces fixed column
  types across rows:
  https://github.com/huggingface/datasets/issues/7322
- Curator's local `BaseRequestProcessor.create_dataset_files()` streams every
  dictionary returned by `parse()` directly into one `ArrowWriter`; it does
  not normalize heterogeneous top-level schemas before finalization.
- The local Curator patch already supports a successful response being
  deliberately filtered with `[]`, but removing empty sentinels would lose
  negative-extraction cache and audit evidence used by this pipeline.

Decision:

- Define one explicit top-level proposition audit-row schema. Materialized
  propositions set `empty_extraction: false`; empty results set
  `empty_extraction: true` and populate every other field with type-compatible
  neutral/source-derived values.
- Keep the empty sentinel so a valid zero-proposition result remains distinct
  from provider failure and can be cached. Do not encode heterogeneous rows as
  arbitrary JSON or buffer the entire Curator result merely to infer a union
  schema.
- Add a regression test containing both a real proposition and an empty
  extraction, and assert identical top-level keys plus compatible nested
  shapes before Arrow serialization.

Risks and validation:

- Neutral values must never pass the accepted-proposition filter; the existing
  empty flag and blank proposition ID remain mandatory rejection signals.
- A stable top-level schema does not itself prove proposition quality.
- Resume pilot-009 only after unit tests and an Arrow round-trip regression
  pass. The user, not Codex, performs the model-backed rerun.

Implementation and verification:

- [x] Added `empty_extraction: false` to every materialized proposition.
- [x] Added a full-schema empty-extraction materializer with immutable source
  metadata, type-compatible neutral fields, blank proposition ID, and explicit
  `empty_extraction: true`.
- [x] Kept accepted-record filtering unchanged: only nonblank proposition IDs
  with passing deterministic checks are eligible.
- [x] Added regression coverage for parser empty batches and an actual
  `ArrowWriter` round trip containing an empty extraction followed by a real
  proposition.
- [x] Passed all 46 focused procurement tests and Ruff checks.
- [ ] Resume pilot-009 with the same command/run ID and inspect all downstream
  artifacts and manifest counts before considering the pilot complete.

### 2. Construct connected reasoning paths before questions

- [x] Build explicit two-hop path types:
  - comparison;
  - bridge;
  - temporal transition;
  - complementary procedure;
  - exception/condition interaction;
  - cross-domain comparison.
- [x] Require compatible subjects or an explicit bridge entity.
- [x] Store the path independently of its natural-language question.
- [x] Reject paths that merely contain two unrelated facts.
- [x] Do not treat similarity as adoption, equivalence, precedence,
  supersession, deletion, or current applicability.

Required path fields:

```json
{
  "path_id": "...",
  "relationship_type": "bridge",
  "required_source_ids": ["source_a", "source_b"],
  "input_claim_ids": ["claim_a", "claim_b"],
  "operations": ["lookup", "combine"],
  "output_claim_id": "claim_c"
}
```

Acceptance criteria:

- Every input claim is evidence-grounded.
- Every operation has valid inputs and an explicit output.
- Removing either required input breaks an answerable path.

#### Connected reasoning-path research record (2026-07-29)

Status: research and deterministic implementation complete; model-backed
validation remains a user-run gate.

Questions researched:

- What distinguishes a genuine two-hop path from two topically similar facts?
- Which path representation permits deterministic connectivity and ablation
  checks before natural-language question generation?
- How should comparison, bridge, temporal, complementary,
  condition/exception, and cross-domain relationships differ?
- Which parts of the current and reference bundle/claim designs are reusable?

Verified local and reference findings:

- The current `build_bundles` stage ranks configured manual pairs by lexical
  and section overlap. It correctly states that similarity does not establish a
  legal relationship, but the bundle has no proposition IDs, bridge entity,
  operations, or derived output claim. The generator invents those structures
  only after it has already written the question and answer.
- The reference project has stronger downstream claim IDs and links rationale
  evidence to them, but it likewise begins with lexically aligned source
  bundles and creates claims inside QA generation. It has no independent,
  pre-question reasoning-path object.
- HotpotQA supplies sentence-level supporting facts, but Min et al. demonstrate
  that compositional wording and multiple annotated documents do not guarantee
  multi-hop necessity; dataset shortcuts can make a question answerable with
  one hop.
- MuSiQue represents each multi-hop question with an underlying DAG composed
  from single-hop questions and explicitly targets connected,
  non-decomposable composition. The graph representation is a better
  supervision contract than prose rationale alone.
- QASC records two source facts, a bridge concept, and a composed fact. This
  makes the otherwise hidden connection auditable and supports explicit
  input/output claim identity.
- 2WikiMultiHopQA includes evidence and reasoning paths, reinforcing that
  supporting facts and their ordered relationship should be part of the data,
  not reconstructed after question generation.

Research-supported design:

- Build paths only from accepted, source-grounded propositions. A path is a
  separate versioned artifact created before questions and answers.
- Represent paths as a small DAG containing `path_id`, relationship/path type,
  two distinct required source IDs, ordered input proposition IDs, explicit
  operations, bridge entities or compatibility keys, and a separate derived
  output claim with its own stable ID.
- Permit the following typed forms:
  - `comparison`: compatible subjects/actions with a compare operation;
  - `bridge`: an exact entity/object from one proposition links to the subject
    or scope of the other, followed by combine;
  - `temporal_transition`: compatible propositions under the same authority
    family with distinct, ordered as-of states; it describes dated states and
    does not assert currentness or supersession;
  - `complementary_procedure`: one proposition's output/object is an explicit
    prerequisite or input to the other;
  - `exception_condition_interaction`: the same governed action/scope is linked
    to an explicit condition or exception;
  - `cross_domain_comparison`: compatible rule signatures across distinct
    procurement domains without asserting adoption or equivalence.
- Candidate retrieval may use normalized lexical/entity signatures, but path
  acceptance requires exact proposition-field anchors. Generic procurement
  words and headings cannot establish connectivity.
- A model may propose a concise derived claim and select an allowed operation,
  but it may reference only supplied proposition IDs and exact bridge strings.
  The application derives IDs and validates graph structure, source
  distinctness, relationship constraints, authority/time safety, and input
  coverage.
- Structural ablation must remove each input proposition in turn and confirm
  that the declared operation/output no longer has all required inputs. This is
  necessary but not sufficient; P0.5 later executes answer-level source
  ablation after a question exists.

Alternatives rejected:

- Treat high lexical similarity as a path: generic policy language creates
  unrelated or redundant pairs.
- Generate the question first and infer its path afterward: creates circular
  verification and lets question wording conceal a one-hop shortcut.
- Require only two citations: evidence count does not prove compositional
  necessity.
- Store a free-text rationale without input/output IDs: cannot deterministically
  validate graph connectivity or ablation.
- Infer missing text as deletion, exception, or non-applicability: absence from
  a bounded window is not evidence.
- Collapse Government and NRL rules into one authority node: risks presenting
  guidance as adopted company policy.

Known risks and proposed validation:

- Exact string bridges are high precision but may miss acronyms,
  coreference, or semantically equivalent entities. Begin conservatively and
  measure recall in a user-reviewed pilot rather than adding fuzzy semantic
  repair.
- Comparison paths can still be redundant when both propositions state the
  same rule. Require the derived output to attribute both states and later
  answer-level ablation to confirm both are needed.
- Temporal ordering from metadata does not itself establish amendment or
  supersession. Tests must reject those inferred relations.
- A valid graph can still yield an unnatural or answer-leaking question.
  Question quality is a separate P0.4 gate.
- Codex will validate path schemas, IDs, source separation, operation typing,
  anchors, and structural ablation without model calls. Only the user runs a
  bounded path-planning pilot.

Primary and official sources:

- Yang et al.,
  [HotpotQA](https://aclanthology.org/D18-1259/), provides document-level
  multi-hop questions with sentence-level supporting-fact supervision.
- Min et al.,
  [Compositional Questions Do Not Necessitate Multi-hop Reasoning](https://aclanthology.org/P19-1416/),
  demonstrates that multi-document/compositional form alone does not prove
  multi-hop necessity.
- Trivedi et al.,
  [MuSiQue](https://aclanthology.org/2022.tacl-1.31/), represents composed
  multi-hop questions with an underlying DAG and targets connected
  composition.
- Khot et al.,
  [QASC](https://doi.org/10.1609/aaai.v34i05.6319), annotates two facts, their
  bridge/composition, and the composed fact.
- Ho et al.,
  [2WikiMultiHopQA](https://aclanthology.org/2020.coling-main.580/), includes
  evidence information containing reasoning paths for multi-hop questions.

Implementation result:

- [x] Added a versioned, model-independent path builder that consumes only
  accepted propositions and explicitly configured manual pairs.
- [x] Added all six typed path forms, stable path/output-claim IDs, two
  distinct source IDs, ordered input claim IDs, typed operations, exact
  non-generic connection anchors, explicit per-operation input/output graph
  nodes, a separately attributed output claim, and structural removal results
  for both required inputs.
- [x] Added fail-closed checks for unknown or ungrounded inputs, same-source
  paths, pair mismatch, unsupported anchors, incompatible signatures, missing
  bridges, authority/family/date errors, cross-domain errors, missing
  conditions/exceptions, malformed outputs, unsafe legal-relationship language,
  invalid operations, and invalid structural ablation.
- [x] Preserved immutable proposition revision/as-of metadata and bumped the
  proposition schema/cache validator version so temporal paths never substitute
  page numbers or model-authored dates for registered manifest dates.
- [x] Added `reasoning_paths.jsonl` and
  `reasoning_paths_rejected.jsonl` before the existing question-generation
  stages, plus manifest statistics and configurable per-pair bounds.
- [x] Added deterministic regression tests for stability, connectivity,
  source separation, unrelated-pair rejection, unsafe supersession claims,
  temporal attribution, exact bridges, cache-schema invalidation, and
  structural ablation. Thirty local procurement tests, Ruff, compilation, and
  YAML parsing pass.
- [ ] Validate path yield, false connections, proposition completeness, and
  path-type coverage through a bounded user-run pilot. Existing legacy
  question generation remains isolated until P0.4 makes verified paths its
  required input.

### 3. Use bounded multi-chunk source windows

- [x] Complete and record the mandatory research-first gate for source-window
  construction, section hierarchy, adjacency, cross-references, provenance,
  splitting, and model-aware prompt budgets.
- [x] Group adjacent chunks under reliable section boundaries.
- [x] Include required definitions, exceptions, and referenced provisions when
  resolvable.
- [x] Split oversized sections on chunk boundaries.
- [x] Preserve every constituent chunk ID and page.
- [x] Prevent repeated or generic headings from establishing a match by
  themselves.
- [x] Support one-to-many section relationships.

Implementation progress:

- [x] Added stable corpus `document_order`, per-page chunk order, and explicit
  Markdown heading breadcrumbs without changing chunk IDs or source text.
- [x] Added deterministic section/adjacency window construction with stable
  IDs, ordered constituent chunks, pages, source hash, authority/edition
  isolation, confidence labels, configurable chunk bounds, and conservative
  pre-call token budgets.
- [x] Oversized windows split only between chunks; an individually oversized
  chunk is quarantined with `source_chunk_exceeds_token_budget` and is never
  truncated.
- [x] Added `source_windows.jsonl`, `source_windows_rejected.jsonl`, manifest
  statistics, configuration, documentation, and regression tests. Thirty-two
  local tests, Ruff, compilation, and YAML parsing pass.
- [x] Add exact, unambiguous definition/exception/cross-reference support
  edges and one-to-many path associations.
- [x] Add a selected-tokenizer rendered-chat budgeting utility; the current
  conservative estimator is recorded explicitly and remains the fallback.
- [x] Invoke full rendered-request budgeting in P0.4 immediately before every
  new path-driven Curator call. Legacy generation remains unchanged until that
  migration.

##### Cross-reference and rendered-token-budget research addendum (2026-07-29)

Status: focused research and reusable implementation complete; enforcement on
the new path-driven request stage remains a P0.4 integration gate.

Verified findings and decision:

- OASIS Akoma Ntoso models legal cross-references through explicit,
  referenceable component identities. This supports resolving a printed
  procurement reference such as “para 5.6.8” to a uniquely indexed component,
  not to a fuzzy text/heading match.
- The current Markdown/OCR corpus has no authored component IDs. The pipeline
  must extract conservative paragraph/annexure identifiers from source text,
  retain their exact spelling and chunk location, and resolve references only
  within the same manual/version unless the source explicitly identifies
  another registered document.
- One-to-many is valid only for an explicit enumeration/range whose individual
  targets all resolve uniquely. Missing, duplicated, malformed, external, and
  ambiguous references remain audit records and never become support edges.
- Hugging Face documents that `apply_chat_template(..., tokenize=True)` returns
  token IDs with the tokenizer's chat template and control tokens applied.
  This is more faithful than counting raw source text.
- Unknown OpenAI-compatible hosted-vLLM names can make LiteLLM use generic
  token counting. The pipeline must not treat that estimate as exact.
- Add optional tokenizer identity/revision to each model profile. Load only
  from an already available local path/cache during a run; never download
  corpus-adjacent model assets implicitly. Record tokenizer identity, revision,
  template hash, count method, reserved completion, and safety margin.
- The complete rendered request—not the source window alone—is the enforcement
  boundary. A reusable counter may be added now, but P0.4 must invoke it after
  constructing the actual messages and structured-output schema and before
  Curator/provider submission.
- When no exact local tokenizer/template exists, use the configured
  conservative fallback and mark it approximate. If a profile requires exact
  counting, fail before any request.

Rejected alternatives and risks:

- Fuzzy section/reference matching can attach the wrong legal provision.
- Resolving across editions by paragraph number can silently mix temporal
  states.
- `len(text)/4`, advertised context size, or LiteLLM's generic counter alone
  does not include reliable model-template/schema overhead.
- A local tokenizer with a different revision/template can still disagree with
  the server. Pin and record revisions and retain a safety margin; endpoint
  preflight remains necessary.

Official sources:

- [OASIS Akoma Ntoso 1.0 vocabulary](https://docs.oasis-open.org/legaldocml/akn-core/v1.0/akn-core-v1.0-part1-vocabulary.html)
  specifies referenceable legal-document components and cross-reference
  mechanisms.
- [OASIS Akoma Ntoso naming convention](https://docs.oasis-open.org/legaldocml/akn-nc/v1.0/csd02/akn-nc-v1.0-csd02.html)
  separates the reference point from the identified target resource.
- [Hugging Face chat-template documentation](https://huggingface.co/docs/transformers/en/chat_templating_writing)
  documents tokenizer-owned templates, generation prompts, special tokens, and
  tool-schema template inputs.
- [Hugging Face tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer)
  documents tokenized chat-template output including control tokens.

Implementation result:

- [x] Added conservative same-manual component indexing for explicit numbered
  paragraphs/clauses/sections and annexures.
- [x] Explicit comma/`and` enumerations create one-to-many edges only when each
  target resolves uniquely. Missing and duplicate targets remain audited as
  `missing` or `ambiguous` and never become support edges.
- [x] Windows now retain both accepted `support_edges` and complete
  `reference_audit` entries with source chunk, raw citation, normalized target,
  resolution status, and target chunk IDs.
- [x] Added reusable full-request measurement that counts chat-template control
  tokens and response-schema tokens with a supplied local tokenizer, records
  tokenizer/template identity, reserves completion and safety margin, and can
  require exact counting. Its fallback is explicitly labeled conservative.
- [x] Added regression tests for unique, missing, ambiguous, same-manual,
  one-to-many, tokenizer-template, response-schema, safety-margin, fallback,
  and exact-mode behavior. Thirty-six local tests, Ruff, compilation, and YAML
  parsing pass.

##### Pilot-009 model-context budgeting correction (2026-07-29)

Status: implemented and locally verified; next user-run pilot pending.

Observed behavior and root cause:

- All 12 accepted reasoning paths were rejected before question generation.
  Their approximate rendered prompts were 4,037–4,131 tokens; with 4,096
  completion tokens and a 256-token margin, each narrowly exceeded 8,192.
- The selected Nemotron profile has no `context_window`, so the path stages
  incorrectly fall back to `source_windows.max_input_tokens: 8192`. A source
  window construction bound is not the serving model's context window.
- The user confirmed that the private Nemotron deployment is configured for a
  131K context length. This is below the model's theoretical maximum and is
  therefore the operational limit the client must enforce.

Official findings:

- NVIDIA's official model card and Nemotron repository state that Nemotron 3
  Super supports up to 1M tokens, but “up to” is a model capability rather
  than proof of a particular server's launch-time limit:
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
  and
  https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/super3/README.md
- Hugging Face documents that chat messages become a model-specific token
  sequence and recommends `apply_chat_template(..., tokenize=True)` for exact
  template-aware counting:
  https://huggingface.co/docs/transformers/main/en/chat_templating
- LiteLLM documents model-specific/custom tokenizers but falls back to a
  generic tokenizer for unsupported model names. Such a fallback must not be
  labelled exact:
  https://docs.litellm.ai/docs/completion/token_usage

Decision:

- Add `context_window: 131072` to the Nemotron profile. Keep this value
  profile-local and editable in configuration; never hard-code it in pipeline
  logic or apply it to GLM, Gemma, Qwen, or future models.
- Require every generation profile used by a full rendered-request budget to
  declare a positive context window. Do not silently substitute
  `source_windows.max_input_tokens`.
- Preserve the independent 8,192-token Gemma judge limit and the 8,192-token
  source-window construction bound. Those are separate constraints.
- Continue using the explicitly labelled conservative character estimator
  until an exact, server-matching local Nemotron tokenizer/template is
  configured. Retain completion reserve and safety margin.

Rejected alternatives and risks:

- Setting the client to the advertised 1M maximum could exceed this deployment.
- Increasing `source_windows.max_input_tokens` would enlarge evidence windows
  globally and confuse source selection with provider capacity.
- Removing the preflight budget would turn a deterministic quarantine into
  provider-side context failures.
- A configured context limit does not guarantee GPU KV-cache capacity or
  acceptable latency at that length; pilot prompts remain far below 131K.

Validation plan:

- Unit-test explicit profile resolution, rejection of missing/non-positive
  context limits, and isolation from `source_windows.max_input_tokens`.
- Re-evaluate the 12 saved pilot-009 path prompts locally and confirm they pass
  under 131,072 while preserving the same token estimates and reserves.
- The user performs the next model-backed pilot; Codex does not run it.

Implementation result:

- [x] Added the private Nemotron deployment's confirmed
  `context_window: 131072` to its model profile.
- [x] Added fail-closed model-context resolution and removed all three
  path-question/path-answer/ablation fallbacks to the unrelated
  `source_windows.max_input_tokens`.
- [x] Preserved the 8,192-token source-window bound and Gemma judge context as
  separate profile/configuration values.
- [x] Confirmed locally that all 12 saved pilot-009 path-question requests pass
  with the corrected 131,072-token deployment limit without changing their
  estimated prompt size, completion reserve, or safety margin.
- [x] Passed 47 focused procurement tests, Ruff, and YAML configuration
  validation.
- [ ] Validate path-question, answer, and ablation yields in the next bounded
  user-run pilot.

Acceptance criteria:

- A source window never crosses manual, issuer, edition, or policy scope.
- Prompt size is checked before generation.
- Exact evidence still resolves to its original chunk, page, and offsets.

#### Bounded source-window research record (2026-07-29)

Status: research gate complete; production implementation has not started.

Questions researched:

- When should adjacent chunks be joined without crossing a legal/manual
  section boundary?
- How should headings, pages, definitions, exceptions, tables, and explicit
  paragraph cross-references affect a window?
- How can windows preserve exact evidence provenance while remaining within
  different local models' rendered context limits?
- Which reference-project behaviors are safe to reuse?

Verified local and reference findings:

- The current corpus splitter works within one physical page and groups blank
  line-delimited paragraphs up to a character limit. It carries only the most
  recently observed final heading string. It does not retain heading level or
  breadcrumb, chunk ordinal, paragraph IDs, table continuation, or explicit
  cross-reference edges. Consequently, equal heading text cannot prove a
  shared section and adjacency cannot be reconstructed safely from chunk IDs.
- Current proposition evidence already resolves to immutable original chunk
  IDs, pages, exact quotes, and offsets. A window should reference these chunks
  rather than concatenate them into a new evidentiary source.
- Current cross-document bundles pair one chunk per manual using lexical and
  heading overlap. That is retrieval, not source-window construction, and
  repeated generic headings can inflate its score.
- The reference `build_reasoning_windows` unconditionally joins adjacent
  retained pages in pairs. It preserves both chunk IDs and page range, but can
  cross unrelated sections, misses more-than-two-chunk sections and
  cross-references, and has no rendered-prompt budget. Its newer annealing
  helper groups canonicalized headings with a character cap, but identical
  heading labels and character estimates remain insufficient proof/budgeting.
- Research consistently finds a trade-off: small fixed chunks lose semantic
  completeness while large chunks introduce irrelevant context. Structure-
  aware segmentation can improve retrieval, but no paper establishes a
  universal chunk size for these procurement manuals.
- Long advertised context does not guarantee reliable use of all positions.
  Therefore the maximum server context must be treated as a hard ceiling, not
  a target window size.
- Hugging Face tokenizers can render chat templates directly to token IDs,
  including model-specific control tokens. A raw text or `len(text)/4`
  estimate cannot be the sole production check for the complete request.

Research-supported design:

- First enrich corpus chunks with immutable `document_order`, `page`,
  `chunk_index`, and a heading stack derived from explicit Markdown headings.
  OCR/PDF rows without reliable hierarchy use physical adjacency only and are
  marked with lower boundary confidence; inferred plain-text headings cannot
  silently become authoritative hierarchy.
- Build a base window from consecutive chunks in one manual, source hash,
  issuer, edition/as-of date, and policy scope. Continue while the reliable
  section breadcrumb is unchanged. A page boundary is allowed; a manual,
  authority, edition, or reliable section boundary is not.
- Treat blank/generic/repeated headings as non-keys. Normalize headings only
  for comparison while retaining exact heading text and level. Never join
  non-adjacent chunks merely because their labels match.
- Attach definitions, exceptions, and referenced provisions as explicit
  support chunks only when a parsed reference resolves uniquely within the
  same registered authority/edition. Record the edge type and target; preserve
  unresolved or ambiguous references for audit instead of guessing.
- Split oversized sections only between constituent chunks. Do not cut or
  rewrite the source chunks. Every window stores ordered chunk IDs, page
  ranges, source hashes, section breadcrumb, boundary confidence, support
  edges, and a stable ID derived from this versioned structure.
- Budget the fully rendered chat request, including system/user text,
  structured-output/tool schema overhead, model control tokens, reserved
  completion tokens, and a safety margin. Use the selected model tokenizer and
  chat template when its tokenizer is locally available. Otherwise use a
  configurable conservative estimator and label the estimate method; reject
  or split before any provider call.
- Keep one-to-many relationships at the path/window association layer. Do not
  duplicate or merge evidence text to make a one-to-one pair.

Alternatives rejected:

- Fixed two-page windows: simple, but crosses sections and cannot express long
  or referenced provisions.
- Group all chunks with equal heading text: repeated “General”, “Note”, and
  OCR headings create false, non-adjacent groups.
- LLM-authored section boundaries: expensive and can fabricate structure;
  it may later be evaluated as an explicitly uncertain enrichment, not source
  truth.
- Character count as the only context guard: model tokenization and chat/tool
  wrappers vary.
- Use the configured maximum context exactly: leaves no room for completion,
  schemas, template overhead, or server-specific limits and worsens irrelevant
  context exposure.
- Copy referenced text into the anchor chunk: destroys source-specific offsets
  and obscures which provision supplied each fact.

Known risks and proposed validation:

- OCR headings and split tables may lack reliable Markdown structure. Emit
  boundary-confidence/reason fields and audit these separately.
- Cross-reference syntax varies and may refer to annexures, clauses, tables,
  chapters, or external instruments. Start with exact, tested patterns and
  fail closed on multiple targets.
- A local tokenizer may differ from the serving engine's exact revision or
  chat template. Record tokenizer identity and retain a configurable safety
  margin; an endpoint preflight remains required for newly configured models.
- Section-coherent windows may still omit a definition outside the section or
  include irrelevant boilerplate. Test fixtures and later user review must
  measure both missing-support and excess-context errors.
- Local tests will cover boundary isolation, page continuation, repeated
  headings, ambiguous references, oversized splitting, stable IDs, provenance,
  and budget rejection. Only the user will run model-backed validation.

Primary and official sources:

- Wang et al.,
  [Document Segmentation Matters for Retrieval-Augmented
  Generation](https://aclanthology.org/2025.findings-acl.422/), documents the
  semantic-coherence versus irrelevant-context trade-off and evaluates
  adaptive segmentation.
- Liu et al.,
  [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/), shows that
  effective long-context use degrades with information position even for
  long-context models.
- Trivedi et al.,
  [MuSiQue](https://aclanthology.org/2022.tacl-1.31/), supports preserving
  explicit connected evidence structure for compositional QA.
- [Hugging Face tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer)
  documents chat-template rendering to token IDs including control tokens.
- [CommonMark 0.31.2](https://spec.commonmark.org/0.31.2/) defines explicit ATX
  and Setext heading syntax used as the reliable Markdown boundary source.
- [Bespoke Curator repository](https://github.com/bespokelabsai/curator)
  documents prompt/parse orchestration and caching but provides no
  domain-specific section-window builder or universal safe context budget.

### 4. Generate questions from verified paths

- [x] Complete and record the mandatory research-first gate for path-conditioned
  question planning, separate answering, difficulty control, standalone
  authority/time wording, missing-hop negatives, and false-premise negatives.
- [x] Generate natural-language questions only after a path passes
  deterministic checks.
- [x] Control the intended question type and difficulty.
- [x] Require standalone authority, domain, and date wording when necessary.
- [x] Generate the final answer in a separate pass from question construction.
- [ ] Add missing-hop and false-premise unanswerable contrasts.

Acceptance criteria:

- The question cannot be fully answered without executing the declared path.
- The answer is produced from the supplied sources, not copied from a proposed
  answer embedded in the question-generation prompt.
- Unanswerable records state only supported limitations.

#### Verified-path question-generation research record (2026-07-29)

Status: research gate complete; production implementation has not started.

Verified local/reference findings:

- The current `CrossDocumentGenerator` receives lexically paired passages and
  emits question, answer, claims, evidence, and rationale in one model call.
  Its prompt asks for two-source necessity, but no accepted pre-question path
  constrains the output and the proposed answer can rationalize a weak question.
- The new local proposition/path stages now provide immutable source IDs,
  grounded input claims, typed operations, connection anchors, and an attributed
  output claim before question wording exists. Legacy QA does not consume them.
- The reference project separates topic discovery, question blueprints, and
  later generation for some tracks, which is a useful staged pattern. Its
  cross-document stage still generates question and response jointly from
  bundles, and its blueprints are not backed by the independently validated
  path DAG now available here.
- MuSiQue builds questions from connected single-hop components and includes
  unanswerable constructions formed by disconnecting the reasoning chain. This
  supports missing-hop negatives derived from a valid path rather than arbitrary
  “unanswerable” prompting.
- 2WikiMultiHopQA preserves evidence/reasoning paths and uses controlled
  generation procedures. HotpotQA supplies comparison questions and supporting
  facts, but later shortcut analyses show that multiple documents/citations do
  not alone prove necessity.
- False-premise questions differ from missing-context questions: the premise is
  wrong, not merely unsupported by the visible context. They require explicit
  premise verification and must not be answered as if true.

Research-supported design:

- Stage A generates a typed question proposal from one accepted path. It sees
  path type, operations, input propositions, exact source metadata/evidence,
  intended question type, difficulty, persona, and task. It does not receive a
  model-authored answer. The derived output claim may be used as a private
  planning constraint but must be excluded from the user-facing question and
  checked for answer leakage.
- Deterministically reject unknown path/claim/source IDs, failed path checks,
  unsupported taxonomy, answer text leakage, questions answerable from one
  proposition, and missing authority/domain/date wording where ambiguity
  exists. Difficulty is structural metadata (operation/path form and required
  qualifications), not a request for obscure wording.
- Enforce the complete rendered prompt budget immediately before each Curator
  call using the P0.3 counter, selected local tokenizer when configured, schema,
  completion reservation, and safety margin. Over-budget requests are
  quarantined before provider submission.
- Stage B answers only an accepted question using its immutable path and source
  evidence. It emits material claims/evidence and an optional concise,
  auditable teaching rationale. It cannot modify the question, path, taxonomy,
  or required sources.
- Construct missing-hop negatives deterministically by withholding one required
  source/path input from an otherwise valid question. Label the withheld source
  and expected limitation; do not expose the hidden answer in visible context.
- Construct false-premise candidates only through an explicit, typed mutation
  of one supported premise (authority, date, modality, threshold, condition,
  or entity). Preserve the original and mutation audit, then require a
  verifier to confirm the mutated premise is contradicted—not merely absent.
  Until that verifier exists and passes, keep false-premise candidates out of
  accepted training exports.
- Keep answerable, missing-hop, and verified-false-premise records as distinct
  classes and report their yields separately.

Rejected alternatives/risks:

- Joint question-answer generation permits circular self-consistency without
  source necessity.
- Showing a polished target answer to the question writer encourages lexical
  leakage and answer-shaped questions.
- Treating any withheld source as proof of unanswerability ignores parametric
  knowledge and one-hop shortcuts; P0.5 must execute actual answer ablations.
- Randomly changing a number/name can create a different valid rule or an
  merely unsupported premise. False-premise acceptance requires contradiction.
- “Hard” prompts based on verbosity or obscure phrasing damage naturalness and
  do not measure reasoning depth.

Planned validation:

- Unit tests for path-only eligibility, prompt leakage, standalone
  authority/time wording, stable IDs, profile-independent schemas, prompt
  budgets, immutable question handoff, missing-hop lineage, and quarantined
  unverified false premises.
- P0.5 later validates genuine necessity through full/A-only/B-only answer
  trials. Only the user runs model-backed pilots.

Primary sources:

- Trivedi et al.,
  [MuSiQue](https://aclanthology.org/2022.tacl-1.31/), constructs connected,
  compositional multi-hop questions and disconnected unanswerable variants.
- Ho et al.,
  [2WikiMultiHopQA](https://aclanthology.org/2020.coling-main.580/), includes
  evidence information containing complete reasoning paths.
- Yang et al.,
  [HotpotQA](https://aclanthology.org/D18-1259/), defines multi-document
  comparison questions with supporting-fact supervision.
- Min et al.,
  [Compositional Questions Do Not Necessitate Multi-hop
  Reasoning](https://aclanthology.org/P19-1416/), demonstrates shortcut risks
  in apparently compositional questions.
- Yu et al.,
  [Won't Get Fooled Again: Answering Questions with False
  Premises](https://aclanthology.org/2023.acl-long.309/), distinguishes false
  premises from missing-context unanswerability and evaluates premise
  verification.

Implementation progress:

- [x] Added typed, model-portable question and answer schemas and separate
  Curator stages. The planner receives a verified path and grounded
  propositions without a polished target answer; the answerer receives the
  accepted immutable question.
- [x] Added deterministic checks for failed/unknown path inputs, taxonomy,
  standalone authority/domain/date wording, malformed questions, output-claim
  leakage, exact proposition evidence, two-input coverage, claim count, and
  QA-versus-rationale shape.
- [x] Enforced full rendered planner and answer prompt budgets, including
  response schema, reserved completion, and safety margin, before either new
  provider call. Over-budget inputs are written to dedicated rejection audits.
- [x] Added stable planner/answer IDs, raw/accepted/rejected audit artifacts,
  deterministic QA/CoT assignment, and manifest yield statistics.
- [x] Added two traceable missing-hop twins per accepted question with exact
  withheld source/proposition and visible-source lineage.
- [x] False-premise candidates are explicitly quarantined with
  `contradiction_verifier_not_implemented`; they cannot enter training output.
  Implementing and researching that verifier remains required before this
  checklist item is complete.
- [x] New path answers report `accepted_for_training: 0` until P0.5 real source
  ablation and independent judging complete. Legacy QA behavior remains
  unchanged.
- [x] Thirty-nine local procurement tests, Ruff, compilation, and YAML parsing
  pass. Model-backed quality remains a user-run validation gate.

### 5. Execute real source-ablation trials

- [x] Complete and record the mandatory research-first gate for counterfactual
  source removal, answer trials, claim coverage, invalid-trial semantics,
  caching, and independent adjudication.
- [x] Answer each candidate with both sources.
- [x] Answer it again with only source A.
- [x] Answer it again with only source B.
- [ ] Measure required-claim coverage for all three outputs.
- [ ] Store the actual ablation outputs and decisions.
- [ ] Reject answerable records when either single-source output fully covers
  the canonical answer.

Acceptance criteria:

- Full context covers all required answer claims.
- A-only misses at least one required source-B claim.
- B-only misses at least one required source-A claim.
- The judge reviews actual ablated outputs rather than predicting the result.

#### Real source-ablation research record (2026-07-29)

Status: research gate complete; production implementation has not started.

Verified findings and design:

- The legacy `CrossDocumentJudge` is shown the canonical answer and predicts
  whether removing each source would break it. It does not execute three
  answer calls, so its `unsupported_without_source_ids` is model opinion rather
  than observed counterfactual behavior.
- P0.4 now produces immutable questions, verified path inputs, and candidate
  answers but intentionally exports none for training. These are the correct
  inputs for actual ablation.
- Multi-hop QA research repeatedly finds disconnected reasoning: systems can
  answer intended multi-hop questions from subsets or shortcuts. Correct final
  answer alone does not prove the intended path was used.
- Run three answer trials with the same answer schema, model profile, decoding,
  prompt structure, and completion budget: full inputs, only input A, and only
  input B. The visible context and declared visible proposition IDs are the
  only intended difference.
- Trial outputs must include `answerable`, answer, material claim IDs covered,
  exact evidence, and limitation reason. A trial may abstain. Unknown evidence,
  claims attributed to withheld sources, malformed output, timeout, or missing
  response makes the trial invalid; it does not prove source necessity.
- Compare coverage against stable required claims derived from the verified
  path/canonical full answer. Full context must cover every required terminal
  claim. A-only must fail to cover at least one valid B-grounded required claim;
  B-only must fail analogously for A. Merely changing wording or answer length
  is irrelevant.
- Store raw inputs/outputs, visible/withheld sources, request IDs, prompt
  budgets, deterministic status, claim coverage, and decision for all trials.
  Cache identity must include question/path/claim IDs, visible proposition
  hashes, model profile/endpoint excluding secrets, decoding, prompt/schema,
  and validator version.
- After deterministic coverage, an independent judge reviews all three actual
  outputs together. It cannot replace invalid trials or infer missing calls.
- False-premise verification is distinct: a contradiction trial must compare a
  typed mutated premise against full evidence. Source absence is insufficient.
  Keep it quarantined rather than reuse missing-hop ablation as a contradiction
  label.

Rejected alternatives and risks:

- Judge-only predicted ablation is cheaper but cannot detect actual one-hop
  answering.
- String/semantic similarity between full and ablated answers cannot establish
  material claim coverage.
- Treating timeout, refusal, or schema failure as “source needed” inflates
  necessity.
- Using different models or decoding across contexts confounds source removal
  with inference behavior.
- Even successful ablation is model-relative, not a universal proof that no
  other model or human can exploit a shortcut. Record model/profile and retain
  human review.

Implementation progress (three-trial execution):

- [x] Added one model-portable trial schema and one Curator answer class shared
  by full, A-only, and B-only trials, ensuring the model profile, decoding,
  prompt template, completion schema, and validator are identical.
- [x] Constructed three stable trial IDs per accepted path answer. Only the
  visible proposition list differs; canonical answers, canonical claim text,
  withheld proposition content, and internal variant labels are absent from
  the model prompt.
- [x] Added rendered prompt-budget checks before every trial and dedicated
  prompt-rejection, raw-audit, valid, and invalid JSONL artifacts under the
  dynamic run directory.
- [x] Added deterministic trial validation for empty answerable outputs,
  unsupported abstentions, missing claim evidence, unknown/withheld
  proposition use, and non-exact evidence.
- [x] Added manifest counts for planned, prompt-rejected, valid, and invalid
  trials. Invalid or missing trials remain ineligible for a necessity
  decision.
- [x] Regression tests verify the exact three context variants, prompt
  blindness, and withheld/inexact-evidence rejection. Complete local
  procurement verification: 44 tests passed; Ruff passed. No model-backed
  pipeline was run.

Remaining P0.5 work:

- Derive stable required-claim coverage from the canonical verified answer.
- Independently adjudicate the three actual outputs and enforce the full/A/B
  coverage decision.
- Store the final coverage decision and promote only passing path records from
  `accepted_for_training: 0`.

Primary sources:

- Min et al.,
  [Compositional Questions Do Not Necessitate Multi-hop
  Reasoning](https://aclanthology.org/P19-1416/), shows intended multi-hop
  questions often remain answerable from partial evidence.
- Trivedi et al.,
  [Is Multihop QA in DiRe Condition?](https://aclanthology.org/2020.emnlp-main.712/),
  formalizes disconnected reasoning and contrastive support sufficiency.
- Tang et al.,
  [Do Multi-Hop QA Systems Know How to Answer the Single-Hop
  Sub-Questions?](https://aclanthology.org/2021.eacl-main.283/), shows final
  answer correctness can conceal failure on constituent reasoning.
- Guo et al.,
  [Counterfactual Multihop QA](https://aclanthology.org/2023.acl-long.231/),
  targets answers produced from a single fact rather than true multi-hop
  reasoning.
- Paranjape et al.,
  [Retrieval-guided Counterfactual Generation for
  QA](https://aclanthology.org/2022.acl-long.117/), uses a generate-and-filter
  workflow and highlights answerability risks in counterfactual QA.

### 6. Create a replayable claim and reasoning graph

- [ ] Give every material answer claim a stable ID.
- [ ] Give every evidence item a stable ID.
- [ ] Add `input_claim_ids`, `output_claim_id`, and `evidence_refs` to each
  reasoning step.
- [ ] Verify that the graph is connected and acyclic.
- [ ] Verify that the final answer is covered by terminal claims.
- [ ] Derive QA and QA-with-rationale views from one canonical record.

Acceptance criteria:

- No reasoning step refers to an unknown claim or evidence item.
- Every non-source claim is derived by a declared operation.
- Removing the rationale from the export does not change the final answer or
  lineage.

### 7. Expand deterministic verification

- [ ] Validate answer and rationale numbers, percentages, dates, durations,
  monetary values, names, and email addresses.
- [ ] Preserve `shall`, `must`, `may`, negation, exceptions, and conditions.
- [ ] Require temporal labels for historical comparisons.
- [ ] Reject unsupported currentness or effective-date claims.
- [ ] Reject unsupported adoption, equivalence, precedence, supersession, and
  deletion claims.
- [ ] Require the question's comparison or bridge subject to be supported in
  every relevant source.
- [ ] Require comparison answers to state the actual relationship rather than
  concatenate summaries.

#### Evidence entailment, absence, and deontic-modality research (2026-07-29)

Status: first conservative deterministic gate implemented and locally
verified; atomic claim-to-evidence enforcement remains pending.

Pilot evidence:

- An accepted temporal comparison asserted that a refund provision was “not
  present” in the 2019 manual even though the request supplied only a bounded
  passage from that manual. Another rationale treated the lack of a sentence
  in one selected passage as proof that the whole Government manual lacked it.
- An accepted cross-document answer strengthened “should be sealed” to “must
  be sealed.” An accepted drafting clause strengthened the source's “buyer may
  recover” liquidated damages to “shall ... deduct.”
- Existing deterministic checks detect dropped source qualifiers in ordinary
  QA, but do not reject introduced/strengthened modality, do not cover drafting
  responses, and do not reject whole-document absence claims.

Research-supported conclusions:

- ContractNLI treats `Entailment`, `Contradiction`, and `NotMentioned` as
  distinct document-level labels and requires evidence spans. Failure to find
  a proposition in a bounded passage is neutral/not-mentioned evidence, not
  proof of its negation or absence from a document.
- FActScore evaluates atomic generated facts against reliable supporting
  sources. Validation should therefore operate claim-wise where possible and
  reject any material generated claim that lacks support, rather than accept a
  response because some surrounding text is grounded.
- Research on agent-specific deontic modality in legal language distinguishes
  obligation, prohibition, permission, and entitlement, with cues including
  `shall`, `shall not`, and `may`. Changing permission/recommendation into an
  obligation—or the reverse—is a semantic legal change, not paraphrase.

Primary sources:

- Koreeda and Manning,
  [ContractNLI](https://aclanthology.org/2021.findings-emnlp.164/), defines
  entailed, contradicting, and not-mentioned contract hypotheses with evidence
  spans; the official implementation is
  https://github.com/stanfordnlp/contract-nli-bert.
- Min et al.,
  [FActScore](https://aclanthology.org/2023.emnlp-main.741/), evaluates
  fine-grained atomic facts against reliable knowledge sources.
- Savelka et al.,
  [Agent-Specific Deontic Modality Detection in Legal
  Language](https://aclanthology.org/2022.emnlp-main.795/), models legal
  obligations, permissions, prohibitions, and entitlements separately.
- Kryscinski et al.,
  [FactCC](https://aclanthology.org/2020.emnlp-main.750/), couples factual
  consistency decisions with supporting source spans.

Decision:

- Add shared deterministic checks for high-confidence modality strengthening
  and weakening. Apply them to ordinary QA, cross-document QA/rationales, and
  drafting using only the exact evidence quotes declared for the generated
  material.
- Reject explicit whole-document absence/deletion claims such as “not present
  in the manual” unless the supplied evidence itself explicitly states the
  absence, deletion, withdrawal, or lack of provision. Bounded-window silence
  never establishes absence.
- Keep categories conservative: obligation, recommendation, permission, and
  prohibition. Do not attempt unrestricted semantic NLI with regexes; the
  independent judge remains an additional gate.
- Use stable reason codes that distinguish unsupported absence from
  strengthened/weakened modality, and add pilot-derived regression cases.

Risks and rejected alternatives:

- Exact word equality alone rejects valid `shall`/`must` equivalence; use
  category-level cue equivalence.
- Treating every `will` as obligation is unsafe because it is often
  declarative/future tense; exclude it from the first high-confidence gate.
- Searching an entire manual for a phrase is still not proof of semantic
  absence and is sensitive to paraphrase/OCR; absence generation requires an
  explicit source statement or a separately designed exhaustive document-level
  task.
- A response containing several claims and several evidence spans can mask a
  local modality mismatch. This conservative batch-level gate is an immediate
  safeguard; the planned atomic claim-to-evidence schema remains the stronger
  long-term solution.

Implementation result:

- [x] Added shared, category-aware checks for obligation, recommendation,
  permission, and prohibition cues, including `shall`/`must` equivalence.
- [x] Reject high-confidence permission/recommendation-to-obligation
  strengthening, obligation weakening, and introduced obligations or
  prohibitions with stable reason codes.
- [x] Reject explicit whole-document absence claims when their declared
  evidence does not itself state absence/lack.
- [x] Apply the checks to ordinary QA answers, cross-document answers, each
  cross-document rationale step, and drafting outputs using declared exact
  evidence.
- [x] Re-audited pilot-009 locally: the gate identifies the accepted
  “should”→“must” records, unsupported “not present” comparison, and an
  introduced obligation that was supported only by a noun-phrase fragment.
- [x] Passed 49 focused procurement tests and Ruff checks, including a
  drafting permission→obligation regression.
- [ ] Replace response-level multi-evidence checking with atomic
  claim/block-to-evidence bindings so one correctly modalized source cannot
  mask a different claim's modality drift.

Acceptance criteria:

- Every deterministic failure has a stable machine-readable reason code.
- Unit tests cover wrong-source attribution, swapped dates, dropped
  exceptions, unsupported numbers, and single-source shortcuts.

### 8. Add complete rejection and lineage auditing

- [ ] Write source bundles and proposition sets.
- [ ] Write reasoning-path candidates.
- [ ] Write raw generation candidates.
- [ ] Write deterministic rejections.
- [ ] Write source-ablation rejections.
- [ ] Write judge rejections.
- [ ] Write duplicate and best-of-N rejections.
- [ ] Write accepted canonical audit records.
- [ ] Write forward lineage from source bundle to every derived record.
- [ ] Produce pair-level coverage statistics.

The manifest must report:

- configured pairs;
- pairs with no source bundles;
- bundles per pair;
- candidates per pair;
- accepted records per pair and task type;
- rejection counts by reason;
- model and configuration fingerprints;
- incomplete or rescued stages.

### 9. Replace the split strategy

- [ ] Add explicit held-out validation and test manuals to `config.yaml`.
- [ ] Keep source chunks, section windows, path families, QA/CoT views, RAG
  variants, and distractor variants together.
- [ ] Prevent question paraphrases from crossing splits.
- [ ] Generate a leakage audit by source hash, manual, section, chunk, path,
  and normalized question.
- [ ] Support multiple evaluation folds because the corpus contains only 19
  manuals.

Acceptance criteria:

- Test sources never occur in training evidence.
- Derived views always share the canonical record's split.
- The manifest warns or fails when a split is empty or grossly imbalanced.
- No split is silently collapsed by a giant cross-document component.

## P1 — required after the first pilot

### 10. Separate generation and judging responsibilities

- [ ] Add an independent procurement task-classification judge.
- [ ] Add an independent grounding and qualification judge.
- [ ] Keep the specialized cross-document and source-ablation judge.
- [ ] Prefer a judge model distinct from the generator.
- [ ] Record when the same model is reused for a pilot.
- [ ] Add bounded rescue behavior for incomplete structured outputs.

### 11. Add best-of-N and diversity selection

- [ ] Group candidates by evidence/path family.
- [ ] Rank candidates using:
  - deterministic validity;
  - judge score;
  - ablation strength;
  - path completeness;
  - naturalness;
  - difficulty;
  - novelty;
  - underrepresented relationship coverage.
- [ ] Retain the strongest candidate per family.
- [ ] Prevent the highest lexical-similarity passages from dominating every
  manual pair.

### 12. Add bounded novelty passes

- [ ] Allow one optional novelty pass for source bundles with additional
  supported questions.
- [ ] Track already covered questions and paths.
- [ ] Stop after the configured pass limit or repeated empty passes.
- [ ] Record incomplete bundles rather than retry indefinitely.

### 13. Add RAG difficulty variants

- [ ] Oracle-only contexts
- [ ] Oracle plus hard topical distractors
- [ ] Missing one required source
- [ ] Wrong-edition distractors
- [ ] Wrong-authority distractors
- [ ] Same-domain but wrong-condition distractors

Acceptance criteria:

- Distractors never change the canonical answer.
- Missing-hop variants have an abstain/incomplete target.
- The golden chunks are never sampled as distractors.

### 14. Run and review a controlled pilot

- [ ] Generate a stratified pilot across every relationship type.
- [ ] Sample at least 100 accepted records for human review.
- [ ] Review rejected samples to find overly strict gates.
- [ ] Record inter-reviewer agreement where multiple reviewers are available.
- [ ] Adjust thresholds only from recorded pilot evidence.

Pilot review dimensions:

- factual correctness;
- answer completeness;
- source attribution;
- temporal correctness;
- qualification preservation;
- genuine multi-source necessity;
- rationale faithfulness;
- question naturalness;
- training usefulness.

## P2 — after two-hop quality is stable

- [ ] Three-document amendment and corrigendum chains
- [ ] Three- and four-hop bridge questions
- [ ] Graph-based complementary-document discovery
- [ ] Deterministic calculation nodes
- [ ] Difficulty calibration using retrieval and answer models
- [ ] Human-reviewed gold evaluation set
- [ ] Multiple group-based evaluation folds
- [ ] Regression dashboards for retrieval and generation

## Curator online startup-capacity research

Status: researched and implemented on 2026-07-28.

Question:

- Does the reference repository contain a justified workaround for Curator
  dispatching local-model requests too slowly?

Evidence:

- Curator's official API reference defines `max_requests_per_minute` and
  `max_tokens_per_minute` as online processor limits, and its LiteLLM guide
  shows callers explicitly configuring both.
- Official Curator 0.1.29 initializes an online tracker with one available
  request and zero available tokens. Its replenishment loop then adds capacity
  linearly and caps it at the configured per-minute limits. Consequently a
  known-safe local endpoint unnecessarily starts with an almost-empty bucket.
- Curator repository commit `39fca352` identifies the same issue and
  initializes both buckets from their configured limits. This is a focused
  Curator token-bucket fix, not a reference-pipeline monkey patch.
- Pilot-004 exposed a separate startup cost: LiteLLM's
  `_ensure_rate_limits()` makes a real completion request to inspect response
  headers even when explicit manual RPM and TPM are both configured. Curator's
  documented and implemented precedence selects the manual limits regardless
  of those headers, so the probe cannot change the effective limits in this
  configuration.
- Pilot-004 measured about 2.5 seconds for the generation probe and 9.5 seconds
  for the cross-generation probe. Skipping this redundant request will remove
  repeated per-stage endpoint work, but it cannot remove the dominant model
  latency measured below.
- The separate `nrl_curator_native_glm52` application does not patch
  `OnlineStatusTracker` and its history contains no
  `available_request_capacity` change. It configures official backend knobs
  (10,000 RPM, 100,000,000 TPM, and endpoint-specific concurrency) and patches
  Curator 0.1.27's deferred retry scheduler. The retry fix prevents a
  low-concurrency retry tail but cannot reduce a model server's time to
  generate one response.
- The `pilot-001` logs confirm manual limits of 10,000 RPM and 100,000,000 TPM
  were active. They also show that all five cross-document tasks started with
  no rate-limit errors. The stage still took about 7 minutes 53 seconds because
  the GLM endpoint returned its three usable responses after roughly 4–8
  minutes; two generated requests had no materialized response. Therefore
  startup capacity is a real general throughput bug but was not the dominant
  cause of that pilot's runtime.

Decision:

- [x] Initialize explicitly configured request and token buckets at full
  capacity for combined and separate token-limit strategies.
- [x] Preserve the conservative one-request/zero-token state when limits are
  absent and Curator must discover them.
- [x] Cover configured combined, configured separate, and unconfigured startup
  behavior with unit tests.
- [ ] When both manual RPM and manual TPM are present, skip the LiteLLM
  header-discovery completion. Preserve discovery whenever either manual limit
  is absent, and cover both branches with unit tests. Do not edit the imported
  processor while pilot-004 is running.
- [ ] Separately investigate the GLM server's long generation latency and the
  two cross-document responses that were not materialized; do not mislabel
  these as Curator throttling.

Pilot-004 live evidence:

- All five single-document generation requests were created within
  milliseconds. The three materialized GLM responses finished after about 99,
  170, and 296 seconds; Curator reported zero API, rate-limit, or other errors.
- Curator counted all five calls as succeeded, but only three response rows
  were materialized. The other two requests parsed to no dataset rows and were
  copied to `failed_requests.jsonl`; this is an observability/coverage problem,
  not proof of an API failure.
- The one batched Nemotron judge request completed in about 15.5 seconds.
- All five cross-document requests were dispatched together after the startup
  probe. The first materialized response arrived after about 84 seconds while
  the remaining requests were still outstanding. Thus the displayed low
  lifetime requests/minute is not evidence of serial Curator dispatch.

Pilot-004 terminal audit:

- The run completed with manifest status `partial`, not `complete`. It produced
  five canonical records: three `qa`, one `qa_cot`, and one
  `cross_document_qa`; it produced no `cross_document_qa_cot`.
- Single-document coverage was 3/5 planned requests materialized, yielding six
  generated records and four accepted records. Cross-document coverage was
  1/5 planned requests materialized, yielding one accepted record. Missing
  request IDs are preserved in the manifest.
- Both drafting seeds generated, passed deterministic checks and independent
  judging, and were exported with seed-compatible citation IDs plus structured
  manual/tender citation details. No drafting record was rejected.
- The final cross-generation stage took about eight minutes after all five
  requests were dispatched together, but only one request materialized a
  record. Nemotron judge stages completed their single batches in roughly
  4–16 seconds. The dominant observed latency and materialization loss remain
  on GLM generation, not Nemotron judging or Curator rate limiting.
- Citations on accepted QA and cross-document records include manual ID/title,
  source file, page, section, chunk ID, exact quote, and character offsets.
  The inspected accepted cross-document comparison uses one exact quotation
  from each manual and passes both-source ablation.
- Two otherwise fully grounded `qa_cot` records were rejected only because the
  Nemotron judge placed multiple individually exact but non-contiguous source
  excerpts into its single `answer_quote`. The deterministic citations resolve
  every excerpt exactly, and all other judge dimensions were true with score
  5. This is a judge-witness serialization defect: a concatenation of separated
  quotations is not itself one exact substring. Research and implement a
  general witness contract (for example, a typed list of independently
  resolved quotes) rather than weakening grounding or special-casing these two
  records.

Pilot-003 correction and retry-tail finding:

- Pilot-003 proves that the earlier inference about the cross-document delay
  was incomplete. All five calls were dispatched, but only one produced a
  terminal response record (106 seconds). Four original requests were written
  to `failed_requests.jsonl`; Curator then kept the stage active for about
  11.5 minutes.
- Curator 0.1.29's official `RequestProcessorConfig` defaults
  `max_retries=10`. This pipeline passed explicit RPM, TPM, concurrency, and
  `require_all_responses=false`, but did not pass `max_retries`; structured
  output/schema failures therefore incurred the default retry tail.
- The reference project independently documents and configures
  `max_retries: 1` because repeating deterministic truncation or structured
  output failures consumes time without changing the contract.
- PR #734's work-conserving retry implementation is already merged into this
  checkout as commit `f956b921` and is imported from the editable environment.
  It prevents deferred low-concurrency retries, but intentionally still honors
  the configured retry count. Reapplying its old monkey patch cannot solve an
  excessive retry count.
- The reference monkey patch was read in full from
  `src/nrl_curator_native/compat.py`, including its installer,
  compatibility self-test, and retry tests. Its committed implementation from
  reference commit `d9364846` is the same scheduling strategy now implemented
  directly in Curator core here: there is no deferred `retry_queue`, each
  request retries within its active task, and its semaphore permit is released
  exactly once.
- Runtime inspection with `.curator/bin/python` resolves
  `BaseOnlineRequestProcessor` to this checkout's editable source at
  `src/bespokelabs/curator/request_processor/online/base_online_request_processor.py`;
  its active method signature has no `retry_queue`, and
  `process_requests_from_file` gathers the concurrently scheduled first-attempt
  tasks. This rules out accidentally importing an unpatched wheel.
- `pilot-003` predates commit `ca31c9df`, which added and propagated
  `max_retries: 1`. Its manifest records no `max_retries` field, proving that
  the observed run used the earlier configuration and therefore cannot
  validate or invalidate the current retry-tail fix.

Additional decision:

- [x] Set an explicit, model-profile-configurable `max_retries: 1` for both
  generation and judge roles and pass it through `backend_params`.
- [x] Keep `require_all_responses=false`; missing requests remain visible in
  coverage and fail the run's quality requirements instead of aborting before
  audit artifacts are written.
- [ ] Pilot-004 must confirm that a structured-output failure incurs at most
  one retry and that stage duration no longer contains the ten-retry tail.

Primary references:

- [Curator API reference](https://docs.bespokelabs.ai/bespoke-curator/api-reference)
- [Curator LiteLLM guide](https://docs.bespokelabs.ai/bespoke-curator/how-to-guides/using-litellm-with-curator)
- [Official Curator GitHub repository](https://github.com/bespokelabsai/curator)

## Holistic pilot-quality and OCR research

Status: researched on 2026-07-28; implementation intentionally deferred until
the complete failure system was examined.

### Scope

The `pilot-001` failures must not be treated as isolated prompt or regular
expression bugs. The quality system spans:

1. source registration and OCR provenance;
2. page/chunk preparation;
3. representative input selection;
4. task planning;
5. structured generation and bounded recovery;
6. deterministic grounding checks;
7. independent model judgment;
8. leakage-safe exports and complete run reporting;
9. controlled human review before scale.

### Research basis

- [Official Chandra OCR repository](https://github.com/datalab-to/chandra)
  documents page-oriented PDF conversion to structured Markdown/HTML/JSON and
  reports materially different accuracy across tables, layout, scans, and
  languages. Page completion alone is therefore necessary but not a semantic
  accuracy guarantee.
- [Official Chandra OCR 2 model card](https://huggingface.co/datalab-to/chandra-ocr-2)
  identifies the exact model family, supported output forms, benchmark
  categories, throughput, and model-weight license.
- [Official Curator structured-output guide](https://docs.bespokelabs.ai/bespoke-curator/getting-started/structured-output)
  and [API reference](https://docs.bespokelabs.ai/bespoke-curator/api-reference)
  establish Pydantic response formats and parse-stage conversion. A valid
  object schema does not establish that free text inside a string is grounded,
  correctly formatted, or complete.
- [Official vLLM structured-output documentation](https://docs.vllm.ai/en/latest/features/structured_outputs/)
  establishes schema-constrained decoding for OpenAI-compatible servers. The
  pipeline must still perform domain validation after decoding.
- [Pydantic validator documentation](https://docs.pydantic.dev/latest/concepts/validators/)
  supports deterministic before/after validation, but repair must be narrow,
  observable, and must not invent procurement content.
- [Instructor retry documentation](https://python.useinstructor.com/concepts/retrying/)
  supports validation-aware retries. Recovery must remain bounded and preserve
  the failed attempt for audit rather than silently reducing expected output.
- Yehudai et al., [Achieving Human Parity in Content-Grounded Datasets
  Generation](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a774503daed55eb53c634847ae071ec7-Abstract-Conference.html),
  separates content preparation, generation, and faithfulness filtering.
- Min et al., [FActScore](https://ai.meta.com/research/publications/factscore-fine-grained-atomic-evaluation-of-factual-precision-in-long-form-text-generation/),
  evaluates long-form output as atomic claims supported by reliable sources.
  Drafting validation should likewise judge material claims, not only declared
  evidence lists.
- Alberti et al., [Training Question Answering Models From Synthetic
  Data](https://aclanthology.org/2020.emnlp-main.468/), uses round-trip
  consistency filtering for synthetic QA.
- Rajpurkar et al., [SQuAD 2.0](https://aclanthology.org/P18-2124/), includes
  plausible unanswerable questions to teach abstention. It does not justify an
  accidental abstention ratio produced by sampling cover pages.
- Vacareanu et al., [General Purpose Verification for Chain of Thought
  Prompting](https://arxiv.org/abs/2405.00204), validates individual reasoning
  steps for relevance, accuracy, and logical consistency. Answer correctness
  alone is insufficient for rationale supervision.
- Wang et al., [Diversity Measurement and Subset Selection for Instruction
  Tuning Datasets](https://arxiv.org/abs/2402.02318), and Liu et al.,
  [What Makes Good Data for Alignment?](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6091f2bb355e960600f62566ac0e2862-Abstract-Conference.html),
  support explicit quality, complexity, and diversity measurement rather than
  prefix sampling.
- Document-group isolation is required to avoid evaluation leakage; related or
  amendment-connected manuals must remain atomic split groups. A tiny pilot is
  a coverage test, not evidence that train/validation/test proportions have
  converged.

### Local data and OCR audit

Observed under `data/`:

- `manuals.yaml` registers 19 unique documents with 19 unique paths: 16
  Government of India Markdown sources and three NRL PDFs.
- The three PDF SHA-256 values exactly match their `.chandra-cache.json`
  records.
- Chandra metadata reports 79 consultancy/services pages, 270 goods pages, and
  148 works pages. The corpus page parser resolves exactly 79, 270, and 148
  pages respectively, with no empty pages or Unicode replacement characters.
- The cache records the input hash, `vllm` method, and pagination flag but not
  the OCR model identifier, package/model revision, generation parameters, or
  output hashes. An input match therefore cannot prove reproducible extraction.
- The loaded corpus contains 3,006 chunks. Of these, 621 contain image Markdown
  and 679 contain HTML layout elements, mainly tables. Image descriptions and
  duplicated cover text can generate low-value questions; table markup may
  contain essential policy and must not be stripped indiscriminately.
- `--limit 5`, `--limit 20`, and even `--limit 100` currently select only the
  beginning of `goods_2017`. The five-row pilot sampled pages 1, 2, 4, 6, and
  8, including cover/front matter. This directly explains the narrow authority
  coverage and elevated unanswerable share.
- Drafting seed IDs resolve to this corpus's current chunk IDs. Both seeds were
  generated in `pilot-001`; neither passed the final drafting gate.

### `pilot-001` quality audit

- Ten canonical records were accepted: seven single-document QA, two
  cross-document QA, and one cross-document QA-with-rationale.
- All accepted records were assigned to `train`; a ten-record, highly connected
  pilot is too small to assess requested split proportions.
- Three of seven single-document records are abstentions. The observed ratio is
  caused by front-matter prefix sampling and is not an authored target.
- `qa_cot_sft.jsonl` contains the same cross-document rationale also written to
  `cross_document_qa_cot_sft.jsonl`; there is no accepted single-document
  `qa_cot` record.
- The three cross-document records are well grounded and require both sources,
  but two of five requested cross-document generations did not materialize as
  dataset rows. The run manifest does not report that incompleteness.
- Both drafting generations were rejected. The NIT output used `<br>` instead
  of newline characters. The LD output was one line, invented `NRL Procurement
  Division`, and exposed a numeric checker that treats list ordinals and a
  hyphenated tender ID as policy numbers.
- The manifest reports accepted QA task counts only. It omits expected versus
  materialized requests, deterministic/model rejection counts and reasons,
  drafting counts, repair counts, latency, source coverage, OCR fingerprint,
  and model/config fingerprints.

### Design decisions

#### Source and OCR integrity

- [ ] Validate registered source uniqueness, source hashes, OCR page count, and
  non-empty pages before generation.
- [ ] Extend OCR cache provenance with OCR model, package/model revision,
  relevant command settings, and hashes of canonical Markdown and metadata.
- [ ] Emit a corpus-quality report with page/chunk counts, short-page outliers,
  image-only/front-matter candidates, HTML/table counts, replacement
  characters, and seed-anchor resolution.
- [ ] Remove image references and generated image captions from text supplied
  to QA generation while retaining page provenance. Preserve meaningful table
  structure through a tested canonical representation.
- [ ] Do not drop cover/front matter globally: authority, edition, and issuance
  facts can be useful. Classify it and sample it deliberately at a bounded rate.

#### Representative planning

- [ ] Replace prefix limiting with deterministic round-robin stratification by
  manual, authority/source category, document family, page band/section, and
  content class.
- [ ] Make a pilot coverage plan explicit before model calls. Report requested,
  generated, accepted, and rejected counts per manual, task type, question
  type, answerability, authority, and relationship type.
- [ ] Treat `--limit` as total planned source units, not the first N corpus
  rows. A limit smaller than the number of required strata must fail clearly or
  report which strata were intentionally omitted.
- [ ] Use an authored/configured answerable/unanswerable target range. Generate
  unanswerable examples from plausible evidence gaps or counterfactual
  removals, not because cover text lacks an answer.

#### QA and rationale contracts

- [ ] Plan `qa` and `qa_cot` separately. Do not depend on the generator to
  choose the run's task distribution.
- [ ] Assign `qa_cot` only to evidence windows with at least two connected
  material claims or operations. Never force a decorative rationale onto a
  direct lookup.
- [ ] Validate every rationale step for an explicit operation, grounded inputs,
  supported output, connectivity to adjacent steps, and contribution to the
  final answer.
- [ ] Keep single-document QA/CoT exports distinct from cross-document QA/CoT
  exports. A combined export, if desired, must have a different explicit name.
- [ ] Preserve abstention records separately in metrics and optionally in a
  dedicated export so their training weight can be chosen from downstream
  validation rather than accidental corpus order.

#### Grounded drafting

- [ ] Normalize only lossless surface variants before validation: line endings,
  surrounding whitespace, and explicit `<br>` tags to newline characters.
  Record every repair. Do not infer headings, facts, clauses, or missing text.
- [ ] Reject remaining HTML markup in final plain-text drafting records.
- [ ] Remove ordered-list markers before numeric fact comparison and ensure
  hyphenated identifiers are compared as identifiers, not partial numbers.
  Continue rejecting genuine unsupported percentages, amounts, dates,
  durations, emails, and identifiers.
- [ ] Extract material drafting claims/fields (organization, authority,
  contacts, references, thresholds, remedies, conditions, exceptions) and
  require support from tender facts or manual evidence. A model-declared
  `tender_facts_used` list is not sufficient.
- [ ] Explicitly reject unsupported labeled authority/organization fields such
  as the observed `NRL Procurement Division`.
- [ ] Send deterministically valid drafts to the independent judge; do not let a
  formatting false positive prevent semantic judgment. Do not weaken
  deterministic grounding to increase acceptance.

#### Completeness, recovery, and judging

- [ ] Track expected request IDs through generation, parse, deterministic
  validation, judge, and export. No requested row may silently disappear.
- [ ] Quarantine malformed/missing outputs with exact failure class and raw
  cache lineage.
- [ ] Add bounded rescue only for explicitly recoverable failures such as
  schema truncation or lossless formatting. Rescue uses its own cache stage and
  attempt budget; it must not retry deterministic unsupported content
  indefinitely.
- [ ] Enforce generator/judge independence for production profiles and record
  endpoint/model identities. Add a small adversarial judge preflight containing
  supported, unsupported, qualification-losing, and malformed examples.
- [ ] Report judge score distributions and disagreement with deterministic
  checks. An all-5 pilot is a calibration warning, not proof of perfect data.

### Pilot-007 validation-overhaul research record (2026-07-29)

Status: deterministic implementation complete; local regression tests pass.
A user-run validation pilot remains required.

Capability and observed failures:

- Pilot-007 completed every provider request and eliminated the earlier missing
  response-file crash, but accepted no `cross_document_qa_cot` record.
- Three otherwise supported records exposed deterministic or judge-witness
  false negatives: `not` matched the substring in `note`; categorical
  paraphrases of `shall` were rejected solely by lexical mismatch; and Gemma
  concatenated multiple individually exact evidence spans into one judge quote.
- Three other cross-document records contained genuinely modified or
  misattributed evidence. Any repair must continue rejecting those records.

Verified local and reference findings:

- `validation.py` currently finds qualifiers with unrestricted substring
  membership. It therefore cannot distinguish the token `not` from `note`.
- The single- and cross-document judge parsers require every judge witness to
  be one literal substring of the entire source passage. They do not recognize
  a witness that losslessly concatenates already-verified, non-contiguous
  evidence spans.
- The reference project normalizes harmless whitespace and removes one balanced
  decorative outer quote pair before exact grounding. It persists only the
  canonical source-grounded quote and rejects changed internal wording. It
  does not implement concatenated judge-witness verification.
- Pilot-007's accepted citations already retain exact source text, chunk/page
  identity, and offsets. Judge-witness tolerance must not mutate that primary
  evidence or replace its deterministic checks.

Research-supported decision:

- Tokenize qualifier checks with word boundaries and compare modality classes,
  so lexical substrings cannot trigger a failure and equivalent mandatory or
  permissive forms can preserve modality. Continue treating lost negation,
  conditions, exceptions, and exclusivity as failures.
- Keep generator evidence exact. For judge `answer_quotes`, accept an exact
  source span after whitespace/balanced-wrapper normalization, or a lossless
  concatenation of two or more already deterministically verified evidence
  spans in their original record order. Do not allow fuzzy matching,
  punctuation changes, reordered spans, partial-token matching, or
  model-authored replacement text.
- Judge semantic booleans, taxonomy agreement, score threshold, deterministic
  evidence checks, and cross-source ablation remain independently mandatory.
  The change repairs witness serialization only; it does not turn a judge score
  into proof of grounding.

Alternatives rejected:

- Removing deterministic modality checks: increases acceptance by discarding a
  material procurement safeguard.
- Fuzzy quote similarity: may silently accept changed thresholds, negations,
  authorities, or remedies.
- Trusting a score-5 judge without a grounded witness: conflicts with the
  pipeline's independent traceability contract.
- Rewriting generated evidence to the nearest source span: can conceal
  attribution errors and corrupt audit lineage.

Known risks and validation:

- Modality equivalence is necessarily narrower than full natural-language
  entailment. Regression tests must include preserved and genuinely dropped
  mandatory, permissive, conditional, negative, and exception cases.
- Concatenation tolerance could be over-broad unless it is limited to complete,
  pre-verified evidence items in record order. Tests must reject altered,
  partial, reordered, and foreign-source spans.
- Unit tests establish deterministic behavior only. The conclusion that this
  improves production yield remains provisional until the user runs another
  bounded pilot; Codex must not run that model-backed pilot.

Implementation result:

- [x] Match qualifiers as tokens rather than unrestricted substrings.
- [x] Preserve mandatory/permissive modality through narrow, tested
  equivalence classes while continuing to reject a `shall` to `may` weakening.
- [x] Accept whitespace-normalized, balanced-wrapper judge witnesses and
  lossless concatenations of consecutive, already-verified evidence items.
- [x] Continue rejecting altered, reordered, partial, and foreign-source
  witnesses; persisted generator evidence and citations remain unchanged.
- [x] Re-evaluate Pilot-007 locally without model calls: the two holiday-list
  records and performance-notice record clear their false deterministic
  failures; the hospitality/gifts judge witness clears its serialization
  failure; the truncated final-payment evidence remains rejected.
- [x] Keep the malformed LD judge witness rejected because it changes source
  punctuation with an unmatched trailing quote. The existing prompt already
  requires each judge witness to be a contiguous verbatim span, so a future
  well-formed response can pass without weakening validation.
- [x] Add regression coverage and pass the complete local procurement test
  suite and Ruff checks.
- [ ] Confirm production yield and `cross_document_qa_cot` coverage through a
  bounded user-run pilot.

Primary and official sources:

- [Bespoke Curator repository](https://github.com/bespokelabsai/curator)
  documents Pydantic structured responses followed by application-defined
  `parse` processing, caching, and recovery.
- [Google Gemini structured-output documentation](https://ai.google.dev/gemini-api/docs/structured-output)
  explicitly states that schema-valid JSON does not guarantee semantically
  correct values and recommends application validation.
- [Hugging Face structured-output documentation](https://huggingface.co/docs/huggingface_hub/guides/inference#structured-outputs--json-mode)
  distinguishes schema conformance from downstream provider/model behavior.
- Alberti et al.,
  [Synthetic QA Corpora Generation with Roundtrip Consistency](https://aclanthology.org/P19-1620/),
  supports filtering generated QA through an independent answer-consistency
  check instead of accepting schema-valid generations directly.
- Gao et al.,
  [Enabling Large Language Models to Generate Text with Citations](https://aclanthology.org/2023.emnlp-main.398/),
  treats citation correctness and answer support as distinct verifiability
  dimensions.

#### Exports, splits, and run manifest

- [ ] Write the manifest last, atomically, with a terminal status of
  `complete`, `partial`, or `failed`.
- [ ] Include code revision, source/OCR fingerprints, non-secret model/config
  fingerprints, stage timing, expected/materialized/accepted/rejected counts,
  rejection reasons, repairs, retries, cache reuse, and coverage distributions.
- [ ] Never label a partial run complete. Preserve audit and rejection files on
  failure, but withhold publishable final files when required gates fail.
- [ ] Keep amendment/edition/authority-connected documents in atomic split
  groups. Report absent splits as a pilot-size limitation; require minimum
  independent group counts before producing evaluation claims.
- [ ] Add a frozen, human-reviewed evaluation set outside generated training
  data. Generated `eval.jsonl` is useful for pipeline testing but is not an
  independent gold benchmark.

### Acceptance criteria before the next pilot

- [ ] Corpus preflight passes and writes an auditable quality report.
- [ ] A fixed-seed pilot covers multiple manuals, both authority classes,
  multiple document families/page bands, single QA, genuine single QA-CoT,
  cross-document QA, and cross-document QA-CoT.
- [ ] Every planned request has a terminal lineage state.
- [ ] Final task-specific exports are non-overlapping and their counts reconcile
  exactly with canonical records and manifest statistics.
- [ ] Both drafting seeds either pass all deterministic and independent-judge
  gates into `drafting.jsonl`, or remain rejected with accurate non-spurious
  reasons.
- [ ] No unsupported authority, identifier, numeric value, email, condition,
  exception, or remedy is present in accepted drafting.
- [ ] Human review of all records in the small pilot confirms grounding,
  naturalness, task usefulness, and rationale faithfulness before scale.

## Quality metrics

Every generation run must report the following.

### Grounding

- Exact evidence resolution rate
- Material claim coverage
- Numeric/date/name/email support rate
- Modality and exception preservation
- Authority correctness
- Temporal correctness

### Multi-hop authenticity

- Both-source full-answer rate
- A-only full-answer rate
- B-only full-answer rate
- Connected-path rate
- Unsupported relationship rate
- Missing-hop abstention accuracy

For answerable cross-document records, the desired pattern is a high
both-source rate and very low A-only and B-only full-answer rates.

### Diversity and coverage

- Record counts by question type
- Record counts by relationship type
- Manual-pair coverage
- Section and source-chunk coverage
- Semantic duplicate rate
- Difficulty distribution
- Answer-style distribution

### Retrieval

- Recall of all oracle chunks
- Partial-hop retrieval rate
- Wrong-edition confusion
- Wrong-authority confusion
- Oracle-context answer accuracy
- Retrieved-context answer accuracy

## Research basis

- [HotpotQA](https://aclanthology.org/D18-1259/) — multi-document questions
  with supporting-fact supervision.
- [2WikiMultiHopQA](https://aclanthology.org/2020.coling-main.580/) — explicit
  evidence paths and comprehensive reasoning-step evaluation.
- [MuSiQue](https://aclanthology.org/2022.tacl-1.31/) — bottom-up connected
  question composition and unanswerable contrasts.
- [Counterfactual Multihop QA](https://aclanthology.org/2023.acl-long.231/) —
  reducing disconnected reasoning through counterfactual analysis.
- [MIMG](https://aclanthology.org/2025.acl-long.1316/) — separate single-hop
  generation, question merging, multiple sampling, and verification.
- [HopWeaver](https://aclanthology.org/2026.acl-long.1295/) — complementary
  document discovery and authentic cross-document reasoning paths.
## Task, QA format, and user-role taxonomy clarification (2026-07-28)

Research conclusion before implementation:

- `task` describes the underlying procurement capability requested by the user.
  It is not the serialization or reasoning format. For example, a request to
  produce tender language is `drafting`; a request to populate NIT form fields is
  `nit_filling`.
- `task_type` describes the supervision format and reasoning shape: `qa`,
  `qa_cot`, `cross_document_qa`, or `cross_document_qa_cot`. Therefore both a
  drafting question and an NIT-filling question may independently be emitted as
  QA or QA-with-CoT without changing their capability label.
- `persona` describes the actor whose work or information need the example
  represents. It is independent of both `task` and `task_type`.
- Seed-authored `task` values are authoritative for examples derived from that
  seed and must be copied, not reclassified from surface words. Corpus-derived
  examples must select a supported task and persona from committed taxonomies,
  and the independent judge must verify both.
- Do not infer `nit_filling` merely because “NIT” appears. Drafting an NIT is
  `drafting`; entering structured NIT fields in an e-procurement system is
  `nit_filling`.

Primary references:

- [Government of India GeM overview](https://assets-bg.gem.gov.in/resources/upload/shared_doc/gem-overview-ppt-12-august-2024_1724322936.pdf),
  “Buyer User Roles Based on Segregation of Duties”: Primary User, Buyer,
  Consignee, Indentor, DDO, PAO, and Technical Evaluator.
- [Government eProcurement role guidance](https://www.gepnic.gov.in/admin/WriteReadData/File/1663216819.pdf):
  Tender Creator is responsible for filling tender details, which supports a
  distinct `nit_filling` capability.
- [Government of India Manual for Procurement of Works, 2025](https://doe.gov.in/files/manuals_documents/Works_Manual_SE_2025.pdf):
  an NIT is a tender-process artifact with legal importance, supporting
  separation of tender drafting from portal field-entry work.
- [Google Research FLAN Collection](https://research.google/pubs/the-flan-collection-designing-data-and-methods-for-effective-instruction-tuning/):
  balancing diverse tasks and mixing ordinary and chain-of-thought prompt
  settings improves instruction tuning; task identity and prompt/reasoning
  setting are separate experimental axes.
- [Hugging Face dataset-card metadata](https://huggingface.co/docs/huggingface_hub/en/package_reference/cards):
  `task_categories` and `task_ids` are explicit metadata, reinforcing documented
  task identity.

## Pilot-005 response persistence and citation ordering (2026-07-28)

Status: implemented and locally verified; a model-backed rerun remains user-owned.

Pilot evidence and root cause:

- The GLM endpoint was reachable. Both generation stages reported 5/5 provider
  successes, zero API errors, and zero rate-limit errors.
- Single-document generation persisted only four of five successful responses;
  cross-document generation persisted none of five. Curator then raised
  `No responses files found` after the cross stage.
- Curator's online writer called the pipeline `parse()` before persistence and
  silently returned when parsing produced `[]` or `None`. This conflated a
  successful provider response filtered by deterministic validation with a
  missing or failed request.

Implemented contract:

- Persist every completed provider response, including responses whose parsed
  row set is empty. Store `parsed_response_message=null` for filtered responses.
- If at least one provider response succeeded but every row is filtered,
  materialize an empty dataset and allow the calling pipeline to continue.
  Preserve the existing hard failure when every provider request genuinely
  failed.
- Future user-facing and audit JSONL writers preserve all fields but serialize
  top-level `citations` last. `citation_details`, when present, immediately
  precedes `citations`.
- Completed historical run artifacts such as `outputs/pilot-004` are not
  rewritten; the presentation rule applies to newly written output.

Research basis:

- [Curator Key Concepts](https://docs.bespokelabs.ai/bespoke-curator/getting-started/key-concepts)
  defines `parse` as converting each provider response into zero or more final
  rows; Curator's own documented feature-extraction example returns `[]` when a
  response cannot be converted.
- [RFC 8259](https://www.rfc-editor.org/rfc/rfc8259) defines a JSON object as an
  unordered collection, so citation-last is a stable presentation contract, not
  a semantic JSON requirement.
- [Python `json` documentation](https://docs.python.org/3/library/json.html)
  states that encoders preserve input order by default, which makes the
  requested presentation deterministic in JSONL.

Local verification:

- `tests/unittests/test_online_retry_scheduler.py`: provider responses survive
  parse filtering and all-filtered successful responses return an empty dataset.
- `tests/nrl_procurement/test_pipeline.py`: compact drafting, canonical, QA,
  QA-CoT, cross-document QA, and cross-document QA-CoT JSONL serialize
  `citations` as the final top-level key.

## Pilot-006 output and content audit (2026-07-28)

Status: audited; persistence fix confirmed, but the run is not suitable as a
full acceptance pilot because cross-document yield is zero.

Observed results:

- All provider calls completed and all raw responses were persisted: 5
  single-document generation responses, 9 single-document judge responses, 5
  cross-document generation responses, 2 drafting generation responses, and 2
  drafting judge responses. The Pilot-005 missing-response crash did not recur.
- Final accepted output contains 8 single-document records (7 QA and 1 QA with
  rationale) plus 2 drafting records. One additional single-document QA-with-
  rationale record was rejected.
- All 5 cross-document provider responses contain structured candidates, but
  deterministic parsing filtered every candidate. Consequently all
  cross-document audit, rejected, and SFT files are empty and the manifest
  correctly reports the run as `partial`.
- Every non-empty JSONL file is syntactically valid. Whenever a top-level
  `citations` field is present, it is the final serialized key. Drafting
  `citations` intentionally contains compact citation IDs; full traceability is
  carried by the preceding `citation_details`.
- Citation objects contain manual ID/title, source file, page, section, chunk,
  quote, and offsets. Drafting records additionally trace authored tender facts
  to the tender and seed IDs.

Quality findings:

- The 8 accepted single-document records are generally grounded and useful:
  seven received judge score 5 and one received score 4.
- The score-4 final-payment/PBG record relies on a source quote truncated at
  “from the”. It is understandable but should not qualify as exemplary
  production evidence.
- Both drafting records received score 5, preserve the authored seed facts, and
  provide source traceability. The NIT record retains the requested
  organization, GeM two-part bidding, E-File reference, job name, contacts, and
  footer. The LD record retains the delayed-portion basis, 0.5% weekly rate, 5%
  cap, cancellation limitation, and risk-purchase meaning.
- The rejected single-document QA-with-rationale record also received score 5.
  It was rejected only because the judge returned an `answer_quote` made from
  two exact excerpts joined by an ellipsis, while the acceptance checker
  requires one contiguous source substring.

Root causes requiring a general fix:

- The numeric regex currently consumes a following word (`2019 Manual`, `2025
  Manuals`, `10 percent`) and compares that combined token literally with claim
  evidence. It also validates answer metadata dates only against claim quotes.
  These checks falsely rejected otherwise grounded cross-document candidates.
- One of five cross-document candidates genuinely violated its preassigned
  format by returning `cross_document_qa_cot` for a
  `cross_document_qa` request; this rejection is correct.
- Deterministic parse failures are preserved in raw Curator responses but are
  not materialized into `cross_rejected.jsonl`, so the user-facing rejection
  audit still lacks reason codes for parse-filtered candidates.

Required follow-up before the next pilot:

- [x] Replace numeric/quantity detection with typed normalization that checks
  actual numeric values and units without swallowing arbitrary following words,
  and allow source metadata to support authority/version dates.
- [x] Make judge quotation evidence structurally multi-span or require and validate
  one exact contiguous quote consistently.
- [x] Emit deterministic rejection audit rows with explicit reason codes before
  discarding candidates.
- [x] Add a source-integrity gate for truncated evidence fragments and regression
  tests using all five Pilot-006 cross-document response shapes.

Implementation status (2026-07-28):

- Typed quantity validation now separates numeric values from arbitrary prose,
  canonicalizes `%`, `percent`, and `per cent`, retains explicit duration units,
  and admits document identity/version values only when present in supplied
  metadata. Pilot-006 forms such as `2019 Manual`, `2025 Manuals`, and
  `10 (ten) per cent` have regression coverage.
- The judge schema now uses `answer_quotes`, a bounded list of zero to three
  independent verbatim spans. Each span is checked separately against the
  supplied source, preventing ellipsis-joined excerpts from being mistaken for
  a single quotation. This is a provider-neutral JSON Schema array contract.
- Generator parse stages now materialize every schema-valid candidate with
  `deterministic_checks.passed` and explicit `issues`. Generation audit files
  retain all candidates; only passing records proceed to judging, and
  deterministic failures are also written to the corresponding rejected file.
- Focused verification after these three changes: 20 tests passed and Ruff
  passed. No model-backed pipeline was run.
- Source-integrity verification now rejects high-confidence dangling evidence
  endings such as `from the`, without imposing a punctuation requirement that
  would reject valid headings or table cells. Typed normalization also excludes
  trailing prose punctuation and recognizes a parent section locator such as
  `2` when the supplied evidence contains subsection `2.3`.
- Final stored-response replay: four of the five Pilot-006 cross-document
  candidates now pass deterministic validation. The fifth retains the correct
  `planned_task_type_mismatch:cross_document_qa` rejection. Combined focused
  verification: 25 tests passed and Ruff passed.

Research basis:

- [Python `re` documentation](https://docs.python.org/3/library/re.html):
  explicit groups and boundary assertions permit typed extraction without the
  former broad `\s+\w+` suffix.
- [JSON Schema array reference](https://json-schema.org/understanding-json-schema/reference/array):
  homogeneous lists use `items`, with bounded cardinality represented by array
  length constraints.
- [Curator API reference](https://docs.bespokelabs.ai/bespoke-curator/api-reference):
  `parse()` converts a provider response into one or more output rows.
- [Hugging Face Datasets processing](https://huggingface.co/docs/datasets/process):
  rows can be materialized first and filtered afterward, allowing audit rows to
  remain observable while quality gates stay strict.
- [Unstructured chunking documentation](https://docs.unstructured.io/open-source/core-functionality/chunking):
  preserving document elements and semantic boundaries is preferred to blind
  character splitting.
- [spaCy sentence-boundary documentation](https://spacy.io/usage/linguistic-features#sbd):
  robust general sentence-boundary detection is model-based, while punctuation
  segmentation is only a simpler rule-based alternative. The implemented gate
  therefore targets only high-confidence dangling function words rather than
  pretending punctuation alone proves completeness.

Pilot artifacts:

- `outputs/pilot-006/files/manifest.json`
- `.curator_working/pilot-006/cross_generation/035c2fe81112a763/responses_0.jsonl`

## Pilot-008 judge-batch cardinality and identity integrity (2026-07-29)

Pilot-008 completed without transport failures, but the cross-document judge
returned the same judgment three times for
`nrlxd-0626e718e2f5fb3eaf5c`. The pipeline materialized every returned item,
inflating three unique accepted cross-document records to five rows and
propagating the duplicate into canonical, SFT, RAG, and evaluation exports.

Research conclusions before implementation:

- Curator intentionally lets `parse(input, response)` return multiple output
  dictionaries. Curator validates and transports the structured response but
  does not know the pipeline's expected record-ID set; cardinality and identity
  correspondence therefore belong in the pipeline parse boundary.
- JSON Schema `uniqueItems` compares complete array items, not a selected
  property such as `record_id`. Two judgments with the same ID but different
  decision content can still satisfy `uniqueItems`, so schema-only uniqueness
  is insufficient.
- Pydantic validates the declared model shape. Cross-field or collection-level
  business invariants can be added with validators, but expected IDs come from
  the current input batch and must still be compared during parsing.
- A malformed judge batch must not be partially trusted. Duplicate, missing, or
  unexpected IDs indicate that response-to-input correspondence is broken.
  Quarantine all returned judgments from that batch with explicit audit reason
  codes rather than silently choosing the first or last response.
- Export-time stable-ID uniqueness is still required as defense in depth.
  Duplicate IDs must fail the quality gate rather than being counted as
  additional accepted examples.

Required implementation:

- [x] Add a reusable exact-ID-set/cardinality validator for all batched judges.
- [x] Quarantine malformed judge batches with duplicate, missing, and unexpected
  ID diagnostics; do not accept a subset from an invalid batch.
- [x] Enforce unique stable record IDs before canonical, SFT, RAG, evaluation,
  and drafting export.
- [x] Count unique accepted records in the manifest and report duplicate
  rejection counts explicitly.
- [x] Add regressions reproducing Pilot-008's triplicated judgment and covering
  missing and unexpected IDs.

Implementation status:

- Single-document and cross-document judge parsers now compare the returned ID
  multiset with the exact expected batch before reading any decision. Any
  duplicate, missing, unexpected, or cardinality mismatch quarantines every
  expected original as rejected with stable diagnostic reason codes.
- Quarantine preserves one audit row per expected input, so malformed model
  output cannot inflate coverage or make unrelated records silently disappear.
- Accepted procurement and drafting records pass a stable-ID uniqueness gate
  before any canonical or task-specific export is written.
- The terminal manifest reports quarantined record counts separately for
  single-document and cross-document judges. Since invalid batches cannot enter
  `accepted`, exported statistics count unique accepted identities.
- Regression coverage reproduces Pilot-008's duplicated cross-judge ID and
  verifies duplicate, missing, unexpected, cardinality, and export-gate
  behavior. Complete local procurement verification: 42 tests passed; Ruff
  passed. No model-backed pipeline was run.

### Pilot-009 singular judge response research (2026-07-29)

Status: implemented and locally verified; user-run pilot validation pending.

Observed behavior:

- The fail-closed identity gate worked, but Gemma returned duplicate arrays for
  four one-record cross-document judge requests. Those four records were safely
  quarantined, reducing useful pilot yield.
- Configuration deliberately uses `judge_batch_size: 1` for the private
  8,192-token Gemma endpoint, yet both judge schemas still ask the model to
  return an arbitrarily sized `judgments` array. The transport contract exposes
  cardinality freedom that the request does not have.

Official/schema findings:

- JSON Schema distinguishes variable-length lists from fixed/singular
  structures and supports explicit `minItems`/`maxItems` constraints:
  https://json-schema.org/understanding-json-schema/reference/array
- Pydantic fields produce and validate JSON Schema constraints, but a direct
  object is simpler than a one-element array when exactly one result is
  requested:
  https://docs.pydantic.dev/latest/concepts/fields/
- Gemma uses model-specific prompt/control formatting; structured transport
  support does not eliminate the need for an unambiguous task schema:
  https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4

Decision:

- For configured batch size one, use the existing `JudgedCandidate` or
  `CrossJudgedCandidate` object directly as the response schema—no enclosing
  judgments array. The prompt supplies one review object and requests one
  preserved record ID.
- Wrap the singular validated object internally and reuse the existing
  fail-closed parse/identity logic. Keep the batch schemas and parsers available
  for future explicitly enabled batches and existing integrity regressions.
- Reject unsupported judge batch sizes at orchestration until a profile has a
  researched context budget and the batch response path is deliberately
  enabled. Do not silently switch contracts based on model name.
- Preserve independent judging, exact witness grounding, audit rows, and
  terminal coverage behavior.

Risks and validation:

- A provider can still return malformed JSON or the wrong ID; existing retry,
  response validation, and identity quarantine remain necessary.
- Singular calls increase request count compared with true batching, but the
  current deployment is already configured at one record per request.
- Unit-test direct schema shape, singular parser wrapping, wrong-ID
  quarantine, and fail-fast rejection of batch sizes other than one. Validate
  yield only in a user-run pilot.

Implementation result:

- [x] Added direct-object single-document and cross-document judge classes
  using the existing `JudgedCandidate` and `CrossJudgedCandidate` schemas.
- [x] Singular prompts now contain one review object and explicitly request one
  preserved record ID without a `judgments` array.
- [x] Singular responses are wrapped internally and reuse the existing exact
  ID/cardinality parser, witness grounding, rubric, and quarantine behavior.
- [x] Orchestration uses singular judges and fails closed unless
  `quality.judge_batch_size` is exactly one.
- [x] Kept batch judge classes and duplicate/missing/unexpected-ID regressions
  for future deliberately enabled batch support.
- [x] Passed 52 focused procurement tests, Ruff, and compilation, including
  direct-schema and wrong-ID singular-response regressions.
- [ ] Confirm duplicate-cardinality quarantine drops and judge yield improves
  in the next bounded user-run pilot.

Research basis:

- [Curator repository and quickstart](https://github.com/bespokelabsai/curator):
  `parse()` converts one input and structured response into a list of output
  dictionaries, so application-specific row correspondence is deliberately
  controlled by the pipeline.
- [Curator API reference](https://docs.bespokelabs.ai/bespoke-curator/api-reference):
  `response_format` enforces structured output while `parse()` defines the
  resulting dataset rows.
- [JSON Schema array reference](https://json-schema.org/understanding-json-schema/reference/array):
  `minItems`, `maxItems`, and `uniqueItems` constrain array instances, but
  `uniqueItems` applies to whole items rather than a nested identity field.
- [Pydantic validation documentation](https://docs.pydantic.dev/2.12/errors/usage_errors/):
  field validators validate declared values; collection business invariants
  require explicit validation logic and raise validation errors on failure.

Pilot evidence:

- `outputs/pilot-008/files/manifest.json`
- `outputs/pilot-008/files/cross_document_qa_sft.jsonl`
- `.curator_working/pilot-008/cross_judge/d2fbb3ff6891fb63/responses_0.jsonl`

## QA answer completeness and atomic attribution research (2026-07-29)

Status: researched and implemented on 2026-07-29.

Problem:

- Pilot review found an answer that was an extracted sentence fragment rather
  than a complete response to its question.
- Single-document QA currently attaches a flat evidence list to the whole
  answer. This proves that quotations occur in the source, but it does not
  expose which quotation supports each material answer claim.
- Cross-document QA already requests claim/evidence groups, but deterministic
  validation checks only quote location and aggregate answer support. It does
  not validate each claim against its own evidence, allowing an individually
  unsupported claim to hide behind evidence attached to another claim.

Official and primary-source findings:

- Curator's official structured-output documentation makes the Pydantic
  response model the generation contract and leaves application row validation
  to `parse()`. Claim/evidence bindings therefore belong in the response schema
  and deterministic parser rather than being reconstructed from citations
  after generation:
  https://docs.bespokelabs.ai/bespoke-curator/getting-started/structured-output
- Curator's official API reference likewise separates `response_format`
  enforcement from application-specific `parse()` behavior:
  https://docs.bespokelabs.ai/bespoke-curator/api-reference
- *Atomic Fact Decomposition Helps Attributed Question Answering* decomposes
  long answers into atomic facts and verifies evidence at that granularity,
  avoiding attribution of a whole answer to merely related evidence:
  https://arxiv.org/abs/2410.16708
- *Can LLMs Evaluate Complex Attribution in QA?* evaluates fine-grained
  attribution categories rather than treating citation presence as sufficient:
  https://aclanthology.org/2025.acl-long.837/
- Research on long-form QA evaluation reports that correctness alone is not
  enough and explicitly treats answer completeness as a quality dimension:
  https://aclanthology.org/2023.acl-long.181/

Design decision:

- Add explicit material claims with exact claim-level evidence to ordinary QA,
  matching the already established cross-document contract.
- Validate every claim independently for exact evidence location, unsupported
  quantities, legal modality, and unsupported absence assertions.
- Require answerable records to contain at least one claim and require every
  flat evidence item to be used by a claim. Derive exported evidence and
  citations from validated claim bindings so citation lineage cannot drift.
- Retain a concise whole-answer judge for relevance and completeness. Atomic
  validation complements rather than replaces holistic review.
- Add only high-confidence deterministic fragment checks: empty/whitespace
  answers, dangling function words, terminal comma/colon/semicolon, unmatched
  brackets, or an ellipsis indicating truncation. Do not reject concise noun
  phrases that correctly answer direct-fact questions.
- Preserve existing output fields and append claim attribution; do not remove
  messages, evidence, reasoning, provenance, or citations.

Validation plan:

- Unit-test valid concise answers, incomplete fragments, claim-specific
  unsupported modality/number/absence, unused evidence, and cross-claim
  evidence leakage.
- Run focused procurement tests, Ruff, and compilation locally without calling
  any configured model endpoint.
- Confirm yield and answer quality in the next user-run bounded pilot.

Implementation result:

- [x] Added ordinary-QA material claims with exact claim-level evidence while
  retaining the existing top-level evidence field.
- [x] Derived persisted claim evidence, flat evidence, and citations from the
  same validated quote bindings, including stable claim IDs and source
  locations.
- [x] Required answerable QA records to contain material claims and rejected
  unused or missing flat evidence through bidirectional claim/evidence matching.
- [x] Added per-claim exact-location, numeric, modality, and unsupported-absence
  validation for both ordinary and cross-document QA.
- [x] Added conservative truncation checks for dangling function/auxiliary
  words, terminal fragment punctuation or ellipses, empty answers, and
  unbalanced brackets while preserving concise direct answers.
- [x] Included atomic claims in the independent judge payload so holistic
  completeness review sees the same attribution structure.
- [x] Passed 55 focused procurement tests, Ruff, and compilation without
  invoking a configured model endpoint.
- [ ] Confirm accepted yield, fragment rejection, claim/citation completeness,
  and judge behavior in the next bounded user-run pilot.

## Optional proposition arguments and structured-output retries (2026-07-29)

Status: researched and implemented on 2026-07-29.

Problem:

- Pilot 010 showed Nemotron returning otherwise grounded propositions for
  copular and intransitive clauses with `object: ""`.
- `PropositionDraft.object` requires at least one character, so Instructor
  retried the same semantically valid response four times and ultimately
  discarded the request.
- The repeated failures increase latency and reduce source coverage; they are
  not Curator throttling or model-endpoint failures.

Official and primary-source findings:

- vLLM's official structured-output guidance says the prompt should align with
  the enforced schema and optional values must be represented explicitly in
  the schema. A schema must not require content that the underlying semantic
  structure does not always contain:
  https://docs.vllm.ai/en/stable/features/tool_calling/
- Pydantic's official field documentation defines `min_length` as a string
  length constraint. `Field(min_length=1)` therefore rejects the pipeline's
  existing empty-string sentinel before application-level validation can
  assess the proposition:
  https://docs.pydantic.dev/latest/concepts/fields/
- Universal Dependencies treats the nonverbal predicate, rather than an
  invented object, as the head of a copular clause and explicitly distinguishes
  intransitive predication. Forcing every proposition into a
  subject-action-object triple is therefore linguistically invalid:
  https://universaldependencies.org/en/dep/cop.html
- Instructor's documented retry mechanism feeds validation errors back to the
  model. It is useful for genuinely malformed results, but schema/semantics
  mismatches cause repeated generation rather than repair:
  https://python.useinstructor.com/concepts/retrying/

Design decision:

- Keep `subject` and `action` mandatory, because together they must express a
  complete proposition.
- Permit `object` to be the existing empty-string sentinel only when the source
  clause has no separate grammatical object; never ask the model to invent one.
- Preserve `object` as a string in materialized records for Arrow, cache, and
  reasoning-path compatibility instead of changing stored records to nullable
  values.
- Clarify the extraction prompt and retain all exact-substring, evidence,
  modality, polarity, condition, exception, and threshold validation.
- Bump proposition schema/validator versions so incompatible cached
  extractions are not silently reused.

Validation plan:

- Unit-test schema acceptance and materialization of a grounded proposition
  with an empty object.
- Assert that subject and action remain non-empty and that fabricated
  non-verbatim objects still fail deterministic validation.
- Run focused tests, Ruff, and compilation without invoking model endpoints.
- Confirm the retry failure disappears in the next user-run pilot.

Implementation result:

- [x] Allowed the established empty-string sentinel for a proposition with no
  separate grammatical object while retaining non-empty subject and action.
- [x] Clarified the extraction contract so models use an empty object only for
  genuinely intransitive or copular predicates and never invent one.
- [x] Preserved the stored string shape used by Arrow, caches, rendering, and
  reasoning-path construction.
- [x] Bumped proposition schema and validator versions to invalidate
  incompatible cache entries.
- [x] Added focused coverage for schema parsing, deterministic validation, and
  materialization of a grounded clause without a separate object.
- [ ] Confirm the retry failure and accepted proposition yield in the next
  user-run pilot.

## Curator structured failure persistence (2026-07-29)

Status: researched and implemented on 2026-07-29.

Problem:

- Pilot 010 completed 29 of 30 proposition requests, but one request exhausted
  structured-output retries.
- Curator correctly constructed a failed `GenericResponse` with
  `response_message=None` and populated `response_errors`, then passed that
  missing payload into the configured Pydantic response model.
- `PropositionBatch(**None)` raised `TypeError`, aborting finalization and
  preventing the 29 successful responses from becoming a dataset.

Official-source findings:

- Curator's own `GenericResponse` contract documents `response_message=None`
  as the representation used when errors occur:
  https://github.com/bespokelabsai/curator/blob/main/src/bespokelabs/curator/types/generic_response.py
- Curator's online processor creates exactly that failed response after retry
  exhaustion and sends it to the common persistence path:
  https://github.com/bespokelabsai/curator/blob/main/src/bespokelabs/curator/request_processor/online/base_online_request_processor.py
- Curator's dataset finalizer already treats non-null `response_errors` as a
  failed request and continues processing successful records:
  https://github.com/bespokelabsai/curator/blob/main/src/bespokelabs/curator/request_processor/base_request_processor.py
- Curator's formatter assumes a structured response is a mapping and invokes
  the Pydantic model with `**response_dict`; it is therefore not the correct
  layer for parsing a failed response with no payload:
  https://github.com/bespokelabsai/curator/blob/main/src/bespokelabs/curator/llm/prompt_formatter.py
- Curator PR 734 is retained in this repository. Its work-conserving retry
  scheduling correctly persists terminal failures, but the discovered
  `None`-payload parsing edge case must be handled without removing or
  reverting that scheduler:
  https://github.com/bespokelabsai/curator/pull/734

Design decision:

- In the common response-processing boundary, return no parsed rows immediately
  when `response_errors` is populated or `response_message` is `None`.
- Still append the full `GenericResponse` to the responses JSONL so errors,
  request identity, timing, and audit information survive.
- Do not invoke the application `parse()` callback for failed provider/model
  responses.
- Preserve existing successful-response behavior, including successful
  responses intentionally filtered to zero rows.
- Add regression coverage using a real Pydantic response format, not an
  identity mock.

Validation plan:

- Verify a terminal structured-output failure is persisted with its errors and
  a null parsed payload without raising.
- Verify the application parser is not called for the failed response.
- Retain existing retry, successful persistence, and all-filtered dataset tests.
- Run focused unit tests, Ruff, and compilation without model calls.

Implementation result:

- [x] Short-circuited common response parsing for terminal responses carrying
  `response_errors` or a null `response_message`.
- [x] Preserved the full failed `GenericResponse` in the responses JSONL with
  request identity and error details.
- [x] Prevented both Pydantic structured parsing and application `parse()` from
  receiving a failed response.
- [x] Left PR 734's work-conserving retry scheduling unchanged.
- [x] Added a regression test using a real Pydantic response format and parser.
- [x] Confirmed partial-success finalization in user-run pilot 011: failed
  requests were persisted and successful siblings reached the stage outputs.

## Nemotron path-answer structured-output failures (2026-07-29)

Status: researched and implemented on 2026-07-29.

Problem:

- Pilot 011's path-question stage completed 61/61 requests, but the path-answer
  stage repeatedly generated JSON-like text with unquoted `answer` or
  `statement` values.
- Instructor's prompt-only Markdown JSON recovery sometimes selected a nested
  `claims` or `evidence` list and attempted to validate that list as the root
  `PathAnswerDraft`, producing `Input should be an object`.
- Curator's terminal-failure persistence fix worked: 41 successful path answers
  finalized and the one permanent failure was audited rather than crashing the
  pipeline.

Official-source findings:

- NVIDIA's official Nemotron 3 Super model card uses
  `temperature=1.0`, `top_p=0.95` and serves vLLM with
  `--enable-auto-tool-choice --tool-call-parser qwen3_coder`. The model is
  trained for tool use and structured-output environments:
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
- The official NVIDIA generation configuration confirms
  `temperature=1.0` and `top_p=0.95`; lowering temperature is not the documented
  remedy:
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/blob/main/generation_config.json
- vLLM's official structured-output documentation says JSON Schema decoding
  constrains output to the schema and recommends also describing the expected
  schema in the prompt:
  https://docs.vllm.ai/en/stable/features/structured_outputs/
- vLLM's official tool-calling documentation says named or required function
  calls use schema-constrained decoding and recommends nullable schema types
  for genuinely optional fields:
  https://docs.vllm.ai/en/stable/features/tool_calling/
- Instructor distinguishes prompt-parsed Markdown JSON from tool/function
  modes. Prompt-only recovery cannot provide the structural guarantee of a
  server-constrained tool call:
  https://python.useinstructor.com/modes-comparison/
- LiteLLM's official vLLM provider documentation treats the endpoint as an
  OpenAI-compatible transport; actual schema/tool support still depends on the
  private server's launch configuration:
  https://docs.litellm.ai/docs/providers/vllm

Local evidence:

- The committed Nemotron profile uses `structured_output_mode: md_json`, so
  Instructor asks for JSON in text and parses the result after generation.
- The saved Curator requests contain the complete `PathAnswerDraft` JSON Schema,
  but prompt-only mode does not constrain token generation at the vLLM server.
- The malformed completions contain substantively useful grounded answers; the
  failure is serialization, not model reachability, context length, rate
  limiting, or missing evidence.
- Multiple inner Instructor repair attempts repeated the same quoting mistake,
  inflating token use and latency. Accepting a nested list as the root would
  silently lose the answer and other claims and is therefore unsafe.

Design decision:

- Keep the documented Nemotron sampling values unchanged.
- Change only the Nemotron endpoint profile from prompt-only `md_json` to
  schema-constrained `tools`, matching NVIDIA's documented vLLM serving path.
- Retain `json_schema` for GLM and Gemma; structured-output transport remains a
  per-profile configuration rather than a model-name conditional.
- Do not add heuristic malformed-JSON repair or coerce nested lists into root
  response objects.
- Preserve Curator's partial-success behavior so an isolated structured-output
  failure is audited without aborting the run.

Validation plan:

- Assert the Nemotron profile selects tools while other profiles retain their
  configured modes.
- Run configuration/unit tests, Ruff, and compilation without contacting model
  endpoints.
- Confirm the private Nemotron deployment accepts forced tool calls in the next
  user-run pilot. If it rejects them, its serving flags must be aligned with
  NVIDIA's official launch command before production generation.

Implementation result:

- [x] Initially changed only the configurable Nemotron profile from
  prompt-parsed `md_json` to schema-constrained `tools`.
- [x] Kept the official Nemotron sampling values and context configuration.
- [x] Left GLM and Gemma on their independently configured native
  `json_schema` modes.
- [x] Added configuration regression assertions preventing an accidental
  return to prompt-only Nemotron JSON.
- [x] Retained strict Pydantic and deterministic validation without malformed
  JSON coercion.
- [ ] Replace forced named tools with validated auto-tool transport based on
  the live endpoint verification below.

Live endpoint verification and revised decision (2026-07-29):

- The private endpoint advertised model ID `nvidia/nemotron-3-super`, vLLM
  ownership, and `max_model_len: 131072`, matching committed configuration.
- A minimal request containing no corpus data returned HTTP 500 for both a
  named `tool_choice` and `tool_choice: "required"`.
- The identical schema with `tool_choice: "auto"` returned HTTP 200 and one
  valid `record_probe` call with arguments
  `{"value": "alpha", "count": 7}`.
- Inspection of installed Instructor 1.15.4 shows `Mode.TOOLS` unconditionally
  replaces `tool_choice` with a named function. Passing `auto` through
  generation parameters cannot override it.
- Therefore, the private server's tool parser is active, but its forced-tool
  path is incompatible with Instructor's standard tools mode. Leaving
  `structured_output_mode: tools` would make the next run fail.

Revised design:

- Add an explicit, reusable `tools_auto` Curator/LiteLLM mode rather than a
  Nemotron model-name branch.
- For structured requests in this mode, send exactly one Pydantic-derived tool,
  set `tool_choice: "auto"`, and add a transport-level instruction requiring
  exactly one call to that tool.
- Require exactly one returned tool call with the expected name and validate
  its JSON arguments through the configured Pydantic model.
- Treat missing, multiple, wrongly named, malformed, or schema-invalid tool
  calls as normal request failures handled by Curator's existing retry and
  partial-success machinery.
- Select `tools_auto` only in the Nemotron profile. Preserve native
  `json_schema` for endpoints that support it.

Implementation result:

- [x] Added `tools_auto` as an explicit online backend configuration value.
- [x] Built a transport-level single-tool request with `tool_choice: "auto"`
  and a system instruction requiring exactly one call.
- [x] Required exactly one expected tool name and Pydantic-validated its JSON
  arguments before Curator parsing.
- [x] Kept malformed, missing, multiple, or wrongly named calls in Curator's
  normal retry/failure path.
- [x] Selected `tools_auto` only for the verified Nemotron profile.
- [x] Documented the new model-portable mode and added focused regression
  coverage.
- [ ] Confirm path-answer yield in the next user-run pilot.

## Pilot 011 completed-run audit (2026-07-29)

Status: inspected after the user-run pilot completed.

Outcome:

- The run exported 60 canonical records: 46 `qa`, 7 `qa_cot`, 5
  `cross_document_qa`, and 2 `cross_document_qa_cot`.
- All 60 canonical records have unique IDs, non-empty claims, evidence, and
  citations; keep `citations` as the last field; pass deterministic validation;
  and have an accepted 5/5 judge result.
- CoT records contain reasoning steps and plain QA records do not. All seven
  accepted cross-document records cite at least two manuals and manually
  inspected questions compare substantively aligned provisions.
- The run correctly persisted partial stage success. Eight requests failed:
  three single-document generations, one path answer, three ablation trials,
  and one cross-document judge request.
- The Nemotron failures are consistent with the old prompt-parsed `md_json`
  transport used by this already-running pilot: malformed root JSON, nested
  list recovery, or a missing `examples` root field. Pilot 011 therefore does
  not test the subsequently committed `tools_auto` transport.
- The cross-judge failure was independent: the Gemma endpoint reported an
  8,192-token context limit when prompt plus reserved output required at least
  8,193 tokens.
- The terminal manifest status is `failed` because both planned drafting
  records failed deterministic validation with
  `drafting_tender_fact_aggregate_mismatch`; the pipeline deliberately exits
  when drafting is enabled but no drafting record is accepted.
- Path QA produced 42 accepted questions, 16 accepted answers, and 45 valid
  ablation trials, but remains explicitly pending source-ablation judgment and
  contributed no training records.
- Leakage-safe component splitting yielded 57 train, 0 validation, and 3 test
  records. This avoids connected-manual leakage but is too imbalanced to serve
  as a production evaluation split at this pilot size.

Follow-up:

- [ ] Confirm Nemotron `tools_auto` path-answer and ablation yield in the next
  user-run pilot.
- [x] Reconciled drafting block-level evidence and tender facts by deriving
  stable top-level aggregates from the validated block-local attribution.
- [x] Bounded judge requests against the endpoint's actual 8,192-token context
  limit, including reserved output tokens, a safety margin, and an auditable
  pre-request quarantine.
- [x] Replaced independent split hashing with deterministic whole-component
  target optimization so small pilots retain leakage isolation without an
  avoidable empty validation/test split.

### Pilot 011 remediation research (2026-07-29)

Status: official-source research completed before implementation.

Questions:

- How should the pipeline prevent judge requests from exceeding a served
  model's real context limit?
- How can small pilots retain leakage-safe connected groups while avoiding an
  accidentally empty validation split?
- Should a redundant drafting aggregate be trusted over block-local
  attribution when the two disagree?

Official-source findings:

- vLLM defines `max_model_len` as the combined prompt and generated-output
  context length. Its renderer validates the prompt against the model limit
  after reserving requested output tokens:
  https://docs.vllm.ai/en/stable/api/vllm/config/
  https://docs.vllm.ai/en/stable/api/vllm/renderers/
- scikit-learn's official group-splitting documentation confirms that split
  proportions operate on whole groups rather than individual samples. Group
  isolation therefore has priority over exact record-level proportions:
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
- PyPA specifies `[project.scripts]` as the standard mapping from an installed
  command name to a callable object, relevant to the still-open unified CLI:
  https://packaging.python.org/en/latest/specifications/pyproject-toml/

Local evidence:

- Pilot 011's rejected cross-judge request contained at least 6,145 input
  tokens while reserving 2,048 output tokens against an 8,192-token endpoint.
- Pilot 011 has eight populated leakage components containing
  26, 16, 6, 3, 3, 3, 2, and 1 accepted records. The current independent hash
  assignment happened to place 57 records in train, none in validation, and
  three in test even though a materially better whole-component allocation
  exists.
- Drafting block attribution already validates every block-local fact against
  the tender seed. The top-level `tender_facts_used` field is a redundant
  model-produced aggregate; both Pilot 011 drafts failed solely because it did
  not exactly reproduce the stable block union.

Design decision:

- Apply rendered-request budgeting to judge inputs using the selected judge
  profile's context window and reserved completion budget. Quarantine
  over-budget rows with an auditable reason before an API request.
- Reduce the Gemma judge completion reservation to 1,024 tokens, which is
  ample for the bounded decision schemas and leaves capacity for source
  context. Keep a safety margin.
- Derive top-level drafting evidence and tender-fact aggregates as stable
  first-use unions of block-local attribution before deterministic validation
  and export. Continue rejecting unknown or unsupported block-local support.
- Replace independent component hashing with deterministic whole-component
  allocation that minimizes deviation from configured record targets and,
  when enough populated components exist, assigns at least one component to
  every positive split.
- Do not claim that Pilot 011 validates these changes; validation requiring
  model calls remains a user-run Pilot 012 gate.

Implementation result:

- [x] Reduced the Gemma judge completion reservation from 2,048 to 1,024
  tokens.
- [x] Added complete rendered judge-request measurement and separate
  `qa_judge_prompt_rejected.jsonl` /
  `cross_judge_prompt_rejected.jsonl` audit artifacts.
- [x] Kept over-budget records out of API calls and retained them in the
  ordinary rejection exports with `judge_prompt_exceeds_context_window`.
- [x] Derived drafting aggregates from block-local support without weakening
  unknown-fact, verbatim-evidence, numeric, authority, or citation gates.
- [x] Assigned connected components by deterministic normalized target error
  rather than independent random buckets.
- [x] Added focused regression tests; 59 procurement-pipeline tests pass and
  Ruff reports no errors for the changed modules.
