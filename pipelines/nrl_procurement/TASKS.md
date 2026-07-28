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

### 1. Extract grounded atomic propositions

- [ ] Add a proposition schema containing:
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
- [ ] Extract propositions independently for each source window.
- [ ] Reject propositions whose complete factual content is not supported by
  exact evidence.
- [ ] Cache propositions by source hash, chunk ID, model profile, and schema
  version.

Acceptance criteria:

- Every proposition resolves to one registered source and exact offsets.
- Numbers, dates, names, and modalities match the evidence.
- Government propositions cannot be labelled as NRL policy.
- Re-running unchanged sources reuses the cached proposition set.

### 2. Construct connected reasoning paths before questions

- [ ] Build explicit two-hop path types:
  - comparison;
  - bridge;
  - temporal transition;
  - complementary procedure;
  - exception/condition interaction;
  - cross-domain comparison.
- [ ] Require compatible subjects or an explicit bridge entity.
- [ ] Store the path independently of its natural-language question.
- [ ] Reject paths that merely contain two unrelated facts.
- [ ] Do not treat similarity as adoption, equivalence, precedence,
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

### 3. Use bounded multi-chunk source windows

- [ ] Group adjacent chunks under reliable section boundaries.
- [ ] Include required definitions, exceptions, and referenced provisions when
  resolvable.
- [ ] Split oversized sections on chunk boundaries.
- [ ] Preserve every constituent chunk ID and page.
- [ ] Prevent repeated or generic headings from establishing a match by
  themselves.
- [ ] Support one-to-many section relationships.

Acceptance criteria:

- A source window never crosses manual, issuer, edition, or policy scope.
- Prompt size is checked before generation.
- Exact evidence still resolves to its original chunk, page, and offsets.

### 4. Generate questions from verified paths

- [ ] Generate natural-language questions only after a path passes
  deterministic checks.
- [ ] Control the intended question type and difficulty.
- [ ] Require standalone authority, domain, and date wording when necessary.
- [ ] Generate the final answer in a separate pass from question construction.
- [ ] Add missing-hop and false-premise unanswerable contrasts.

Acceptance criteria:

- The question cannot be fully answered without executing the declared path.
- The answer is produced from the supplied sources, not copied from a proposed
  answer embedded in the question-generation prompt.
- Unanswerable records state only supported limitations.

### 5. Execute real source-ablation trials

- [ ] Answer each candidate with both sources.
- [ ] Answer it again with only source A.
- [ ] Answer it again with only source B.
- [ ] Measure required-claim coverage for all three outputs.
- [ ] Store the actual ablation outputs and decisions.
- [ ] Reject answerable records when either single-source output fully covers
  the canonical answer.

Acceptance criteria:

- Full context covers all required answer claims.
- A-only misses at least one required source-B claim.
- B-only misses at least one required source-A claim.
- The judge reviews actual ablated outputs rather than predicting the result.

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
- [ ] Separately investigate the GLM server's long generation latency and the
  two cross-document responses that were not materialized; do not mislabel
  these as Curator throttling.

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
