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
