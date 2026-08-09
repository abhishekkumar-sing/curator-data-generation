# NRL procurement synthetic data

This pipeline reads procurement manuals from
`/home/abhishek/curator/data/source` and generates grounded question-answer
examples through an OpenAI-compatible local endpoint. Existing Markdown
documents are consumed directly; PDFs are converted with Chandra OCR 2 first.

## Model configuration

Runtime settings live in the repository-root `.env`:

```dotenv
CURATOR_LOCAL_ONLY=1
CURATOR_VIEWER=0
TELEMETRY_ENABLED=false

GENERATION_PROFILE=glm
JUDGE_PROFILE=gemma_structured

MODEL=gemma-4-31b-it
LLM_DEPLOYMENT_ID=gemma-4-31b-it-10.180.148.183-8010-v1
LLM_BASE_URL=http://127.0.0.1:3005/v1
LLM_API_KEY=replace-me

GLM_MODEL=GLM-5.2-NVFP4-FP8
GLM_DEPLOYMENT_ID=glm-5.2-nvfp4-fp8-v1
GLM_BASE_URL=http://127.0.0.1:3005/v1
GLM_API_KEY=replace-me

NEMOTRON_MODEL=replace-me
NEMOTRON_BASE_URL=http://127.0.0.1:8001/v1
NEMOTRON_API_KEY=replace-me

MINISTRAL_MODEL=ministral-3-14b-instruct-2512
MINISTRAL_DEPLOYMENT_ID=ministral-3-14b-instruct-2512-deployment-v1
MINISTRAL_BASE_URL=http://127.0.0.1:3006/v1
MINISTRAL_API_KEY=replace-me
```

The Python code contains no fixed model name or endpoint. Named profiles for
GLM, Nemotron, thinking-enabled generation Gemma, Gemma, Ministral, and Qwen
are declared in `config.yaml`; credentials and
served-model IDs stay in `.env`. Switch either role by changing only
`GENERATION_PROFILE` or `JUDGE_PROFILE`. Use different generator and judge
models for production when possible. `.env` is gitignored; `.env.example` is
the safe template that can be committed.

Both shared-gateway Gemma profiles send `temperature=1.0`, `top_p=0.95`, and
`top_k=64`. The selected `gemma_structured` profile sends
`extra_body.chat_template_kwargs.enable_thinking=false`; these stages need a
compact schema-constrained decision, not a provider reasoning trace. The
`gemma_thinking` profile remains available for an explicit reasoning experiment
and sends `enable_thinking=true`. Model profiles own this chat-template choice
so a role-level default cannot silently override it.
Because both RPM and TPM are configured explicitly, Curator does not spend an
extra inference request attempting to rediscover those limits from headers.

The configured production roles use GLM for generation and the independent
non-thinking structured `gemma-4-31b-it` route for judging. Both model aliases may be
served through the same LiteLLM gateway, while their credentials and deployment
identities remain independent. `LLM_DEPLOYMENT_ID` identifies the underlying
Gemma deployment, so it does not need to match the gateway port in
`LLM_BASE_URL`. The judge uses
native JSON-schema mode, the role-level 2,048-token ordinary ceiling with a
4,096-token missing-row rescue, and the profile's 45-request concurrency
cap. Ministral remains available as an explicit fallback profile. Every
endpoint change still requires the exact structured-output probe.

The former direct port-8010 qualification failed at TCP connection setup on
2026-08-07. The replacement shared-gateway profile was independently qualified
on 2026-08-08: both GLM generation and thinking-enabled Gemma judging passed
every structured-output probe check. Selecting `gemma_structured` changes the
request contract and therefore requires a fresh judge probe. Endpoint/model/key
changes also produce a new fingerprint and require another live qualification.

The GLM profile uses a 1,200-second request timeout and one retry. The first
load test measured about 256 seconds median, 339 seconds p99, and 365 seconds
maximum for ordinary blueprint requests, but fresh cross-document pass 5 still
had productive long-running requests reach the former 600-second boundary at
concurrency 32. A silent request is therefore retried after twenty minutes
instead of holding the final progress slot for the old 1,800-second timeout.
Timeout, retry, concurrency, RPM, and TPM values are transport tuning: changing
them does not alter scientific checkpoint compatibility, although every run
manifest still records their exact values.

GLM generation uses 45 concurrent requests for the shared-gateway smoke after
128 concurrent long-output requests previously produced a synchronized
server/queue timeout wave. During saturation replay, the pipeline reuses the
integrity-checked historical
checkpoint from the earlier attempt and verifies that reconstructed outcomes
exactly match persisted saturation state; it does not regenerate an old pass or
weaken the mismatch guard. For the comprehensive shared-gateway smoke, ordinary
GLM generation, GLM rescue, Gemma judging, and judge rescue are all set to 45
concurrent requests. These transport limits must be reevaluated from recorded
latency and timeout yield before any further increase.

Terminal structured outputs omitted after ordinary retries receive one
missing-row-only recovery stage. Generation rescues use at most 12,000 tokens;
judge rescues use at most 4,096. Each rescue is separately checkpointed and is
submitted only when the complete rendered prompt plus the larger completion
reserve fits the selected deployment context. Existing successful rows are
never regenerated, and partial/truncated judge JSON is never accepted. Curator
also returns an empty dataset when an entire stage fails and
`require_all_responses=false`, allowing the same audited rescue to run instead
of crashing before missing-row reconciliation.

The ordinary GLM completion reserve is 5,000 tokens, matching the stable native
GLM pipeline. The 12,000-token rescue remains available for rare verbose rows.
Client concurrency is a ceiling, not a throughput guarantee: a stage containing
7 or 12 requests cannot use a limit of 45 beyond those 7 or 12 requests. For
server-side throughput, measure vLLM queue time, time to first token, decode
throughput, and KV-cache pressure; tune `--max-num-seqs` and
`--max-num-batched-tokens` on the actual deployment, with chunked prefill enabled
where supported. Raising only Curator concurrency can increase queue latency and
recreate the synchronized timeout wave seen at 128.

The 2026-08-08 same-deployment A/B supports this split. On the structure probe,
thinking used 401 output tokens and 6.87 seconds; non-thinking used 73 and 1.71
seconds, with both passing every contract check. On the same three real
procurement judge inputs, non-thinking reproduced the thinking run's two score-5
acceptances and one score-3 rejection, completed all three in 4.91 seconds with
644 output tokens, and did not truncate. Thinking took 76.43 seconds plus 46.68
seconds of rescue and used at least 5,621 successful output tokens; failed
truncated attempts add further unreported consumption. This is enough to select
non-thinking operationally for compact judging, but not to claim broad quality
equivalence. The remaining gate is a 100-200-record stratified, human-labeled
A/B covering direct/CoT, single/cross-document, answerability, boundary, and
adversarial decisions.

### Diverse instruction and QA-CoT generation

Treat diversity as a coverage-and-selection problem, not a request to “word the
question differently.” Before model calls, the pipeline now plans a
source-feasible question intent, wording style, answer format, direct/CoT shape,
reasoning operation, difficulty, and material-focus axis. The grounded blueprint
then supplies the source-supported procurement task, persona, and concrete
persona need. Planned cells are written to `instruction_coverage_plan.jsonl` and
accepted completed cells to `instruction_coverage_matrix.jsonl`.

A candidate is materially new only if it changes a
supported rule, condition, exception, threshold boundary, stakeholder decision,
evidence requirement, temporal state, or reasoning path. Cross product only the
axis combinations supported by the source; do not invent a scenario merely to
fill a quota. Intermediate CoT cells generate two independent candidates and
advanced cells generate three. Every candidate is deterministically checked and
independently judged; one winner is retained by grounded quality and
qualification preservation before any diversity tie-break.

Use `qa_cot` only for genuine two-to-four-step problems. Student-facing
rationales are concise auditable steps with an operation, an evidence-based
inference, and exact supporting quotes. They are distinct from the provider's
private thinking stream: Gemma thinking can be disabled for the judge while the
dataset still contains verified teaching rationales. Direct facts retain
`reasoning_steps=[]`; adding ceremonial steps to easy questions teaches verbose
artifacts rather than reasoning.

This design follows the generate/filter loop in Self-Instruct, controlled
complexity in Evol-Instruct, joint complexity-quality-diversity selection in
DEITA, domain-conditioned task generation in Bonito, and evidence/distractor
training in RAFT. CoT candidates follow the verification lesson of STaR and the
selection lesson of self-consistency: keep rationales that reach a verified
answer, not rationales merely because they are long. Exact research links and
the remaining implementation gates are recorded in `TASKS.md`.

`MINISTRAL_MODEL` must be the deployment's advertised OpenAI model ID, not the
Hugging Face repository path. The current private endpoint advertises
`ministral-3-14b-instruct-2512` and a 65,536-token served context window.

Each profile declares one of Curator's structured-output transports:
`auto`, `tools`, `tools_auto`, `json_schema`, `json`, or `md_json`. The choice belongs to the
specific model-and-server deployment, not just the model family. Run a small
structure probe after changing an endpoint. Native `json_schema` is preferred
when verified; `md_json` provides prompt-based JSON plus Pydantic validation
when the server's native modes are broken or unavailable.

Before an unbounded generation, probe both exact role deployments through their
configured production transports:

```bash
.curator/bin/python pipelines/nrl_procurement/probe_structure.py
```

Use `--role generation` or `--role judge` to probe one role. The probe verifies
a nested enum/object/list contract with exact sentinels and writes only a
secret-free, fingerprinted result under `.curator_working/structure_probes/`.
Changing the endpoint, served model, deployment identity, transport mode,
schema dereferencing, generation parameters, probe contract, or relevant
transport-library version requires a new probe. Full runs fail closed without
matching successful generation and judge results; bounded `--limit` pilots can
still run for diagnosis. A successful transport probe does not replace the
bounded data-quality pilot or human review.

`config.yaml` contains committed, non-secret defaults for paths, model
parameters, model environment-variable names, and privacy behavior. `.env`
overrides its Curator switches and supplies endpoint-specific values and
credentials.

### Semantic-diversity selection and calibration

The checked-in embedding analysis profile is enabled. It uses NVIDIA's
OpenAI-compatible `llama-nemotron-embed-1b-v2` endpoint only for generated
question text; source passages, answers, evidence, and credentials are never
included in the embedding input. Configure a newly issued key in the untracked
`.env` and probe the exact deployment before generation:

```bash
.curator/bin/python pipelines/nrl_procurement/semantic_calibration.py probe
```

An enabled run caches question vectors under
`.curator_working/semantic_embeddings/` and writes nearest-neighbor pairs to
`outputs/<run-id>/files/semantic_calibration.jsonl`. The default
`verified_equivalence` selection mode is active without a cosine cutoff. It
removes a paraphrased question only when the independently judged record has the
same normalized answer, exact evidence, source identity, task, persona, intent,
answer format, operation, difficulty, and material focus. Cosine similarity is
retained in the rejection audit but cannot override those invariants.

For broader embedding-only clustering, switch to `calibrated_threshold` only
after reviewers set
`human_label` to `duplicate`, `related`, or `distinct`. Produce a secret-free
development calibration report with:

```bash
.curator/bin/python pipelines/nrl_procurement/semantic_calibration.py calibrate \
  --input outputs/<run-id>/files/semantic_calibration.jsonl \
  --output outputs/<run-id>/files/semantic_calibration_report.json
```

The command will not recommend a threshold without at least 50 labeled pairs,
including at least 10 duplicates and 10 non-duplicates. Its recommendation is
still in-sample: validate the proposed threshold on a separate reviewed holdout
before configuring `selection_mode: calibrated_threshold` and its recommended
`similarity_threshold`. The verified-equivalence default does not depend on
that optional broader threshold.

The comprehensive configuration also enables proposition extraction, connected
reasoning paths, temporal artifacts, path QA, and cross-document saturation.
The default saturation policy has no numeric pass cap and stops only after each
parent reaches the configured consecutive-empty convergence rule. Use an
explicit positive `--max-passes` for bounded smoke tests.

The active embedding transport was capability-probed successfully on
2026-08-08 with one 1,024-dimensional query vector. This verifies transport and
schema compatibility only; it does not authorize semantic deletion, and any
credential change requires another probe.

`CURATOR_LOCAL_ONLY=1` is a hard guard around Curator's hosted Viewer, including
explicit `push_to_viewer()` calls. Telemetry is disabled by default and also
suppressed by local-only mode. The pipeline rejects public generation and OCR
endpoints unless `private_endpoint_only` is deliberately changed in
`config.yaml`; the checked-in configuration accepts only `localhost`,
loopback, link-local, or private IP addresses.

To allow Curator's hosted Viewer later, make both changes in `.env`:

```dotenv
CURATOR_LOCAL_ONLY=0
CURATOR_VIEWER=1
```

To enable anonymized telemetry without enabling the Viewer:

```dotenv
CURATOR_LOCAL_ONLY=0
TELEMETRY_ENABLED=true
```

## PDF preprocessing with Chandra OCR 2

Install Chandra's vLLM client as an isolated tool:

```bash
uv tool install chandra-ocr==0.2.0
```

Chandra is intentionally isolated from Curator's `.curator` environment:
Chandra OCR 2 requires `python-dotenv>=1.1.1`, while Curator's pinned
`litellm==1.83.7` requires `python-dotenv==1.0.1`. The pipeline calls the
isolated `chandra` executable and passes its model endpoint through environment
variables.

Configure the OCR role in `.env`. The initial engine is Chandra, but its model
name and endpoint are ordinary configuration values:

```dotenv
OCR_MODEL=datalab-to/chandra-ocr-2
OCR_BASE_URL=http://127.0.0.1:8001/v1
OCR_API_KEY=replace-me
OCR_OUTPUT_DIR=data/interim/ocr
OCR_COMMAND=chandra
```

Then convert all PDFs:

```bash
.curator/bin/python pipelines/nrl_procurement/preprocess_pdfs.py
```

The command enables Chandra's official paginated output. Corpus loading
requires exactly one OCR result for every registered PDF and retains its page
number, source hash, revision, issuer, and policy scope.

## Run

Start with a small pilot:

```bash
.curator/bin/python pipelines/nrl_procurement/generate.py \
  --run-id pilot-001 \
  --limit 5
```

Remove `--limit` to process all Markdown pages:

```bash
.curator/bin/python pipelines/nrl_procurement/generate.py
```

When `--run-id` is omitted, the command creates a unique UTC run ID such as
`run-20260728T153012-123456Z`. Explicit run IDs may contain only letters,
digits, dots, underscores, and hyphens. A non-empty existing run is rejected
instead of overwritten.

All Curator request, response, metadata, Arrow, and recovery caches are kept
under `.curator_working/<run-id>/<stage>/`, followed by Curator's own
fingerprinted directory. The reusable, source-keyed proposition cache is kept
under `.curator_working/proposition_cache/`; its key covers source/chunk
content, model configuration, prompt, schema, and validator versions. Caches
are never mixed with exported datasets. All artifacts from one execution are
written under:

```text
outputs/<run-id>/files/
```

That `files/` directory contains:

- `canonical.jsonl`: lossless records, provenance, checks, and judge output
- `propositions.jsonl`: accepted source-isolated atomic procurement propositions
- `propositions_generated_audit.jsonl`: all proposition extractions, including
  deterministic rejections and explicit empty results
- `propositions_rejected.jsonl`: proposition records that failed exact grounding
- `source_windows.jsonl`: bounded section/adjacency windows retaining every
  constituent chunk, page, and source hash. Currently audit-only: computed and
  reported in the manifest (`source_windows.consumed_by: []`), but no
  generation stage reads these windows as an input yet -- every stage still
  builds its inputs from single-chunk rows
- `source_windows_rejected.jsonl`: source chunks that cannot fit the configured
  conservative prompt budget
- `reasoning_paths.jsonl`: accepted pre-question, two-source reasoning DAGs
- `reasoning_paths_rejected.jsonl`: path candidates rejected by deterministic
  connectivity, authority, temporal, or structural-ablation checks
- `path_questions.jsonl` / `path_questions_rejected.jsonl`: accepted and
  rejected questions planned from verified paths
- `path_answers.jsonl` / `path_answers_rejected.jsonl`: separately generated
  grounded answers pending real source ablation and independent judging
- `path_missing_hop_contrasts.jsonl`: traceable one-source-withheld negatives
- `path_false_premise_quarantine.jsonl`: non-exportable candidates awaiting a
  contradiction verifier
- `semantic_calibration.jsonl`: nearest generated-question pairs awaiting
  duplicate/related/distinct human labels when embeddings are enabled
- `semantic_rejected.jsonl`: quality-ranked semantic-cluster removals; empty
  until a separately validated threshold is explicitly enabled
- `instruction_coverage_plan.jsonl`: source-feasible intent, operation,
  difficulty, format, and material-focus assignments made before model calls
- `instruction_coverage_matrix.jsonl`: accepted cells completed with grounded
  procurement task, authentic persona need, and judge score
- `qa_generation_validation_rescue_audit.jsonl`: the single bounded corrected
  replacement attempted for each blueprint whose primary candidates all failed
  deterministic grounding or format validation; original failures remain in
  `qa_generated_audit.jsonl`
- `qa_sft.jsonl`: concise QA chat training data
- `qa_cot_sft.jsonl`: QA with short evidence-based teaching rationales
- `rag.jsonl`: questions, contexts, and answerability labels
- `eval.jsonl`: reference answers and evidence for evaluation
- `manifest.json`: run ID, source metadata, and output statistics

Only files registered in `data/source/manuals.yaml` are consumed. Records are
rejected for non-verbatim evidence, unsupported answer numbers, lost
qualifications, invalid rationale evidence, or a failing judge score.
Near-duplicate questions are removed. Train/validation/test assignment keeps an
entire manual—and manuals connected by an amendment—in one split to reduce
leakage.

`qa_cot` is used only for genuinely multi-step scenarios, conditions,
exceptions, procedures, or temporal questions. Its steps are concise,
auditable rationales tied to quoted source evidence; they are not represented
as a model's private hidden chain of thought.

## Cross-document QA

Cross-document generation is enabled by default. Approved manual relationships
are listed under `cross_document.pairs` in `config.yaml`; passages are never
combined from arbitrary documents. The configured relationships cover:

- Government and NRL manuals for the same procurement domain
- Historical and newer Government manual editions
- NRL Goods, Works, and Services cross-domain comparisons

Lexical alignment only proposes candidate passage pairs. It does not establish
adoption, equivalence, precedence, supersession, or current applicability.
Every accepted answerable record must:

- contain source-specific claims grounded by exact quotations from both manuals;
- preserve both manuals' issuer, policy scope, revision, and page provenance;
- pass a connected-reasoning check;
- be fully supported with both sources present; and
- become unsupported or materially incomplete when either required source is
  removed, as determined by the configured judge.

This last source-ablation test prevents records that merely quote two documents
but are actually answerable from one.

For a small combined pilot, `--limit` bounds both the single-document chunks and
cross-document bundles:

```bash
.curator/bin/python pipelines/nrl_procurement/generate.py --limit 5
```

To control cross-document volume independently:

```bash
.curator/bin/python pipelines/nrl_procurement/generate.py \
  --limit 20 \
  --cross-document-limit 10
```

Use `--skip-cross-document` for a single-document-only run.

### Cross-document novelty saturation

`--max-passes` controls iterative cross-document novelty generation. Omitting
the flag uses `config.yaml`; with the checked-in `saturation.enabled: false`,
that means one pass. A positive value is a hard total-pass limit. Zero removes
the numeric limit and continues until every source-bundle parent independently
produces `saturation.empty_passes_required` consecutive successful passes with
no novel record accepted after validation, near-duplicate removal, and the
independent judge:

```bash
.curator/bin/python pipelines/nrl_procurement/generate.py \
  --run-id saturation-pilot-001 \
  --limit 500 \
  --cross-document-limit 500 \
  --drafting-limit 2 \
  --max-passes 0
```

A novel accepted record resets only its own parent's empty streak. Missing or
malformed generation, invalid output, and missing judge output quarantine that
parent and leave the run `partial`; they never count as no-progress evidence.
Transient generation/validation failures are reactivated when the same run ID
is resumed. A deterministic prompt-context overflow remains quarantined until
the inputs or model configuration change. The controller stores exact parent
outcomes atomically in
`outputs/<run-id>/files/saturation/cross_document.json`; cached pass replay must
match that history. Use a new run ID after changing or refreshing a completed
generation/judge pass. Because zero has no emergency numeric ceiling, run a
positive bounded pilot first and monitor the pass audit before using it.

This convergence controller currently applies to cross-document novelty
passes. Single-document QA, drafting, and other one-shot families do not repeat
merely because `--max-passes 0` is set.

Additional cross-document exports are:

- `cross_document_qa_sft.jsonl`
- `cross_document_qa_cot_sft.jsonl`

QA-with-CoT records contain short, auditable operations such as lookup,
comparison, authority/time resolution, condition application, combination, and
conclusion. The rationale is evidence-linked teaching supervision rather than
a claim about hidden model reasoning.

The design follows the supporting-fact supervision in
[HotpotQA](https://arxiv.org/abs/1809.09600), connected question construction
in [MuSiQue](https://arxiv.org/abs/2108.00573), explicit evidence paths in
[2WikiMultiHopQA](https://aclanthology.org/2020.coling-main.580/), and
contrastive support sufficiency from
[DiRe](https://arxiv.org/abs/2005.00789).

## Seed-driven grounded drafting

Authored requests in `data/seeds/drafting_requests.jsonl` generate tender and
clause drafting examples by default. The seed path and feature switch are
configured without Python edits:

```yaml
paths:
  drafting_seeds: data/seeds/drafting_requests.jsonl

drafting:
  enabled: true
```

Each seed names exact chunk IDs from the current corpus, supplies
instance-specific tender facts, and requests one draft. Unknown or stale chunk
IDs fail immediately. The local generation model must return exact manual
evidence and declare which complete tender facts it used. Deterministic checks
reject non-verbatim evidence and unsupported numbers or email addresses before
the configured judge runs.

Accepted compact rows are written to
`outputs/<run-id>/files/drafting.jsonl` with
`id`, `tender_id`, `task`, `instruction`, `context`, `response`, and
`citations`. `drafting_generated_audit.jsonl`, `drafting_canonical.jsonl`, and
`drafting_rejected.jsonl` preserve quality and lineage details. Drafting stays
separate from the QA exports.

Run one drafting request during a pilot:

```bash
.curator/bin/python pipelines/nrl_procurement/generate.py \
  --limit 5 \
  --drafting-limit 1
```

Use `--skip-drafting` to disable it for one run, or set
`drafting.enabled: false` in `config.yaml`.

For a local development run without judging, first set
`quality.allow_unjudged_exports: true` in `config.yaml`, then pass
`--skip-judge`. This is intentionally disabled by default.

## Tests

After installing the project environment:

```bash
.curator/bin/python -m pytest -q --confcutdir=tests/nrl_procurement tests/nrl_procurement
```
## Dataset label axes

Every QA record keeps three independent labels:

- `task`: the procurement work requested, such as `drafting`, `nit_filling`,
  `evaluation_and_award`, or `contract_management`.
- `task_type`: the training-response format: `qa`, `qa_cot`,
  `cross_document_qa`, or `cross_document_qa_cot`.
- `persona`: the procurement actor represented by the user request.

The seed-provided `task` is authoritative. Drafting an NIT is `drafting`;
populating structured NIT fields in an e-procurement workflow is `nit_filling`.
Adding a rationale changes `task_type`, not `task`.
