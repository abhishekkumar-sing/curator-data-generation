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
JUDGE_PROFILE=nemotron

GLM_MODEL=replace-me
GLM_BASE_URL=http://127.0.0.1:8000/v1
GLM_API_KEY=replace-me

NEMOTRON_MODEL=replace-me
NEMOTRON_BASE_URL=http://127.0.0.1:8001/v1
NEMOTRON_API_KEY=replace-me
```

The Python code contains no fixed model name or endpoint. Named profiles for
GLM, Nemotron, Gemma, and Qwen are declared in `config.yaml`; credentials and
served-model IDs stay in `.env`. Switch either role by changing only
`GENERATION_PROFILE` or `JUDGE_PROFILE`. Use different generator and judge
models for production when possible. `.env` is gitignored; `.env.example` is
the safe template that can be committed.

Each profile declares one of Curator's structured-output transports:
`auto`, `tools`, `json_schema`, `json`, or `md_json`. The choice belongs to the
specific model-and-server deployment, not just the model family. Run a small
structure probe after changing an endpoint. Native `json_schema` is preferred
when verified; `md_json` provides prompt-based JSON plus Pydantic validation
when the server's native modes are broken or unavailable.

`config.yaml` contains committed, non-secret defaults for paths, model
parameters, model environment-variable names, and privacy behavior. `.env`
overrides its Curator switches and supplies endpoint-specific values and
credentials.

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
