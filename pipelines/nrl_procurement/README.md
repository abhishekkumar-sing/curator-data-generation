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

GENERATION_MODEL=nvidia/nemotron-3-super
GENERATION_BASE_URL=http://10.180.148.183:3011/v1
GENERATION_API_KEY=replace-me

JUDGE_MODEL=nvidia/nemotron-3-super
JUDGE_BASE_URL=http://10.180.148.183:3011/v1
JUDGE_API_KEY=replace-me
```

The Python code contains no fixed model name or endpoint. Generation, judging,
and OCR can each be changed through `.env` without editing Python. Use a
different judge model for production when one is available; using the
generation model for both roles is supported for pilots. `.env` is gitignored;
`.env.example` is the safe template that can be committed.

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
.curator/bin/python pipelines/nrl_procurement/generate.py --limit 5
```

Remove `--limit` to process all Markdown pages:

```bash
.curator/bin/python pipelines/nrl_procurement/generate.py
```

Curator request and response caches are written under
`data/synthetic/.cache`. Accepted records are written to:

- `canonical.jsonl`: lossless records, provenance, checks, and judge output
- `qa_sft.jsonl`: concise QA chat training data
- `qa_cot_sft.jsonl`: QA with short evidence-based teaching rationales
- `rag.jsonl`: questions, contexts, and answerability labels
- `eval.jsonl`: reference answers and evidence for evaluation
- `manifest.json`: source hashes, metadata, and output statistics

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

For a local development run without judging, first set
`quality.allow_unjudged_exports: true` in `config.yaml`, then pass
`--skip-judge`. This is intentionally disabled by default.

## Tests

After installing the project environment:

```bash
.curator/bin/python -m pytest -q --confcutdir=tests/nrl_procurement tests/nrl_procurement
```
