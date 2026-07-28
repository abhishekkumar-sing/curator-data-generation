# NRL procurement synthetic data

This example reads procurement manuals from
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
```

The Python code contains no fixed model name or endpoint. To use another model,
change these three values only. `.env` is gitignored; `.env.example` is the safe
template that can be committed.

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
python pipelines/nrl_procurement/preprocess_pdfs.py
```

## Run

Start with a small pilot:

```bash
python pipelines/nrl_procurement/generate.py --limit 5
```

Remove `--limit` to process all Markdown pages:

```bash
python pipelines/nrl_procurement/generate.py
```

Curator request and response caches are written under `data/synthetic`. The
final Hugging Face dataset is saved under
`data/synthetic/procurement_qa`.

The generator consumes both the 16 page-preserving Markdown documents under
`data/source/procurement_manuals` and OCR Markdown under
`data/interim/ocr`. Native page markers are retained where available;
otherwise Chandra Markdown is divided into bounded, numbered source chunks.
