# NRL procurement synthetic data

This example reads procurement manuals from
`/home/abhishek/curator/data/source` and generates grounded question-answer
examples through an OpenAI-compatible local endpoint. Existing Markdown
documents are consumed directly; PDFs are converted with Chandra OCR 2 first.

## Model configuration

Runtime settings live in the repository-root `.env`:

```dotenv
GENERATION_MODEL=nvidia/nemotron-3-super
GENERATION_BASE_URL=http://10.180.148.183:3011/v1
GENERATION_API_KEY=replace-me
```

The Python code contains no fixed model name or endpoint. To use another model,
change these three values only. `.env` is gitignored; `.env.example` is the safe
template that can be committed.

## PDF preprocessing with Chandra OCR 2

Install Chandra's vLLM client:

```bash
pip install chandra-ocr
```

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
python examples/nrl-procurement/preprocess_pdfs.py
```

## Run

Start with a small pilot:

```bash
python examples/nrl-procurement/generate.py --limit 5
```

Remove `--limit` to process all Markdown pages:

```bash
python examples/nrl-procurement/generate.py
```

Curator request and response caches are written under `data/synthetic`. The
final Hugging Face dataset is saved under
`data/synthetic/procurement_qa`.

The generator consumes both the 16 page-preserving Markdown documents under
`data/source/procurement_manuals` and OCR Markdown under
`data/interim/ocr`. Native page markers are retained where available;
otherwise Chandra Markdown is divided into bounded, numbered source chunks.
