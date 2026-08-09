# vLLM Server-Capacity Measurement Runbook

Status: tooling/documentation only. No numbers in this document are
recommendations — every `max_concurrent_requests` value currently in
`config.yaml` was tuned by trial and error (128 → 45 → 32, 600s → 1200s →
1800s timeouts) against symptoms (timeout waves, retry tails) rather than
measured server capacity. This runbook is for the user, who has access to the
shared `10.180.148.183` vLLM deployment; the implementing agent that wrote
this cannot reach that host and did not invent any thresholds here.

**Audit reference:** audit §3.7 / `TASKS.md`'s own open item — "Ask the vLLM
deployment owner to capture queue/TTFT/TPOT/KV-cache metrics and validate
`--max-num-seqs`, chunked prefill, and `--max-num-batched-tokens`; client
concurrency cannot improve a stage that contains fewer requests than its
ceiling."

## Why this matters

Every role/profile in `config.yaml` (`models.generation`, `models.judge`, and
each `model_profiles.*` entry) sets a client-side `max_concurrent_requests`.
That number is a *client admission limit* — it caps how many requests Curator
will have in flight at once. It says nothing about whether the vLLM server can
actually process that many requests concurrently without queuing so long that
client-side `request_timeout` expires. Tuning the client number without ever
measuring the server side is exactly what produced the `saturation-500-001`
synchronized-timeout-wave incident (see `TASKS.md`'s "Follow-up result and
remediation (2026-08-05)" entry and this session's T18/T20 research notes):
128 concurrent long-output requests degraded the server enough that 180/225
requests failed within the same ~1-second window.

## Deployment topology to confirm first

All five model profiles point at the same host on different ports (from
`.env.example`, replace with the deployment's real values):

| Profile (`config.yaml`) | Env var | Example port | Purpose |
|---|---|---|---|
| `glm` | `GLM_BASE_URL` | `127.0.0.1:3005` (tunneled) | generation |
| `nemotron` | `NEMOTRON_BASE_URL` | `127.0.0.1:8001` | generation (alt) |
| `gemma_thinking` / `gemma_structured` | `LLM_BASE_URL` | `127.0.0.1:3005`/`8010` | judge |
| `gemma` | `GEMMA_BASE_URL` | `127.0.0.1:8002` | generation (alt) |
| `ministral` | `MINISTRAL_BASE_URL` | `127.0.0.1:3006` | judge (alt) |
| `qwen` | `QWEN_BASE_URL` | `127.0.0.1:8003` | generation (alt) |

Before trusting any per-profile concurrency number, confirm with the
deployment owner:

1. **Are any of these ports served by the same vLLM process / same GPU(s)?**
   If GLM and Gemma share a GPU, raising GLM's concurrency can starve Gemma
   even if Gemma's own client-side limit never changes, and vice versa. This
   is not visible from the Curator side at all — it requires the deployment
   owner's process/GPU topology.
2. **What is each server's `--max-num-seqs` and `--max-num-batched-tokens`?**
   vLLM documents `--max-num-seqs` as the maximum number of sequences
   processed in one scheduler iteration
   (<https://docs.vllm.ai/en/stable/cli/serve/#max-num-seqs>). A client
   concurrency setting above this number cannot improve throughput — it only
   grows the server-side queue, which is exactly what produces long queue
   time and client-side timeouts. `--max-num-batched-tokens` bounds the
   token budget per iteration and interacts with chunked prefill
   (<https://docs.vllm.ai/en/stable/configuration/optimization/>).

## Metrics to collect, and where they come from

vLLM's OpenAI-compatible server exposes Prometheus metrics at `/metrics` on
the same port as the API
(<https://docs.vllm.ai/en/latest/serving/metrics.html>). If the deployment
owner can `curl` this endpoint (or already scrapes it into Prometheus/Grafana),
the metrics below answer the four questions this runbook exists to answer.
Names are current as of vLLM's documented metrics list; confirm exact names
against the deployed vLLM version's `/metrics` output, since metric names have
changed across vLLM releases.

| Question | Metric family (name may vary by vLLM version) | What to look for |
|---|---|---|
| How long do requests wait before the server starts working on them? | `vllm:time_to_first_token_seconds` (histogram) and `vllm:request_queue_time_seconds` | Rising p50/p99 as concurrency increases is the direct signal that client concurrency has exceeded sustainable server capacity. |
| How fast does the server generate tokens once it starts? | `vllm:time_per_output_token_seconds` (TPOT) or `vllm:tokens_per_second` decode throughput gauges | Use this to sanity-check whether a given `max_tokens`/`output_rescue_max_tokens` budget is achievable within the configured `request_timeout` at all (see T18: rescue asks for 2-2.4x the ordinary token budget). |
| Is the server actually saturated, or queuing for another reason? | `vllm:num_requests_running`, `vllm:num_requests_waiting` | If `num_requests_waiting` grows while `num_requests_running` stays flat at (or below) `--max-num-seqs`, the client is sending more concurrent requests than the server will ever run at once — raising client concurrency further only adds queue time, not throughput. |
| Is KV-cache pressure limiting how many sequences can run at once? | `vllm:gpu_cache_usage_perc` (or `kv_cache_usage_perc` depending on version) | High (near 100%) cache usage caps effective concurrency independently of `--max-num-seqs`; this is the number that explains why two profiles sharing a GPU can starve each other even with generous per-profile client limits. |

If Prometheus scraping isn't already wired up, a single manual snapshot is
still useful:

```bash
curl -s http://<vllm-host>:<port>/metrics | grep -E \
  'time_to_first_token|time_per_output_token|num_requests_(running|waiting)|cache_usage'
```

Take this snapshot **while the pipeline is actively running a stage at its
current configured concurrency** (e.g. during a generation or cross-document
pass), not at idle — idle metrics say nothing about capacity under load.

## Translating results into `max_concurrent_requests`

Once real numbers exist, the target is: **client concurrency ≈ the server's
sustainable in-flight request count at an acceptable p99 queue time**, not a
number tuned by watching timeouts happen. A reasonable procedure:

1. Run a bounded load test (a `--limit` pilot, or a single generation/judge
   stage) at the *current* configured concurrency while capturing the
   `/metrics` snapshot above.
2. If `num_requests_waiting` is consistently > 0 and `time_to_first_token`
   p99 is a meaningful fraction of `request_timeout` (e.g. > 25%), client
   concurrency is already past the server's comfortable capacity — reduce it,
   or accept the queue time and raise `request_timeout` (see T18's rescue
   timeout for a worked example of this tradeoff being made explicitly rather
   than by trial and error).
3. If `num_requests_waiting` stays at 0 and `gpu_cache_usage_perc` has
   headroom, there is room to raise concurrency — raise it in a small,
   measured step (not a large jump like the historical 32 → 128 change that
   caused the synchronized timeout wave) and repeat the measurement.
4. Record the measurement (date, concurrency tested, queue time, cache usage)
   in `TASKS.md` next to the resulting `config.yaml` change, the same way
   existing timeout/concurrency tuning entries are documented, so the next
   person can tell a measured value from a guessed one.

## What this runbook deliberately does not do

- It does not set any `max_concurrent_requests`, `request_timeout`, or
  `submission_jitter_seconds` value. Those remain in `config.yaml`, tunable
  from the measurements this runbook produces.
- It does not assume all five profiles are independent. Confirming GPU/process
  colocation (topology question 1 above) is a prerequisite for trusting any
  single profile's measurement in isolation.
- It does not replace T20's submission jitter mitigation. Jitter smooths out
  *when* requests arrive; this runbook is about *how many* the server can
  actually sustain at once. Both matter and neither substitutes for the
  other.
