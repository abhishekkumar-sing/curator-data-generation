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

## Current pipeline-wide release blockers (reconciled 2026-08-02)

This index does not replace the research and implementation records below. It
separates code that can be completed locally from evidence that cannot be
created by unit tests or inferred from a successful structured response.

Implemented and locally verified in the current working tree:

- [x] Source-quality screening, intent planning, grounded QA blueprints,
  singular final generation, exact evidence/claim checks, independent judging,
  portfolio diversity gates, and leakage-safe train/eval exports.
- [x] Disabled-by-default NVIDIA question embeddings, secret-free endpoint
  probe/statistics, persistent ID-and-question-hash cache, human calibration
  pairs, and quality-aware semantic cluster selection guarded by an explicit
  calibrated threshold.

Remaining engineering:

- [x] Complete the packaged `nrl-curate all` command and researched saturation
  controller. Preserve the direct Python entry point and share one orchestration
  implementation; do not ship a subprocess wrapper or an unbounded pass loop.
- [x] Add a reusable per-endpoint structured-output capability probe for every
  generation/judge profile, not only the embedding endpoint, and persist a
  secret-free result before allowing a new deployment into a large run.
- [x] Implement adversarial unanswerable QA with constructed same-type
  distractors and an independent answerability judge. Ordinary retrieval/OCR
  failures must not become negative training examples.
- [x] Finish the cross-document atomic claim-to-source/path bindings,
  quality-aware best-of-N family selection, optional novelty passes, and
  retrieval-evaluation contexts (oracle, missing-source, wrong-edition,
  wrong-authority, and hard topical distractors).
- [ ] Finish explicit held-out manual/fold configuration, a frozen external
  human-reviewed evaluation set, and regression reporting. Generated test rows
  alone are not a gold evaluation set.
- [x] Finish OCR/source-registry provenance and corpus coverage work still
  listed below, including model/package revision, registered-source uniqueness,
  image-caption removal, and stratified rather than prefix-based pilot limits.
- [ ] Finish the remaining drafting field/claim extraction and validate
  citation/detail completeness on regenerated drafting outputs.

External validation required before release:

- [ ] Rotate the exposed embedding credential, enable the profile with the new
  key, and run the non-sensitive endpoint probe. Never commit the key or put it
  in a URL.
- [ ] Label duplicate/related/distinct calibration pairs, choose a development
  threshold at the required precision, and confirm it on a separate reviewed
  holdout before enabling semantic deletion.
- [ ] Run a user-controlled, fixed-seed, model-backed pilot spanning manuals,
  authority classes, QA/QA-CoT, cross-document relationships, and drafting;
  reconcile every planned request to a terminal lineage state.
- [ ] Perform stratified human review of accepted and rejected records, record
  reviewer disagreement where possible, and calibrate opener/type/extraction/
  length thresholds only from that evidence.
- [ ] Complete disjoint development/holdout judge reviews, run
  `nrl-curate calibrate-judge`, pin the passing artifact SHA-256 in
  `config.yaml`, and confirm the holdout precision target before an unbounded
  run. The command and gate are implemented; no human labels were fabricated.
- [ ] Run downstream retrieval and SFT evaluations before claiming that a
  diversity threshold, temporal curriculum, or saturation rule improves model
  behavior.

## GLM generation / Ministral judge migration (2026-08-05)

- [x] Restore the existing GLM deployment as the active/default generation
  profile and preserve its endpoint credential only in ignored `.env`. Its
  `/models` response omits `max_model_len` and `/tokenize` returns 404, so use a
  conservative explicit 32,768-token prompt-budget fallback rather than an
  unverified checkpoint maximum.
- [x] Add a distinct `ministral` judge profile for
  `mistralai/Ministral-3-14B-Instruct-2512` on the private port 3006 endpoint;
  keep the supplied credential out of tracked files.
- [x] Use native JSON-schema transport provisionally, cap judge concurrency at
  16, retain the role-level 2,048-token output ceiling, and apply the official
  production recommendation with `temperature=0.05`. Do not send Gemma's
  `enable_thinking` chat-template argument to this Instruct checkpoint.
- [x] Make profile sampling/template parameters override generic role defaults
  while preserving role-level `max_tokens` as a non-raisable safety ceiling;
  cover this distinction with regression tests.
- [x] Pass and persist the exact GLM generation probe. On 2026-08-05 it passed
  every nested sentinel/enum/list check in 37.328 s with fingerprint
  `61dbe1c71839baadb4bc7f4acc0704fa8dc26cbf21aac0ee7fb251b18a6981ca`.
- [x] Pass and persist the exact Ministral judge probe before resuming a large
  or full-pipeline run. The 2026-08-05 04:54 UTC attempt found the port down;
  after correcting the model ID, the 05:02 UTC probe passed every nested
  sentinel/enum/list check in 1.241 s with fingerprint
  `54bdd1d2afe550a1dafd06186e842be698b4047ae1cba7acde24e50a9851991b`.
- [x] Manually verify both active OpenAI-compatible deployments on 2026-08-05:
  `/models`, ordinary chat completion, and raw strict JSON-schema completion
  all succeeded for GLM and Ministral. GLM returned `GLM_OK`; Ministral returned
  `MINISTRAL_OK`; both returned the required `{"status":"OK","value":7}`
  object under schema constraints.
- [x] Correct the Ministral request ID from the Hugging Face repository path,
  which the server rejected with HTTP 404, to its advertised deployment ID
  `ministral-3-14b-instruct-2512`. Record the server-reported 65,536-token
  context window and vLLM `0.25.0-076f6001` deployment identity.

Research basis:

- The official model card identifies this as the FP8 Instruct checkpoint,
  recommends vLLM >=0.12 with `mistral-common >=1.8.6`, advertises native JSON
  output, and recommends temperature below 0.1 in production:
  <https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512>.
- The same card advertises a 262,144-token checkpoint limit but explicitly
  permits a smaller served `--max-model-len`; Curator initially used 32,768 as
  a conservative fallback, then replaced it with the private server's verified
  65,536-token deployment limit.

## Gemma thinking judge migration (2026-08-07)

- [x] Select the existing `gemma_thinking` profile as the active and committed
  default judge while retaining GLM as generation. The roles remain independent:
  different served models, deployment identities, and private endpoints.
- [x] Verify without printing secrets that ignored `.env` already contains the
  requested `google/gemma-4-31B-it` model, port-8010 base URL, and deployment
  identity; update only `JUDGE_PROFILE`. Keep the supplied credential exclusively
  in ignored `.env` and placeholders in tracked examples.
- [x] Preserve the requested thinking request contract for this profile:
  temperature 1.0, top-p 0.95, top-k 64, and
  `chat_template_kwargs.enable_thinking=true`. The judge role retains its
  2,048-token ordinary output ceiling, 4,096-token missing-row rescue, and the
  profile's eight-request concurrency cap.
- [x] Update `.env.example`, the README, committed default configuration, and
  regression tests so the selected role and its effective parameters cannot
  drift silently. Ministral remains an explicit fallback profile.
- [x] Qualify the exact Gemma judge deployment. The initial judge-only
  structured-output probe at 2026-08-07 03:45 UTC failed at TCP connection
  setup to port 8010;
  zero requests succeeded and zero tokens were processed. Fingerprint
  `fc409fa091813e4d5fcbe21fe509170dddfd9dc3d6432053966e8b5f5608551e`
  remains a persisted failed probe, not permission to run. The replacement
  shared-gateway profile passed on 2026-08-08 under its new exact fingerprint,
  recorded below.

## Shared LiteLLM Gemma route (2026-08-08)

- [x] Verify the replacement credential against the shared port-3005 LiteLLM
  gateway. Its key-scoped model discovery advertises `gemma-4-31b-it`, and a
  live completion returned the requested `GEMMA_OK` response.
- [x] Confirm that the replacement credential is Gemma-only: the gateway
  rejects `GLM-5.2-NVFP4-FP8` with `key_model_access_denied`. Keep role-specific
  credentials even though Gemma and GLM share one gateway URL.
- [x] Preserve
  `LLM_DEPLOYMENT_ID=gemma-4-31b-it-10.180.148.183-8010-v1` as the supplied
  underlying deployment identity while routing requests through port 3005.
  Deployment identity is intentionally independent of the transport URL.
- [x] Qualify the updated `gemma_thinking` judge profile through the shared
  gateway and persist exact successful structured-output probe fingerprint
  `7819f7ef5afc035be1f03480c34e5d1e11868a71f074b5438f85f457b600c60a`.
  The live probe on 2026-08-08 passed status, labels, and nested integer-list
  checks in 6.992 seconds with thinking enabled, temperature 1.0, top-p 0.95,
  top-k 64, and a 2,048-token judge ceiling.
- [x] Requalify the unchanged GLM 5.2 generator through the same gateway. Exact
  fingerprint `61dbe1c71839baadb4bc7f4acc0704fa8dc26cbf21aac0ee7fb251b18a6981ca`
  passed every structured-output check in 12.103 seconds. The active roles are
  therefore GLM 5.2 generation and Gemma 4 31B IT judging.

## Comprehensive smoke feature enablement (2026-08-08)

- [x] Enable proposition extraction, connected reasoning paths, temporal
  artifacts, path QA, one configured cross-document novelty pass, and
  convergence-based saturation. Keep bounded smoke runs under an explicit
  positive `--max-passes` override.
- [x] Enable semantic embedding analysis with the NVIDIA
  `llama-nemotron-embed-1b-v2` endpoint contract. Keep semantic selection
  disabled until duplicate/related/distinct labels establish a threshold and a
  separate reviewed holdout validates it.
- [x] Insert the user-supplied embedding credential only in ignored `.env` and
  run the non-sensitive capability probe. On 2026-08-08 the configured
  `nvidia/llama-nemotron-embed-1b-v2` endpoint returned one valid 1,024-element
  vector under secret-free cache fingerprint `e57198334cceecb830ba`; the
  Gemma-only gateway credential was not reused.
- [ ] Rotate this embedding credential before release because it was exposed in
  conversation. Re-run the probe after rotation; the current successful result
  permits the requested smoke test but does not close the release credential
  hygiene blocker.
- [x] Set ordinary GLM generation, GLM output rescue, Gemma judging, and judge
  output rescue concurrency to 45 for the comprehensive smoke test. Ministral's
  unused fallback profile remains at 16. Record endpoint stability, queueing,
  and timeout yield from the smoke before raising any active limit again.
- [x] Verify and regression-lock every Gemma profile to temperature 1.0, top-p
  0.95, and top-k 64. Both thinking-enabled and non-thinking profiles declare
  the values explicitly, and the active Gemma judge rescue inherits the same
  sampling contract while changing only its completion ceiling.

## Reference saturation audit (2026-08-05)

- [x] Inspect the reference CLI, configuration, state machine, tests, and run
  documentation rather than inferring behavior from its command examples.
- [x] Confirm that reference `--max-passes 0` means no operational pass cap:
  its `while True` loop ignores the pass-bound condition when zero and stops
  only when no active parent remains.
- [x] Confirm that reference saturation is per parent, not a corpus-wide gain
  ratio. Each parent receives its own exclusion history and must produce two
  consecutive successful zero-novelty passes by default; finding a novel item
  resets that parent's empty-pass counter.
- [x] Confirm that missing responses, invalid-only output, rescue overflow, and
  generation failure never advance saturation. They are rescued or quarantined
  and leave the run incomplete; state is checkpointed after every pass.
- [x] Compare this with Curator's current implementation: Curator applies a
  global cross-document marginal-gain rule, requires `max_passes >= 1`, and
  treats two complete passes below 5% gain as convergence. Omitting the flag
  while `saturation.enabled: false` still selects one pass.
- [x] Adopt unlimited `--max-passes 0` for Curator's cross-document family and
  test per-parent completion/exclusion state, failure quarantine/reactivation,
  prompt-growth budgeting, deterministic resume, and an explicit incomplete
  terminal state. Do not merely relax the CLI validator or copy the reference's
  fuzzy-string novelty rule. Implemented with exact per-parent outcomes after
  deterministic validation, existing near-duplicate selection, and independent
  judging; two successful zero-novelty passes complete a parent, novelty resets
  its streak, transient failures reactivate only on a later invocation, prompt
  overflow remains quarantined, and replay must reproduce checkpointed outcomes.
- [ ] Extend the shared controller to single-document QA, drafting, and any
  other family intentionally made iterative before calling `--max-passes 0` a
  whole-pipeline saturation run. The current flag repeats cross-document
  novelty generation only; the README states this boundary explicitly.
- [x] Validate the Curator implementation with 8 deterministic saturation
  regressions (unlimited semantics, independent parent completion, novelty
  reset, failure quarantine, resume reactivation, replay identity, population
  mismatch, hard limit, and single-pass behavior) and the complete 155-test
  `tests/nrl_procurement` suite on 2026-08-05. Ruff and `git diff --check` pass;
  no model-backed run was started by the implementation agent.

## GLM single-request tail stall (2026-08-05)

- [x] Diagnose `saturation-500-001` from persisted request timestamps rather
  than the progress display alone. Request index 75 began at 05:33:17 and
  succeeded at 06:03:47: the 1,800-second role timeout expired and its retry
  succeeded in roughly 30 seconds. The stage ultimately persisted all 500
  unique request indices with no terminal response errors.
- [x] Measure the successful blueprint latency distribution before copying the
  reference setting. This deployment produced an approximately 256-second
  median, 339-second p99, 365-second ordinary maximum, and 36 successful
  requests above 300 seconds. The reference GLM timeout is 300 seconds, but
  applying it here would create avoidable retries under the observed load.
- [x] Confirm that the reference combines a model-specific GLM timeout with
  one work-conserving retry and explicit missing-row rescue/quarantine. Its old
  deferred-retry monkey patch is not copied because Curator already contains
  the equivalent in-task scheduler in commit `f956b921` with regression tests.
- [x] Initially configure the active GLM profile for a measured 600-second
  timeout and one retry. This bounded the earlier silent tail and provided the
  fresh load evidence below; pass-5 evidence subsequently superseded this
  setting with the 1,200-second policy recorded below.
- [x] Separate transport tuning (`request_timeout`, retries, concurrency, RPM,
  and TPM) from scientific config/model cache identity while retaining it in
  run provenance. Future tuning can reuse compatible partial/completed stage
  artifacts; model/deployment, structured-output mode, sampling parameters,
  schema contracts, inputs, and pipeline revision remain fingerprinted.
- [x] Add regressions proving the GLM override resolves to its configured
  timeout with one retry and that endpoint URL/timeout plus config-level
  transport tuning do not
  alter the scientific contract or stage fingerprint. The complete procurement
  suite plus Curator retry/credential regressions passes 166 tests; Ruff and
  `git diff --check` pass.
- [x] Validate the 600-second boundary on a fresh bounded load test. Pass 5
  showed multiple otherwise non-terminal requests reaching that boundary at
  concurrency 32, so 600 seconds is now retained as measured historical
  evidence rather than the active policy.

Follow-up result and remediation (2026-08-05):

- [x] The resumed 225-request cross-generation replay provided the required
  adverse load evidence: only 45 requests succeeded and 180 timed out on both
  attempts at 600 seconds. The synchronized failure wave shows endpoint/queue
  collapse at 128 concurrent long-output requests, not an ordinary latency tail
  that should be addressed by restoring a 1,800-second wait.
- [x] Override the GLM profile to 32 concurrent ordinary requests. The observed
  long-output throughput was about 5 requests/minute, so 32 remains enough to
  keep the endpoint occupied while preventing nearly the whole pass from
  accumulating inside the active server queue. Cap generation rescue at 16 and
  judge rescue at 8 concurrent requests.
- [x] Preserve the original run artifacts. `saturation-500-001` still contains
  the original 528-record pass-1 generation checkpoint, its matching 161 judge
  decisions, and completed checkpoints through pass 4. The failed restart wrote
  separate 105-record generation and 26-record judge checkpoints; it did not
  overwrite the earlier immutable artifacts or saturation observations.
- [x] Add historical saturation-replay checkpoint selection. Replay may select
  an integrity-hash-verified artifact from an earlier attempt when its input,
  role, and response-affecting model identity match, even across the one-time
  transport/scientific-contract migration. Replay still fails closed unless the
  reconstructed per-parent outcomes and novel record IDs exactly equal the
  persisted saturation observation.
- [x] Verify the selector read-only against the real run: pass-1 generation now
  resolves to checkpoint `0f12dd...` with 528 records and pass-1 judging to
  `523e65...` with 161 records, rather than the degraded restart artifacts.
- [x] Add replay and concurrency regressions. The focused pipeline suite passes
  112 tests; the complete procurement suite passes 160 tests.
- [ ] Resume the same run after this revision and confirm passes 1–4 replay from
  historical checkpoints without model calls, then measure fresh pass-5 GLM
  behavior at concurrency 32. Keep this external endpoint validation open until
  its result is recorded.
- [x] Confirm the resumed run reached fresh pass 5 with 199 requests and the
  configured concurrency of 32, proving historical passes 1–4 replayed without
  another mismatch. At 12:19 UTC, 75 distinct responses were durably persisted,
  all 75 were successful, no terminal response error existed, and 32 requests
  remained active. Six requests had reached their first 600-second timeout and
  entered their one allowed retry; these warnings are not permanent failures.
  Leave the final endpoint-validation item open until the stage and its targeted
  rescue, if needed, reach terminal statistics.
- [x] Raise the active GLM request timeout from 600 to 1,200 seconds at the
  user's request after the fresh pass-5 evidence. Preserve one retry and
  concurrency 32, yielding a twenty-minute attempt and an approximately
  forty-minute bounded two-attempt tail. Keep timeout and concurrency outside
  scientific cache identity; the active process retains 600 seconds in memory,
  while its next invocation loads 1,200 seconds without invalidating completed
  checkpoints.

The concurrency reduction is consistent with vLLM's documented scheduler
control: `--max-num-seqs` is the maximum sequences processed in one iteration,
and vLLM states that real deployments should set it in engine configuration.
The client cap of 32 is a conservative admission limit derived from this
deployment's observed failure wave; it is not asserted to be a universal vLLM
optimum: <https://docs.vllm.ai/en/stable/cli/serve/#max-num-seqs>.

Research basis:

- aiohttp's official `ClientTimeout` contract defines `total` as the ceiling
  for connection establishment, request sending, and response reading, and
  documents 300 seconds as its general default:
  <https://docs.aiohttp.org/en/stable/client_reference.html#clienttimeout>.
- vLLM documents that its online server implements the OpenAI-compatible chat
  completion endpoint used here; client-side liveness deadlines and retry
  accounting therefore remain Curator/LiteLLM responsibilities:
  <https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/>.

## Structured-output max-token recovery (2026-08-05)

- [x] Diagnose the live `saturation-500-001` evidence by stage. The ordinary
  blueprint and cross-generation truncations recovered on their same-budget
  retry, but cross-judge pass 1 permanently lost 6/167 singular judgments at
  the 2,048-token ceiling; one of those rows also encountered a disconnected
  server. This is a terminal output-completeness defect, not saturation.
- [x] Audit `/home/abhishek/nrl_curator_native_glm52` before changing Curator.
  Its relevant mechanism detects request IDs missing from successful outputs,
  retries only those rows in a separate rescue working directory with a larger
  output reserve, checks the complete rendered prompt against the effective
  context window, and leaves unresolved rows explicitly incomplete.
- [x] Implement the same bounded recovery semantics for Curator's single QA
  blueprint/generation/judge and iterative cross-generation/cross-judge stages.
  Normal generation remains capped at 8,192 tokens and normal judging at 2,048;
  only terminally missing generation rows receive a 12,000-token rescue and
  only terminally missing judge rows receive a 4,096-token rescue.
- [x] Give every rescue its own logical stage/checkpoint and include the rescue
  ceiling in that checkpoint's input identity. Treat rescue ceilings as
  recovery tuning so adding them does not invalidate compatible completed
  primary-stage artifacts.
- [x] Preflight every rescue against the selected deployment's configured or
  server-measured context window and persist over-budget rescue inputs as audit
  rejections rather than submitting an impossible request.
- [x] Disable rescue while replaying already-checkpointed saturation passes.
  This preserves deterministic historical outcomes; quarantined failures are
  reactivated by the controller on a later invocation and receive rescue in a
  new pass instead of rewriting prior saturation evidence.
- [x] Add regressions for role-specific rescue ceilings, missing-row-only
  dispatch, separate rescue checkpoint identity, and scientific cache
  compatibility. The focused pipeline suite passes 112 tests; the complete
  procurement suite passes 159 tests.
- [ ] Resume or run a fresh model-backed pilot after the process has loaded this
  revision, and verify non-zero `*_output_rescue` stage events plus zero
  unresolved max-token omissions. A Python process already running before this
  change continues using its in-memory old code and is intentionally not
  interrupted by this implementation.

Research basis:

- Instructor documents `IncompleteOutputException` as a response truncated by
  the token limit and lists increasing `max_tokens`, simplifying the response
  model, partial streaming, or splitting the task as remedies. This pipeline
  already uses singular atomic judges and bounded response lists, so a targeted
  larger-budget retry is preferred over partial judgment acceptance:
  <https://python.useinstructor.com/api/>.
- vLLM defines `max_tokens` as the maximum generated tokens per output sequence
  and separately defines `max_model_len` as the combined prompt/output context
  limit. The rescue therefore raises only the completion ceiling and still
  performs a full rendered-context preflight:
  <https://docs.vllm.ai/en/latest/api/vllm/index.html> and
  <https://docs.vllm.ai/en/latest/api/vllm/config/model/>.
- Globally raising every request to the rescue ceiling was rejected because the
  observed failure tail is sparse and larger live generation reservations can
  reduce serving efficiency. Accepting partial JSON was rejected because a
  missing field can silently change an independent quality decision.

## Generation endpoint migration (2026-08-03)

- [x] Add a dedicated `gemma_thinking` generation profile using the requested
  `MODEL`, `LLM_BASE_URL`, and `LLM_API_KEY` environment contract; make it the
  configured default and active local generation profile in place of
  Nemotron. Preserve Nemotron as an explicit fallback profile.
- [x] Send `temperature=1.0`, `top_p=0.95`, `top_k=64`, and
  `extra_body.chat_template_kwargs.enable_thinking=true` per generation
  request. Move the old non-thinking setting out of the role-level defaults so
  it cannot override the selected profile.
- [x] Keep the independent Gemma judge explicitly non-thinking, and add a
  regression test proving the generation and judge profiles retain their
  distinct chat-template settings.
- [x] Store the supplied credential only in ignored `.env`; commit only
  placeholder environment documentation.
- [x] Pass and persist the structured-output probe for the exact new endpoint
  before using it for an unbounded run. On 2026-08-03 the generation probe
  passed all nested sentinel/enum/list checks with thinking enabled in 7.407 s;
  the secret-free deployment fingerprint is
  `2a98fb7263194c25c284a98f50f7b96d03c3347d74bce91ace0c2f627ccee1a7`.
- [x] Reconcile the first full-pipeline smoke interruption. The blueprint and
  final-generation stages completed 20/20 and 17/17 requests respectively with
  empty failed-request files; the later cross-stage bootstrap call received
  vLLM's fatal `EngineCore` 500 before any cross request was processed.
- [x] Limit the new 31B thinking generator to eight concurrent client requests
  until its deployment-specific capacity is measured. vLLM documents
  `--max-num-seqs` as the per-iteration sequence cap that must be set for real
  deployments: <https://docs.vllm.ai/en/latest/cli/index.html>.
- [x] Skip LiteLLM's header-discovery completion whenever explicit RPM and TPM
  limits are both configured. Real requests still use ordinary bounded retries
  and persistent failure audit. vLLM reports `EngineCore` failures as fatal and
  subsequent requests can continue returning 500 until server recovery, as
  demonstrated in upstream issue 23582:
  <https://github.com/vllm-project/vllm/issues/23582>.
- [x] Recover the thinking-generation deployment. Smoke-002 reused valid
  checkpoints for 20 blueprints and 17 generated candidates and produced 26
  cross-generation candidates with the eight-request concurrency cap; its
  generation failed-request files are empty.
- [ ] Recover the separate non-thinking judge at `10.180.148.183:3009`, which
  refused both cross-judge connections on smoke-002 before any cross candidate
  could be judged. A fresh exact-profile probe at 2026-08-03 11:40 UTC was also
  refused at the TCP connection with zero tokens processed, so the endpoint is
  not yet reachable from the Curator host. Once reachable, rerun the judge
  structured-output probe, resume smoke-002 from its checkpoints, and verify
  non-zero accepted cross-document and drafting records. Do not substitute the
  generator as judge merely to bypass this independence check.

## P0 quality-remediation contract (2026-08-02)

Status: primary-source research and current-code/run audit completed before
implementation. The seven user-reported defects are release blockers. The
`qa-qacot-full-003` measurements remain a pre-blueprint baseline and will not
be misrepresented as evidence for post-fix quality.

Research findings and implementation decisions:

- [Type-controlled question-generation research](https://aclanthology.org/2022.starsem-1.22/)
  and [RAST](https://aclanthology.org/2023.emnlp-main.104/) support explicit
  question-type/style conditioning rather than asking a model to be vaguely
  diverse. Add a deterministic, source-compatible question-style plan outside
  the response schema; reject source-attribution and role-preamble templates;
  measure both style and intent concentration after judge attrition.
- [Explicit Diversity Conditions for Effective Question Answer Generation](https://aclanthology.org/2024.lrec-main.601/)
  reports stronger diversity from explicit conditions. Preserve the existing
  grounded intent planner, lower the provisional concentration ceiling, and add
  a minimum effective-type release gate. Never force a type unsupported by the
  passage merely to satisfy a quota.
- Persona research finds generic generation can neglect an attached persona;
  [PAL](https://aclanthology.org/2025.tacl-1.77/) makes semantic persona
  alignment explicit, while [Quantifying the Persona Effect](https://aclanthology.org/2024.acl-long.554/)
  shows persona variables often explain little variance. A persona label or
  “As a …” prefix is therefore insufficient. Blueprints must state a concrete
  role-relevant information need, role prefixes are forbidden, and the
  independent judge must separately attest persona relevance. Use
  `general_user` when specialization is not material.
- [vLLM tool-calling guidance](https://docs.vllm.ai/en/latest/features/tool_calling/)
  guarantees parseable structure, not semantic quality, and recommends making
  the schema contract explicit in the prompt. [Instructor retry guidance](https://python.useinstructor.com/learning/validation/retry_mechanisms/)
  feeds validation errors into bounded reasks. Keep strict schemas, add only
  narrow JSON-list unwrapping for the repeatedly observed stringified-list
  transport defect, retain every repair in audit lineage, and allow two bounded
  validation retries. Invented enums and missing semantics remain failures.
- vLLM's [Gemma 4 structured-output recipe](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html)
  notes schema descriptions are not automatically visible to the model. Keep a
  singular compact judge contract and explicit prompt fields. Fix role/profile
  parameter merging so the intended judge completion budget is actually used;
  bound issue cardinality/text and preserve prompt-budget checks.
- [LLM-Rubric](https://aclanthology.org/2024.acl-long.745/) finds
  multidimensional rubric signals require human calibration, and
  [Judging the Judges](https://aclanthology.org/2025.ijcnlp-long.18/) documents
  non-random judge bias. Add a development/holdout calibration command that
  selects a score threshold only from immutable completed reviews, reports
  holdout precision/recall/F1 and confusion counts, fingerprints inputs and the
  feature contract, and gates unbounded runs on a pinned passing artifact.
- Zero cross-document or drafting records are absence of evidence, not evidence
  of poor or good quality. Enabled tracks must have configured minimum accepted
  counts, terminal lineage, review sampling, and manifest/validator gates. Unit
  tests can close the enforcement code, but only a live model-backed run plus
  human review can close quality evidence for those stages.

Implementation checklist:

- [x] Add balanced source-compatible question-style planning, template-prefix
  rejection, style metrics, and post-judge release gates.
- [x] Tighten intent concentration and require adequate effective intent
  coverage without inventing unsupported intents.
- [x] Add blueprint `persona_need`, forbid cosmetic role prefixes, and require
  independent `persona_relevant` judgment for specialized personas.
- [x] Add narrow audited stringified-list recovery and a second bounded
  validation retry; keep enum/tool/semantic failures fail-closed.
- [x] Correct nested role/profile generation-parameter merging, enlarge the
  effective judge budget, compact issue output, and test the actual resolved
  configuration.
- [x] Add immutable development/holdout judge calibration, pinned-artifact
  loading, calibrated threshold use, and a full-run gate.
- [x] Require nonzero cross-document/drafting quality-evidence minima when
  enabled; report and validate the gates. Keep live quality review open until a
  post-fix run actually produces those records.
- [x] Run focused and full procurement tests, static checks, build verification,
  update exact results here, and commit the coherent remediation milestone.
  Verified on 2026-08-02: `146 passed` for `tests/nrl_procurement`, Ruff passed
  for pipeline/tests, `git diff --check` passed, both CLI help paths passed, and
  `uv build` produced the sdist and wheel; the milestone is recorded in Git
  history with the remediation commit.
- [x] Fix the `quality-smoke-001` terminal-lineage crash discovered on
  2026-08-02: failed blueprint audit rows now retain `planned_answerable`, and
  downstream rejection materialization is regression-tested. Bound blueprint
  overproduction to four items and recover a scalar `must_cover` as one audited
  list item; empty/non-semantic evidence still fails closed.
- [x] Resume `quality-smoke-001` after this fix and inspect the terminal
  manifest plus accepted/rejected cross-document and drafting samples. Both
  structured-output probes passed; terminal lineage was complete; the manifest
  correctly remained `partial` with 2 QA, 0 QA-CoT, 0 cross-document, and 0
  drafting records. Of 20 single requests, four reached judging and two passed;
  cross generation materialized 21 candidates for 10 requests but only one
  deterministic survivor failed source ablation/persona/quote review; both
  drafting candidates failed exact field/block support binding.
- [x] Implement the deterministic smoke follow-ups without weakening quality:
  normalize only unambiguous CoT operation aliases; remove only an exact
  `As a <declared persona>,` wrapper; inject the preplanned cross serialization
  shape and discard rationale returned for plain cross QA; allow edition/page
  numbers in cross claims only when present in source metadata; and reconcile
  drafting support only to exact source spans, dropping invalid extra labels
  and promoting exact field-claim support to its block. Absence claims,
  modality/qualification checks, source ablation, and exact citation gates
  remain fail-closed. Verified with `150 passed`, Ruff, and `git diff --check`
  on 2026-08-02.
- [x] Manually inspect the two accepted smoke QA records. One was grounded and
  useful; the other made an unsupported contrast that standard procurement
  “does not mandate” the stated GTE precondition. Extend the high-confidence
  absence detector to `does not require/mandate/specify`, with claim- and
  answer-level regression coverage; do not count that record as quality evidence.
  Final verification: `151 passed`, Ruff clean, and `git diff --check` clean.
- [ ] Run a new post-reconciliation smoke attempt and confirm that any increase
  in yield comes from the audited repairs above; manually inspect every accepted
  cross-document and drafting record before changing quality thresholds.

## P0 completion plan — researched implementation contract (2026-08-02)

Status: repository/reference audit and primary-source research complete before
implementation. This section is the acceptance contract for the current P0
work; checked items elsewhere are reused, not reimplemented.

Research findings and decisions:

- [NovelSum (ACL 2025)](https://aclanthology.org/2025.acl-long.908/) finds
  useful instruction-data diversity depends on inter-sample novelty and local
  information density. Saturation will therefore use marginal *accepted*
  semantic/lexical novelty, not a generator's claim that it is finished and
  not an unbounded “generate all” loop.
- [SQuAD 2.0](https://aclanthology.org/P18-2124/) constructs difficult
  unanswerables around plausible same-type answers in the passage. The
  pipeline will derive an isolated negative from a valid answerable seed,
  record the altered/missing premise and a same-type distractor, then require
  an independent full-context answerability judgment. Retrieval/OCR failures
  are never negative labels.
- [SQuAD2-CR](https://aclanthology.org/2020.lrec-1.667/) separates the reason
  for unanswerability from the plausible answer span. Negative records will
  retain both a machine-readable gap reason and the distractor provenance.
- RAG evaluation research separates retrieval from grounded generation and
  warns that ideal oracle contexts do not characterize behavior under imperfect
  retrieval. Each eligible record will therefore carry deterministic oracle,
  missing-source, wrong-edition, wrong-authority, and hard-topical context IDs;
  golden chunks can never be sampled as distractors.
- [NIST AITE](https://pages.nist.gov/ai-technology-evaluation/) uses blind,
  sequestered data to reduce train/test contamination, while the
  [NIST AI RMF measurement guidance](https://airc.nist.gov/airmf-resources/playbook/measure/)
  requires test sets, metrics, and evaluation procedures to be documented.
  Generated folds remain development data; a frozen external human-reviewed
  file is hash-pinned, excluded from all training exports, and scored only by a
  separate regression command.
- [W3C PROV-O](https://www.w3.org/TR/prov-o/) distinguishes source entities,
  derivation activities, revisions, and primary sources. Corpus manifests will
  pin source/content hashes and OCR software/model revisions, reject duplicate
  registered files or hashes, and report coverage by manual/category/page band.
- Chandra 2 explicitly generates layout blocks and image captions. Image and
  figure blocks/captions must be removed from model-visible text, while their
  existence remains visible in corpus-quality metrics.
- The reference repository is useful for its checkpointed parent state,
  explicit negative-construction ideas, and source registry. Its open-ended
  fuzzy saturation loop, runtime patches, and generated-only evaluation corpus
  are not copied.

Implementation acceptance criteria:

- [x] Add a packaged `nrl-curate` CLI whose `all` command calls the same Python
  orchestration function as `generate.py`; include bounded `--max-passes` and
  preserve the direct script entry point. No subprocess wrapper.
- [x] Add a checkpointed saturation controller with a hard maximum, minimum
  marginal-novelty gain, patience, terminal failure/quarantine accounting, and
  deterministic resume. Only successful valid batches count as saturation
  observations.
- [x] Reuse the exact-transport nested structured-output probe and large-run
  gate already implemented for generation and judge roles. Extend tests/config
  validation so every selected profile fingerprint is independent and stale or
  failed results cannot authorize `all`.
- [x] Add isolated adversarial-unanswerable schemas, construction stage,
  same-type distractor provenance, independent answerability judge, and a
  fail-closed promotion rule. Enable a nonzero target only after this exists.
- [x] Persist atomic cross-document claim bindings where every claim names its
  exact source IDs/citations; rank best-of-N siblings by deterministic validity,
  independent judge result, complete two-source support, qualification
  preservation, and novelty. Write auditable loser reasons.
- [x] Wire the bounded novelty controller into additional generation passes.
  Deterministic retrieval contexts are complete: oracle, missing-source,
  wrong-edition, wrong-authority, and hard topical distractors are emitted with
  golden-chunk exclusion enforced.
- [x] Add explicit manual-to-fold configuration and validation, plus a
  hash-pinned external human-reviewed evaluation registry. Emit regression
  reports with baseline deltas without copying frozen rows into generated/SFT
  exports. Supplying and approving the external file remains an external
  validation item, not repository engineering.
- [x] Reject duplicate registered source paths and hashes; require complete OCR
  provenance including package and model revision; remove Markdown/HTML image
  blocks and generated captions from generation text; report registered-source
  and stratified pilot coverage.
- [x] Complete drafting block-level field/claim extraction. Every material
  field/claim must bind to exact manual, tender, or instruction evidence and
  every citation detail must include sufficient source identity and offsets (or
  a typed tender-field binding).
- [x] Add focused regressions for every contract, run the full procurement and
  affected Curator suites, update this record with exact results, and commit in
  coherent milestones. Live model/human evidence remains explicitly pending.

Implementation progress recorded during this work:

- [x] Added `nrl-curate all`, `probe-structure`, `validate-run`, and `regress`
  in-process dispatch. `generate.py`, the probe, and run validator accept an
  argument vector, so the packaged and direct entry points share orchestration.
  The Poetry package includes `pipelines`; top-level and delegated help work,
  and `--max-passes` now drives the controller rather than merely being parsed.
- [x] Added atomic saturation checkpoints keyed by policy/family, hard
  `max_passes`, marginal accepted-novelty gain, patience, strict count
  reconciliation, deterministic resume, and explicit incomplete hard-limit
  status. Failed or invalid passes do not advance saturation patience.
- [x] Wired optional cross-document novelty passes to independently
  fingerprinted/resumable generation and judge checkpoints. Later prompts list
  prior accepted questions, fuzzy duplicates are auditable rejections, only
  distinct judge-accepted parent outcomes count as marginal novelty, every
  pass is reconciled, and the manifest distinguishes saturation convergence
  from an incomplete hard-limit stop. Missing, prompt-quarantined, or malformed
  judge outcomes cannot count as low novelty. Disabled configuration is truly
  one pass, and base refresh aliases expand to numbered pass checkpoints.
- [x] Extended the structured-output probe CLI to resolve active profiles,
  explicit named profiles independently for generation and judge roles, or all
  configured role/profile pairings. Exact role, deployment, transport mode,
  generation parameters, and package versions remain fingerprint-bound; a
  different role/profile result cannot authorize the selected large run.
- [x] Added adversarial negative planning from accepted answerable seeds,
  deterministic same-type non-golden distractors, singular missing-premise
  generation, independent full-context answerability judgment, fail-closed
  promotion, and generated/judged/rejected audit artifacts. The configured
  target is now 10% rather than silently training on retrieval failures.
- [x] Added cross-document claim-level `source_ids` and `citation_ids`,
  bidirectional binding validation, three-sibling generation, quality-aware
  post-judge selection, and auditable lower-quality sibling rejection.
- [x] Added deterministic retrieval-evaluation contexts with exact golden IDs
  and controlled missing-source, wrong-edition, wrong-authority, and topical
  distractor variants. Non-oracle distractors are checked for golden leakage.
- [x] Added OCR provenance contract v2 fields for model revision, package
  revision, package version, generation time, source/Markdown hashes, and page
  counts; registered path/content duplication and declared hash mismatches now
  fail closed. Existing three PDF caches correctly classify as `legacy` until
  regenerated. Chandra image/caption lines are removed from generation text and
  counted, and pilot selection now reports stratum coverage.
- [x] External evaluation engineering is complete: explicit manual folds are
  validated (including amendment families), exports reject records spanning
  folds, approved-review loading is hash-pinned, generated/frozen overlap is
  fail-closed, and `nrl-curate regress` emits deterministic coverage, answer,
  answerability, citation-recall, and baseline-delta metrics. Generated
  validation/test rows are labelled development-only. The actual independently
  reviewed frozen file, SHA-256, and baseline remain external evidence and are
  intentionally not synthesized by this pipeline.
- [x] Added 7 focused evaluation/fold tests covering fold completeness,
  amendment co-location, cross-fold rejection, frozen hash/approval checks,
  overlap detection, and regression metrics.
- [x] Added atomic drafting field claims with exact block indices, rendered
  values, and block-local manual/tender/instruction support. Material supported
  blocks, aggregate evidence, and tender facts must all be claim-bound.
  Tender facts now receive stable per-fact citation IDs plus seed identity,
  fact index, exact fact text, and offsets; compact drafting exports preserve
  field claims and full citation details. Eight drafting-focused tests pass.
- [ ] Regenerate a model-backed drafting sample under the new schema and review
  field-claim/citation completeness and yield; repository tests cannot supply
  this live endpoint and human-review evidence.
- [x] Focused validation so far: 12 saturation/structure-probe tests, 8
  provenance/unanswerable tests, and 6 cross-selection/retrieval/unanswerable
  tests pass; Ruff passes for each completed slice.
- [x] CLI/saturation/cross/probe focused suites now pass 18 tests in aggregate;
  package import and delegated `all --help` were also verified. Poetry itself
  is not installed in the project environment; the final validation therefore
  used `uv build` with Poetry Core's declared build backend.
- [x] Final local validation: all 140 procurement tests pass; Ruff reports no
  issues across the pipeline/tests; `git diff --check` and YAML parsing pass.
  `uv build --wheel` succeeds, the wheel contains the pipeline modules, and its
  console entry point is exactly
  `nrl-curate=pipelines.nrl_procurement.cli:main`.
- [x] Affected Curator request-processor tests produced 17 passes and 13 skips.
  One unrelated OpenAI processor test requires `OPENAI_API_KEY` at constructor
  time and fails in this credential-free environment; this change neither
  modifies that processor nor supplies external credentials. A repository-wide
  collection finds 425 tests, but the run terminates during the code-executor
  suite after its first test in this environment, so it is not represented as
  a successful full-suite run.
- [x] Committed the coherent P0 implementation milestone as
  `feat: complete procurement P0 pipeline controls`. Live endpoint pilots,
  independently reviewed frozen data, credential rotation, and human review
  remain external release evidence and are not closed by this commit.

## Completed-run audit — `qa-qacot-full-003` (2026-08-02)

Status: read-only baseline audit complete. This run was generated before commit
`c3246913`: its logs and cache use `CandidateBatch`, and its output contains no
`qa_blueprints_audit.jsonl`, `source_quality_rejected.jsonl`,
`semantic_calibration.jsonl`, or `semantic_rejected.jsonl`. It therefore cannot
validate the newly committed blueprint, singular-response, source-quality,
question-type portfolio, or semantic-calibration implementation.

Measured improvements relative to `qa-qacot-full-002`:

- [x] Exported 1,572 accepted records: 1,048 QA and 524 QA-CoT. QA-CoT is
  33.33% of canonical output and 407/1,159 (35.12%) of train-only SFT output,
  up from 330/2,493 in the earlier reviewed corpus.
- [x] `qa_sft.jsonl` and `qa_cot_sft.jsonl` contain only the 1,159 train IDs;
  `eval.jsonl` contains 413 validation/test IDs, with zero record-ID overlap.
  The leakage audit passes.
- [x] Normalized exact-question duplicate groups are zero after 70 near-
  duplicate removals.
- [x] Substantial whole-answer evidence-span copying is 402/1,572 (25.57%),
  below the provisional 35% ceiling and materially below the earlier roughly
  one-half observation.
- [x] All accepted records pass the current deterministic answer-completeness,
  claim-support, modality, and incomplete-evidence-fragment checks that can be
  replayed from canonical evidence without the removed source passage.
- [x] A deterministic navigation-trivia scan found no page-number, contents,
  section-location, or “where can this be found” questions.

Remaining defects demonstrated by this run:

- [ ] Question wording is still highly templated: 351/1,572 (22.33%) begin
  `According to`, 741/1,572 (47.14%) begin `As a`/`As an`, and 124/1,572
  (7.89%) begin `As per`. The largest exact four-word opener, `according to the
  manual`, is 237/1,572 (15.08%). Passing the old 15% planning target did not
  produce natural portfolio diversity; the manifest is correctly `partial`.
- [ ] Intent concentration remains excessive: `procedure` is 780/1,572
  (49.62%) and `direct_fact` is 354/1,572 (22.52%). The current 30% per-type
  cap and deterministic intent planner were not active in this run.
- [ ] Persona framing is often cosmetic. Role-prefixed questions are
  765/1,572 (48.66%), while procurement officer, tendering officer, and contract
  manager account for 895/1,572 (56.93%). A natural standalone question should
  not acquire a role preamble merely to vary its surface form.
- [ ] Generation lost 277/2,938 requests (9.43%) after retry: 192 list
  containers serialized as strings, 56 invalid enum/field shapes, 14 missing
  or wrong tool calls, and 15 other validation failures. These are exactly the
  old `CandidateBatch` failure modes; successful API transport is not schema
  success.
- [ ] The judge lost 10/2,436 requests (0.41%) permanently to its 1,024-token
  completion ceiling and logged 163 retry-time incomplete-output events. Those
  ten missing judge responses keep terminal lineage incomplete.
- [ ] Judge discrimination needs calibration: 1,290/1,572 accepted rows score
  5 and 282 score 4, while free-text issue values include inconsistent no-issue
  strings. A high judge score is not a substitute for stratified human review.
- [ ] The 24-record deterministic inspection sample was mostly grounded and
  operationally useful, with concise exact answers where precision matters.
  It also exposed overlong role-prefixed questions, artificial “procedure” and
  QA-CoT framing for lookup/composition tasks, clunky source-like answers, and
  at least one threshold wording risk (`above Rupees two lakh` summarized as a
  minimum of `Rupees two lakh`). This sample is diagnostic, not the required
  100-record human review.
- [ ] The release validator correctly fails the run for `partial` status,
  incomplete request lineage, portfolio-quality failure, and absent human
  review. Cross-document and drafting outputs are zero because those stages
  were skipped, so this run provides no evidence about their quality.

Decision:

- Keep this output as a pre-fix baseline; do not merge it with a post-commit
  evaluation or claim it validates commit `c3246913`.
- Do not rerun at full scale yet. First add the researched endpoint structure
  probe, then run a bounded post-commit pilot that exercises `qa_blueprints` and
  `GroundedCandidateDraft`. Compare identical metrics and manually review both
  accepted and rejected strata before approving a full regeneration.

## Structured-output endpoint probe research refresh (2026-08-02)

Status: research and local/reference audit completed before implementation.
Implementation and offline verification are complete; live user-controlled
generation and judge probes remain pending.

Verified current findings:

- [Curator structured output](https://docs.bespokelabs.ai/bespoke-curator/getting-started/structured-output)
  treats a Pydantic response model as the parsing contract. The local Curator
  processor deliberately trusts every explicitly configured non-`auto` mode,
  so configuration acceptance is not an endpoint capability test.
- [Curator recovery and caching](https://docs.bespokelabs.ai/bespoke-curator/getting-started/automatic-recovery-and-caching)
  fingerprints input, prompt, response format, model, batching, and generation
  parameters. It does not make a separate deployment transport probe a release
  gate; the procurement pipeline must persist that evidence itself.
- [Current vLLM tool-calling documentation](https://docs.vllm.ai/en/stable/features/tool_calling/)
  states that `auto` tool calling requires an enabled model-specific parser and
  that argument schema constraints under `tool_choice=auto` require strict
  tools/structural tags. It also warns that a validly parseable tool call does
  not guarantee a high-quality answer. Current generic vLLM support for named
  or required tools does not override this repository's direct evidence that
  the configured Nemotron deployment works only with its strict-auto path.
- [Instructor mode guidance](https://python.useinstructor.com/modes-comparison/)
  distinguishes native `JSON_SCHEMA`, tool-based `TOOLS`, and prompt-extracted
  `MD_JSON`, explicitly says LiteLLM support depends on the provider, and
  recommends testing the actual provider/model/mode combination.
- The reference project has no real per-endpoint schema probe. It only patches
  Curator's hosted-vLLM name-based capability check and verifies that the patch
  returns true. Copying that behavior would repeat the defect: a boolean local
  override cannot prove the server returns the configured nested structure.
- Run 003 confirms a scalar or `/models` health check is insufficient. The
  endpoint was reachable, yet nested list fields, enum assignments, and exact
  tool invocation still failed in 277 generation requests.

Proposed design before implementation:

- Probe generation and judge roles independently through the exact Curator/
  Instructor transport used in production, including `tools_auto` and schema
  dereferencing behavior. Do not implement a second approximate raw-HTTP
  request builder.
- Use a small nested probe schema containing an enum, a list of objects, and a
  nested list, then verify exact sentinel values after Pydantic parsing. A
  simple scalar JSON object would not cover the observed failures.
- Persist one atomic, secret-free result under `.curator_working` keyed by
  served model, deployment identity or endpoint, role, configured structured-
  output mode, schema-dereference setting, probe schema/prompt version, and
  relevant generation parameters. Never include the API key or authorization
  headers.
- Require a matching successful probe before a new deployment fingerprint can
  start a non-pilot run. Permit an explicit standalone probe command and a
  bounded pilot path; do not silently probe thousands of requests or accept a
  stale result from another endpoint/mode.
- Record timestamp, latency, parsed sentinel checks, configured mode, and safe
  failure class. Do not persist raw provider responses because they may contain
  unexpected text; exact diagnostic errors remain local logs.
- Test fingerprint invalidation, secret redaction, nested-shape rejection,
  wrong/missing tool calls, stale/missing result behavior, role independence,
  and successful gating without contacting a live model. The user performs the
  live endpoint probes.

Rejected alternatives:

- Infer support from model family, `/models`, HTTP 200, or LiteLLM's static
  registry: none exercises the configured response transport and nested schema.
- Treat `tool_choice=required` as universally safer: current vLLM may support
  it, but the exact Nemotron deployment has already returned no usable tool
  call under required/named forcing.
- Probe only `CandidateBatch`: rejected because that obsolete schema would
  preserve the old failure surface. The generic nested contract should detect
  container/tool transport defects, while bounded stage pilots separately test
  the real blueprint, final-record, and judge schemas.

Implementation record:

- [x] Added `structure_probe.py` with a nested enum/object/list Pydantic
  contract and exact sentinel checks. It runs through the same Curator/
  Instructor configuration used by production rather than a parallel HTTP
  approximation.
- [x] Added `probe_structure.py --role generation --role judge`; omitting
  `--role` probes both configured roles. Results contain only safe identity,
  boolean checks, latency, timestamp, and failure class. Raw output, API keys,
  authorization headers, and exception text are not persisted.
- [x] Fingerprint role, profile, served model, base URL, deployment identity,
  structured-output mode, schema dereferencing, generation parameters, probe
  contract, and Curator/Instructor/LiteLLM versions. Reject credential-bearing
  URLs and secret-named generation fields before persistence.
- [x] Execute every live probe in a temporary working directory so an old
  Curator response cache cannot masquerade as a fresh endpoint check. Persist
  the final gate result atomically under `.curator_working/structure_probes/`.
- [x] Require successful current generation and judge fingerprints before an
  unbounded run. A bounded `--limit` pilot bypasses the gate for diagnosis;
  `--skip-judge` requires only the generation result.
- [x] Added offline tests for exact sentinels, role gating, endpoint/mode
  invalidation, failed-result rejection, and URL/parameter secret rejection.
  Full local verification passes: Ruff clean, `git diff --check` clean, and
  115 procurement tests passing.
- [ ] User must run the two live probes and inspect their secret-free results.
  Passing the generic probe authorizes a bounded post-commit stage pilot, not a
  full quality claim about blueprint, final-record, or judge behavior.

## Request-local hosted-vLLM credential propagation (2026-08-02)

Status: implemented and locally validated. Both first live structure probes
failed with `AuthenticationError` before structured parsing; rerunning the live
probe is the remaining deployment check.

Verified findings:

- The configured `NEMOTRON_API_KEY` and `GEMMA_API_KEY` are present after
  `settings.py` loads `.env`; only presence and length were inspected, never
  values. `OPENAI_API_KEY` and `LITELLM_API_KEY` are absent.
- Local Curator 0.1.29 stores `api_key` on
  `OnlineRequestProcessorConfig`, but
  `LiteLLMOnlineRequestProcessor.test_call()` passes only model, messages,
  `api_base`, and generation parameters to `litellm.completion()`.
- The same processor's `create_api_specific_request_online()` adds `api_base`
  but not `api_key`; all three completion branches in `call_single_request()`
  therefore depend on process-global provider credentials unless the key is
  added request-locally.
- [LiteLLM's official documentation](https://docs.litellm.ai/) and upstream
  guidance for OpenAI-compatible providers support passing `api_base` and
  `api_key` on the individual completion call. Request-local values are needed
  when one process uses independent generation and judge endpoints.
- The reference repository independently found the same Curator defect, but
  applies runtime monkey patches to private methods. It also removes the key
  from persisted `raw_request`. This confirms the diagnosis, not the preferred
  implementation for this editable Curator source tree.
- Run 003's successful calls do not disprove the bug: they relied on an
  implicit global credential state in that process. A clean process with only
  role-specific keys reproduces immediate authentication failure for both
  endpoints.

Decision before implementation:

- Edit the local Curator LiteLLM processor normally; do not copy the reference
  runtime monkey-patch framework.
- Pass `self.config.api_key` explicitly to the synchronous rate-limit test call.
- In `call_single_request()`, create a per-request shallow copy of
  `request.api_specific_request`, add the configured key only to that transient
  copy, and use it for unstructured, Instructor, and strict-auto completion
  paths.
- Continue persisting the original credential-free
  `request.api_specific_request` as `GenericResponse.raw_request`; never mutate
  the persisted request mapping or write the key to response JSONL.
- Add unit tests proving rate-limit and async completion calls receive the
  correct role key, the source request remains unchanged, and returned raw
  request data contains no key. Run Curator unit tests plus the complete
  procurement suite before asking the user to repeat the probes.

Rejected alternatives:

- Export one global `OPENAI_API_KEY` or `HOSTED_VLLM_API_KEY`: rejected because
  generation and judge are independent endpoints and construction/concurrency
  order would decide which credential leaks across roles.
- Disable the rate-limit test only: rejected because actual completion requests
  omit the key for the same reason.
- Put `api_key` into `create_api_specific_request_online()`: rejected because
  that mapping is persisted as `raw_request` unless every downstream writer is
  trusted to redact it.

Implementation record:

- Updated Curator's LiteLLM processor to pass `config.api_key` explicitly in
  the synchronous endpoint/rate-limit test call.
- Actual completion calls now add the key to a shallow request-local copy for
  unstructured, strict-auto tool, and Instructor structured-output paths. The
  source request and `GenericResponse.raw_request` remain credential-free.
- Added regression tests covering all four paths and asserting the key is
  neither injected into the source mapping nor retained in the response.
- Validation passed: Ruff on the changed processor/test; 22 focused LiteLLM
  and online retry tests; and all 115 procurement pipeline tests. The warnings
  are pre-existing Pydantic/PyArrow deprecations.
- Remaining action: source `.env` in a fresh shell and rerun
  `pipelines/nrl_procurement/probe_structure.py`. Only successful generation
  and judge results authorize a full run; an authentication fix alone does not
  establish structured-output compliance.
- Post-fix live attempt: the generation request no longer failed immediately
  with `AuthenticationError`, confirming that authentication reached the
  endpoint, but it produced no response for more than three minutes and was
  stopped manually. The judge probe was therefore not reached. Deployment
  validation remains pending; rerun when the generation service is responsive.

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

- [x] Keep model identities, endpoints, credentials, generation parameters,
  structured-output modes, concurrency, retry counts, and rate limits in
  `.env`/`config.yaml`. CLI model/profile switches select configuration; they
  must not require code changes or expose API keys in manifests or logs.
- [x] Preserve independent generation and judge selection. If a shorthand such
  as `--model nemotron120b` is supported, define whether it selects generation
  only or a named paired profile; never silently use the same endpoint as its
  own judge when independent judging is required.
- [x] Define `--max-passes` unambiguously for the current cross-document
  iterative family: `0` means no numeric pass cap; a positive integer caps the
  total number of passes; omission follows configuration and currently means
  one pass. It stops on per-parent consecutive zero accepted novelty, not when
  one corpus-wide pass merely produces fewer records than requested.
- [x] Make cross-document saturation termination safe and auditable: persist per-stage/pass
  state; measure novel accepted records after deterministic validation,
  deduplication, and independent judging; require the configured consecutive
  no-progress condition; stop when the eligible source/planning space
  is exhausted; and record the exact stopping reason and pass metrics in the
  manifest. Resume must continue from persisted saturation state rather than
  reset the evidence window. Contract-v2 state binds the exact parent population,
  stores per-parent outcomes, rejects divergent cache replay, distinguishes
  hard-limit/incomplete/converged endings, and atomically replaces checkpoints.
- [ ] Generalize those guarantees to any future iterative single-document,
  drafting, temporal, graph/path, or dialogue family; they remain one-shot in
  this Curator pipeline today.
- [x] Generate a safe dynamic run ID when `--run-id` is omitted, while allowing
  an explicit run ID for reproducible pilots and resumptions.
- [x] Preserve local-only operation: Curator Viewer and telemetry remain off,
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

### Same-run resumability research record (2026-07-30)

Status: research and local source audit complete; implementation follows this
record.

User requirement:

- Reusing the same `--run-id` must continue an interrupted run.
- Changing a generation or judge model, endpoint, decoding configuration,
  schema, prompt, validator, or other material input must remain safe under the
  same run ID. Unaffected completed work should be reused, while the changed
  stage and all data-dependent downstream work must be recomputed.

Verified findings:

- Curator's local `LLM._hash_fingerprint` includes the dataset fingerprint,
  prompt-function source hash, model name, response schema, batch mode, and
  generation parameters. Its request processor reconstructs parsed datasets
  from persisted responses and sends only unfinished requests when a matching
  cache is resumed.
- Curator's fingerprint does not include the endpoint URL, structured-output
  mode, retry/concurrency/rate-limit backend settings, or validator helpers
  called indirectly by `parse`. Reopening one stage directory without another
  identity layer could therefore reuse an old response after an endpoint
  change when the served model name is unchanged.
- Snakemake recomputes affected DAG jobs when an input, output, or relevant
  workflow definition changes. Its between-workflow cache hashes steps,
  parameters, software stacks, and transitive raw inputs using a Merkle-tree
  design.
- Nextflow resumes an interrupted workflow from cached tasks. These systems
  distinguish immutable cached results from the mutable logical run attempt.

Design decision:

- Keep the logical run at `outputs/<run-id>` and allow repeated invocation with
  the same explicit run ID.
- Add a secret-free outer fingerprint directory at
  `.curator_working/<run-id>/<stage>/<stage-fingerprint>`. The fingerprint
  covers the stage name, role-specific semantic deployment identity,
  structured output and runtime settings, full non-secret pipeline
  configuration, relevant pipeline source hashes, and a versioned fingerprint
  contract. Curator keeps its existing dataset/prompt/schema/request
  fingerprint beneath that directory.
- Distinguish semantic model identity from transport location. Each profile may
  declare a stable, non-secret deployment/cache identity environment variable
  representing weights/quantization/tokenizer/chat template/tool parser and
  serving semantics. When supplied, changing only a port-forward URL does not
  invalidate the cache. When it is absent, the endpoint remains part of the
  fingerprint and an endpoint change fails safe by creating a new cache.
- Do not include API-key values. Record only the configured credential
  variable name.
- Re-execute inexpensive deterministic orchestration on resume. Exact unchanged
  LLM stages load Curator's cached dataset; changed models/endpoints receive a
  different outer directory; changed upstream datasets receive a different
  Curator inner fingerprint, automatically invalidating downstream calls.
- Persist successful stage outputs as immutable logical checkpoints with their
  producer fingerprint and model lineage. A later invocation may reuse a
  completed checkpoint even when the currently selected model changes; the new
  model begins at the first incomplete stage. Downstream fingerprints include
  the reused checkpoint content, so mixed stage provenance remains explicit
  rather than accidental.
- Do not merge partial responses from two model identities inside one
  unfinished stage. Preserve the old partial cache for audit, but restart that
  incomplete stage under the new model. This avoids an unmarked mixed-model
  batch while retaining all prior files.
- Write a `run_state.json` attempt journal atomically. At invocation start,
  preserve the prior terminal manifest in append-only resume history and mark
  the logical run `running`. A stopped attempt cannot leave an old `complete`
  manifest authorizing stale exports.
- Never silently combine outputs from two fingerprints. Final canonical files
  are regenerated from the current attempt, and the terminal manifest records
  stage fingerprints and whether each stage used an existing cache directory.
- Treat a run-ID collision without recognized pipeline state as unsafe.
  Recognized legacy runs remain readable; continuation starts fingerprinted
  caches and records the migration.

Alternatives rejected:

- Blindly remove the existing nonempty-output guard: unsafe because Curator's
  inner hash omits endpoint/backend identity.
- Delete the old run before restarting: destroys audit history and does not
  satisfy continuation.
- Reuse successful responses across actual model changes: invalid because model
  identity is a material dependency. “Continue” means retain the logical run
  and unaffected stages, not mix old-model outputs into a changed stage.
- Treat identical served names on arbitrary machines as proof of identical
  deployments: rejected because weights, quantization, tokenizer, templates,
  parser flags, and server revisions may differ. Explicitly preserving the
  deployment/cache identity is the operator assertion for a tunnel-only move.
- Hash API keys: leaks a stable credential verifier into artifacts and makes
  routine key rotation invalidate scientific results. Endpoint/model/config
  identity is sufficient; secrets remain secret.

Primary and official sources:

- Local Curator implementation:
  `src/bespokelabs/curator/llm/llm.py`,
  `src/bespokelabs/curator/request_processor/base_request_processor.py`, and
  `src/bespokelabs/curator/request_processor/cache.py`.
- Snakemake,
  [workflow caching](https://snakemake.readthedocs.io/en/v9.13.4/executing/caching.html),
  [provenance](https://snakemake.readthedocs.io/en/latest/executing/provenance.html),
  and [DAG recomputation](https://snakemake.readthedocs.io/en/stable/tutorial/basics.html).
- Nextflow,
  [resume behavior](https://nextflow.io/docs/stable/guides/updating-spot-retries.html).
- W3C,
  [PROV-O](https://www.w3.org/TR/prov-o/).

Implementation acceptance criteria:

- An interrupted same-configuration invocation with the same run ID uses the
  same stage fingerprint and Curator cache.
- A generation-model/deployment change creates new generation stage
  fingerprints for incomplete stages. Completed logical stage checkpoints
  remain reusable and retain their original producer lineage. A transport-only
  endpoint change with an unchanged explicit deployment identity also reuses
  partial Curator caches. Changed generated datasets invalidate judge requests
  through Curator's inner dataset hash.
- A judge-only change reuses generation caches and creates new judge stage
  fingerprints for incomplete judge stages. Completed judge checkpoints remain
  reusable with their original judge-model lineage; downstream acceptance and
  exports consume the selected checkpoint explicitly.
- Prompt/schema/validator/config changes cannot reuse an incompatible stage
  cache.
- No fingerprint, state file, manifest, or log contains an API key.
- A running or failed resumed attempt cannot expose a stale terminal
  `complete` manifest.

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
- [x] Add a model-aware preflight that estimates the complete rendered prompt,
  response schema, and output reserve against the selected profile's actual
  server context. A fixed batch size is conservative but cannot prove every
  future source record fits.
  - Research update (2026-07-29):
    - Question: how can the preflight remain exact when generation and judge
      profiles may be switched independently, and which structured-output
      bytes are actually part of the rendered model prompt?
    - Verified fact: vLLM's official `TokenizeChatRequest` accepts chat
      messages, tools, `add_generation_prompt`, and `chat_template_kwargs`.
      Its serving implementation preprocesses the chat through the endpoint's
      active renderer and returns both the exact token count and
      `max_model_len`. This is a closer match to the eventual request than a
      locally guessed tokenizer or a model-name conditional.
    - Verified fact: Hugging Face documents that `apply_chat_template` owns
      model-specific control tokens and that tools are template inputs. A tool
      schema must therefore be supplied to tokenization for `tools_auto`
      profiles. Conversely, vLLM's tokenization request has no
      `response_format` field; a JSON-schema guided-decoding constraint must
      not be counted as literal prompt text unless a future transport actually
      injects it into messages or template inputs.
    - Pilot-012 evidence: all nine judge preflight rejections used
      `conservative_character_estimate`, with estimated prompts of
      7,111-8,851 tokens before the 1,024-token completion reservation and
      256-token margin. The current code never supplies a tokenizer and always
      appends serialized response-schema characters, so these quarantines
      cannot be treated as exact endpoint-capacity failures.
    - Alternatives considered: hardcode tokenizer paths per known model
      (rejected because profiles and served revisions are switchable); keep
      character-only estimates (retained only as an explicitly labeled,
      fail-closed fallback); query endpoint-native `/tokenize` (selected
      because it uses the active served tokenizer, chat template, template
      kwargs, and runtime model limit).
    - Recommended implementation: build tokenization inputs from the selected
      profile's actual structured-output mode; call its private vLLM
      `/tokenize` endpoint with the served model and template kwargs; include
      the strict Pydantic tool only for `tools_auto`; reserve the selected
      profile's completion ceiling and safety margin; audit the count method
      and server limit; and fall back conservatively when tokenization is
      unavailable. Never branch on Gemma, GLM, or Nemotron names.
    - Sources accessed 2026-07-29:
      - vLLM tokenization protocol:
        https://docs.vllm.ai/en/v0.15.0/api/vllm/entrypoints/serve/tokenize/protocol/
      - vLLM tokenization serving implementation:
        https://docs.vllm.ai/en/latest/api/vllm/entrypoints/serve/tokenize/serving/
      - Hugging Face tokenizer/chat-template contract:
        https://huggingface.co/docs/transformers/main_classes/tokenizer
    - Implemented 2026-07-29: judge preflight now calls the selected private
      vLLM endpoint's `/tokenize` route with its served model, chat-template
      kwargs, and exact auto-tool request where applicable. It reconciles the
      configured context with the server-reported `max_model_len`, audits the
      structured-output mode and measurement failure, and uses the smaller
      limit. JSON-schema decoding constraints are no longer charged as
      fabricated prompt text; tool schemas and the mandatory auto-tool system
      message are rendered exactly. A five-second first-failure circuit
      breaker prevents an unsupported tokenization route from stalling every
      row and preserves the labeled conservative fallback for that run.
    - Validation: 70 focused procurement and LiteLLM processor tests passed;
      Ruff and Python compilation passed. No model-backed generation or judge
      call was run.
- [x] Validate schema compliance, exact witness behavior, latency, and judge
  calibration with a user-run pilot before production-scale generation.
  - Pilot-014 completed 2026-07-29 with Nemotron generation and independent
    Gemma judging. Final manifest status was correctly `failed`; production
    scale remains unapproved.
  - Endpoint-native prompt measurement succeeded for all 87 judge calls with
    no fallback or prompt-budget rejection. Single-document prompts used
    1,778-4,115 tokens and at most 5,395 tokens including completion reserve
    and margin. Cross-document prompts used 3,595-5,790 tokens and at most
    7,070 total against the served 8,192-token limit.
  - Gemma returned schema-valid responses for 78/78 singular QA judge calls
    and 9/9 singular cross-document judge calls. Observed call latency was
    31-452 seconds (median 246, p95 380) for QA and 43-119 seconds
    (median 59, p95 119) for cross-document.
  - The judge accepted 70/78 QA records and 6/9 cross-document records.
    Manual review confirmed representative accepted answer/evidence witnesses
    were exact and material. Six QA rejections were taxonomy/persona
    disagreements, while two correctly identified dropped qualifications.
  - Calibration failure: all three rejected cross-document records reviewed
    were grounded in both source documents, but the judge returned an empty
    `unsupported_without_source_ids` prediction. The pipeline therefore
    rejected them through `source_ablation_passed=false` despite full-context
    support. This confirms the documented limitation that predicted ablation
    is not a substitute for executing A-only and B-only trials.
  - Nemotron transport compliance improved but was not perfect: one of 50
    single-generation requests, three of 50 cross-generation requests, one of
    two drafting requests, and one of 25 path-answer requests failed
    permanently after malformed tool arguments. The dominant yield loss was
    deterministic grounding rejection (non-verbatim evidence, unsupported
    numbers/dates, claim/evidence mismatch), not transport failure.
  - Final output contained 76 records, but request coverage was incomplete:
    123 single-document candidates became 78 judged and 70 accepted; 36
    cross-document candidates became 9 judged and 6 accepted; drafting was
    0/2 accepted; path QA remained 0 training records pending real claim
    coverage and independent adjudication.

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
- [x] Add a mandatory per-endpoint structure probe before a newly configured
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

Status: research and production data-artifact implementation complete on
2026-07-30. The effectiveness of any schedule remains an empirical hypothesis
until the user-controlled pilot and controlled training experiment are run.

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

- [x] Verify and complete amendment/currentness metadata against official
  Department of Expenditure and NRL sources; record a verification cutoff.
- [x] Add typed temporal-pair and sampling-schedule configuration with strict
  validation and secret-free fingerprints.
- [x] Build bounded one-to-many temporal alignments from accepted propositions
  and section windows; write candidate and rejected alignment audits.
- [x] Add a source-grounded change extractor that emits historical/target
  proposition IDs, exact evidence, change type, and explicit lineage basis.
- [x] Generate separately validated historical, transition, and target QA/CoT
  exports with visible time and authority scope.
- [x] Export a trainer curriculum manifest; do not implement provider batching
  or model training inside Curator.
- [x] Add leakage-safe temporal splits that keep a change lineage together
  while holding out separate rule families for evaluation.
- [x] Add deterministic and judge checks for identical states, unrelated
  subjects, reversed dates, unsupported currentness/supersession, missing
  temporal labels, changed numbers/modalities, and NRL/Government leakage.
- [ ] Run a user-controlled data pilot, followed by a separate controlled
  training experiment before selecting or claiming benefits from a schedule.

Implementation record (2026-07-30):

- `data/source/manuals.yaml` is the evolving canonical source registry.
  Temporal discovery uses only explicit `temporal_predecessors`; a newly added
  document cannot silently acquire inferred amendment or supersession edges.
- Registered NRL PDFs are the authority originals. Runtime corpus loading uses
  their corresponding Chandra Markdown under `data/interim/ocr`, retaining
  separate `source_sha256` (PDF) and `content_sha256` (OCR) fingerprints.
- Government edition identity was checked against the Department of
  Expenditure publication index at cutoff 2026-07-30. NRL Rev1 PDFs are pinned
  by source hash; absent a public later-lineage index, no post-Rev1 currentness
  assertion is made.
- Temporal candidates require deterministic authority/date/signature checks
  and an independent configured judge verdict before any change or QA export.
  Missing judge responses are terminal rejections, not implicit acceptance.
- Six time-labelled files are emitted: historical QA/QA-CoT, transition
  QA/QA-CoT, and target QA/QA-CoT, plus candidate/rejection/change audits and a
  trainer-only curriculum manifest.
- The schedule is explicitly labelled an unvalidated experiment
  configuration. Curator performs neither model training nor schedule-benefit
  claims.

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
- [x] Add missing-hop and false-premise unanswerable contrasts.

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
- [x] Measure required-claim coverage for all three outputs.
- [x] Store the actual ablation outputs and decisions.
- [x] Reject answerable records when either single-source output fully covers
  the canonical answer.

Acceptance criteria:

- Full context covers all required answer claims.
- A-only misses at least one required source-B claim.
- B-only misses at least one required source-A claim.
- The judge reviews actual ablated outputs rather than predicting the result.

Implementation addendum (2026-07-30):

- Added deterministic three-variant adjudication keyed by stable record and
  proposition IDs. It requires a valid full/A-only/B-only set, complete
  required-claim coverage in the full trial, and rejects a single-source trial
  that still covers the complete answer.
- Persisted `path_ablation_adjudications.jsonl` and separate pass/fail counts.
  Passing path answers remain pending an independent judge over the actual
  trials; they are not silently added to training exports.
- Offline application to Pilot 014 accepts 4/7 path answers and rejects 3/7
  because their full-context trial abstained or missed required claims.

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

- [x] Derive stable required-claim coverage from the canonical verified answer.
- [x] Independently adjudicate the three actual outputs and enforce the full/A/B
  coverage decision.
- [x] Store the final coverage decision and promote only passing path records
  into canonical cross-document exports.

Implementation completion (2026-07-30):

- The independent judge receives the persisted full/A-only/B-only outputs,
  canonical claims, and grounded propositions. It is deliberately not shown
  the deterministic pass/fail decision.
- Promotion requires complete deterministic adjudication, identity-preserving
  independent judgment, four required judge booleans, and the configured score
  threshold. Missing, invalid, or rejected trials remain audit-only.
- Promoted path answers use the same canonical contract and leakage-safe export
  path as the original cross-document generator.

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

- [x] Give every material answer claim a stable ID.
- [x] Give every evidence item a stable ID.
- [x] Add `input_claim_ids`, `output_claim_id`, and `evidence_refs` to each
  reasoning step.
- [x] Verify that the graph is connected and acyclic.
- [x] Verify that the final answer is covered by terminal claims.
- [x] Derive QA and QA-with-rationale views from one canonical record.

#### Release-gate research record (2026-07-30)

Status: research and code audit complete; implementation follows this record.

Verified findings:

- W3C PROV models source and derived objects as entities connected through
  generation, usage, and derivation activities. A replayable record therefore
  needs stable evidence/claim identities and explicit step inputs and outputs;
  a free-form rationale list is insufficient provenance.
- DiRe and counterfactual multi-hop QA research show that a correct final
  answer can come from one supporting-fact subset. Graph validity is necessary
  for auditability but is not evidence of multi-hop necessity; the actual
  full/A-only/B-only trials and independent review remain separate gates.
- Published CoT-faithfulness work finds that stated intermediate reasoning is
  not reliably used to produce the answer. Deterministic validation can prove
  grounding, linkage, and structural necessity, but must not label prose
  rationales as causally faithful without intervention evidence.
- Group-aware evaluation requires dependency groups to remain disjoint.
  Source hashes, manuals, sections, chunks, path families, canonical records,
  and normalized/paraphrase question families are all leakage edges, not
  independent row-level checks.
- NIST's TEVV guidance calls for documenting test sets, metrics, and evaluation
  tooling for repeatability. A generated `eval.jsonl` is not a frozen,
  independent human-reviewed benchmark.

Implementation decision:

- Create one canonical reasoning graph per accepted record. Stable IDs are
  content-addressed; every non-source claim has exactly one producing step;
  every step declares an operation, input claims, output claim, and evidence
  references. Reject unknown references, cycles, disconnected nodes, unused
  source claims, and final answers without terminal-claim coverage.
- Judge the persisted full/A-only/B-only outputs as one immutable bundle using
  the independent judge profile. Promotion requires deterministic ablation and
  this actual-output judgment; invalid/missing trials never count as evidence
  of necessity.
- Export QA and QA-with-rationale from the same canonical record without
  changing answer or provenance. Rationale views expose the auditable graph
  steps; QA views omit their prose only.
- Emit a leakage audit containing collisions and component membership at every
  requested level. Treat any train/evaluation collision as a release failure.
- Extend the terminal manifest with code/config/source fingerprints, stage
  timings, retry/failure distributions, cache reuse, rejection distributions,
  task/source coverage, export cardinalities, and human-review status.
- Provide immutable review/evaluation scaffolding, but do not fabricate human
  labels. The 100-accepted-record review and frozen-gold gates remain open
  until reviewers supply them.

Primary sources:

- W3C, [PROV-O](https://www.w3.org/TR/prov-o/) and
  [PROV constraints](https://www.w3.org/TR/prov-constraints/).
- Trivedi et al.,
  [Is Multihop QA in DiRe Condition?](https://aclanthology.org/2020.emnlp-main.712/).
- Guo et al.,
  [Counterfactual Multihop QA](https://aclanthology.org/2023.acl-long.231/).
- Paul et al.,
  [Making Reasoning Matter](https://aclanthology.org/2024.findings-emnlp.882/).
- scikit-learn,
  [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
  and [data-leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
- NIST AI RMF,
  [Measure playbook](https://airc.nist.gov/airmf-resources/playbook/measure/).

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
- [x] Write reasoning-path candidates.
- [x] Write raw generation candidates.
- [x] Write deterministic rejections.
- [x] Write source-ablation rejections.
- [x] Write judge rejections.
- [ ] Write duplicate and best-of-N rejections.
- [x] Write accepted canonical audit records.
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
- [x] Prevent question paraphrases from crossing splits.
- [x] Generate a leakage audit by source hash, manual, section, chunk, path,
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
- [x] Keep the specialized cross-document and source-ablation judge.
- [x] Prefer a judge model distinct from the generator.
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
- [ ] Sample at least 100 accepted records for human review. A deterministic
  review-template command is implemented; completion requires a run with at
  least 100 accepted records and actual reviewer labels.
- [ ] Review rejected samples to find overly strict gates. Rejected-record
  sampling is implemented; completion requires actual reviewer labels.
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

- [x] Plan `qa` and `qa_cot` separately. Do not depend on the generator to
  choose the run's task distribution.
- [ ] Assign `qa_cot` only to evidence windows with at least two connected
  material claims or operations. Never force a decorative rationale onto a
  direct lookup.
- [x] Validate every rationale step for an explicit operation, grounded inputs,
  supported output, connectivity to adjacent steps, and contribution to the
  final answer.
- [x] Keep single-document QA/CoT exports distinct from cross-document QA/CoT
  exports. A combined export, if desired, must have a different explicit name.
- [ ] Preserve abstention records separately in metrics and optionally in a
  dedicated export so their training weight can be chosen from downstream
  validation rather than accidental corpus order.

#### Grounded drafting

- [x] Normalize only lossless surface variants before validation: line endings,
  surrounding whitespace, and explicit `<br>` tags to newline characters.
  Record every repair. Do not infer headings, facts, clauses, or missing text.
- [x] Reject remaining HTML markup in final plain-text drafting records.
- [x] Remove ordered-list markers before numeric fact comparison and ensure
  hyphenated identifiers are compared as identifiers, not partial numbers.
  Continue rejecting genuine unsupported percentages, amounts, dates,
  durations, emails, and identifiers.
- [ ] Extract material drafting claims/fields (organization, authority,
  contacts, references, thresholds, remedies, conditions, exceptions) and
  require support from tender facts or manual evidence. A model-declared
  `tender_facts_used` list is not sufficient.
- [x] Explicitly reject unsupported labeled authority/organization fields such
  as the observed `NRL Procurement Division`.
- [x] Send deterministically valid drafts to the independent judge; do not let a
  formatting false positive prevent semantic judgment. Do not weaken
  deterministic grounding to increase acceptance.

#### Completeness, recovery, and judging

- [x] Track expected request IDs through generation, parse, deterministic
  validation, judge, and export. No requested row may silently disappear.
- [ ] Quarantine malformed/missing outputs with exact failure class and raw
  cache lineage.
- [ ] Add bounded rescue only for explicitly recoverable failures such as
  schema truncation or lossless formatting. Rescue uses its own cache stage and
  attempt budget; it must not retry deterministic unsupported content
  indefinitely.
- [x] Enforce generator/judge independence for production profiles and record
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

- [x] Write the manifest last, atomically, with a terminal status of
  `complete`, `partial`, or `failed`.
- [ ] Include code revision, source/OCR fingerprints, non-secret model/config
  fingerprints, stage timing, expected/materialized/accepted/rejected counts,
  rejection reasons, repairs, retries, cache reuse, and coverage distributions.
- [x] Never label a partial run complete. Preserve audit and rejection files on
  failure, but withhold publishable final files when required gates fail.
- [x] Keep amendment/edition/authority-connected documents in atomic split
  groups. Report absent splits as a pilot-size limitation; require minimum
  independent group counts before producing evaluation claims.
- [ ] Add a frozen, human-reviewed evaluation set outside generated training
  data. Generated `eval.jsonl` is useful for pipeline testing but is not an
  independent gold benchmark.

### Acceptance criteria before the next pilot

- [x] Corpus preflight passes and writes an auditable quality report.
- [ ] A fixed-seed pilot covers multiple manuals, both authority classes,
  multiple document families/page bands, single QA, genuine single QA-CoT,
  cross-document QA, and cross-document QA-CoT.
- [ ] Every planned request has a terminal lineage state.
- [x] Final task-specific exports are non-overlapping and their counts reconcile
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
- [x] Replace forced named tools with validated auto-tool transport based on
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
- [x] Confirmed path-answer yield in user-run Pilot 012: all 26 requests
  reached successful terminal responses and 9 answers passed deterministic
  validation.

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

- [x] Confirmed Nemotron `tools_auto` path-answer and ablation yield in
  user-run Pilot 012: 26/26 path-answer and 27/27 ablation requests reached
  successful terminal responses after bounded retries; 9 answers and 24
  ablation trials passed deterministic validation.
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

Ledger reconciliation:

- Marked previously stale boxes complete only where committed code and
  Pilot 011 artifacts provide direct evidence: missing-hop/false-premise
  contrasts, stored ablation trials, disjoint task exports, lossless drafting
  normalization, HTML/numeric/authority drafting gates, independent drafting
  judging, independent generator/judge enforcement, atomic terminal manifests,
  partial-run audit preservation, connected-component splitting, corpus
  preflight reporting, reconciled exports, and validated auto-tool transport.
- Empirical gates remain open when they require Pilot 012, human review,
  multiple reviewers, an upstream action, or a capability not present in the
  repository. They must not be closed from unit tests alone.

## Pilot 012 completed-run audit (2026-07-29)

Status: inspected after the user-run pilot completed; manifest status is
`partial`.

Verified improvements:

- `tools_auto` completed all 50 proposition, 36 path-question, 26 path-answer,
  and 27 ablation requests after bounded retries. It eliminated Pilot 011's
  permanent path-stage serialization failure.
- Proposition/path outputs include 89 accepted propositions, 36 accepted
  reasoning paths, 26 accepted questions, 9 accepted answers, and 24 valid
  ablation trials.
- Single-document judging completed 81/81 submitted requests without the
  Pilot 011 context-window API error.
- Drafting aggregation remediation worked: one draft passed deterministic
  validation and independent judging at 5/5. The other failed only its
  block-local instruction-quotation attribution.
- The component allocator produced 57 train, 8 validation, and 8 test records,
  eliminating Pilot 011's empty validation split.
- All 73 canonical records have unique IDs, non-empty claims/evidence/citations,
  `citations` last, passing deterministic checks, accepted 5/5 judges, and
  correct QA versus QA-CoT reasoning contracts.

Remaining defects:

- Ordinary generation had 4 permanent failures out of 50 and cross generation
  had 3 out of 50. Auto-tool retries still encounter absent calls, empty
  objects, stringified arrays, invalid enum values, and missing fields.
- The new judge preflight used the conservative character fallback and
  quarantined 4 single-document and all 5 deterministically valid
  cross-document candidates. As a result, Pilot 012 exported zero
  cross-document records. Exact endpoint-compatible token measurement or a
  safer adaptive budget is required before the next pilot.
- Curator's generated `failed_requests.jsonl` files contain 7 ordinary and 21
  cross-generation rows, while terminal response files contain only 4 and 3
  permanent failures respectively. The files currently include requests that
  failed an earlier attempt but later recovered, so their name/count
  overstates permanent failure.
- Manifest completion currently depends on every planned request producing an
  accepted record. Deterministic or judge rejection therefore makes a run
  `partial` even when every request has an auditable terminal state. Terminal
  completeness and quality acceptance should be reported separately.

Next research gates:

- [x] Research and implement vLLM auto-tool `strict: true`; retain the
  exactly-one-call validator until forced named/required tool calling becomes
  available. Model-level yield remains a next-pilot validation item.
- [x] Research exact Gemma prompt counting and adaptive completion reservation;
  replace the over-conservative judge fallback without allowing API overflow.
- [x] Research Curator retry/failure-file semantics and persist only permanent
  failures in `failed_requests.jsonl`, while retaining attempt history
  separately.
- [x] Separate manifest terminal-request completeness from accepted-record
  yield and required-task coverage.

### Strict auto-tool schema research (2026-07-29)

Status: research complete before implementation.

Question:

- Can `tools_auto` enable vLLM strict schema guidance safely, and how should
  Pydantic response models be converted to the required tool schema?

Official and upstream findings:

- Current vLLM documentation states that `tool_choice: "auto"` is
  schema-constrained only when at least one tool declares `strict: true`.
  Strict schemas require `additionalProperties: false` on every object, every
  property in `required`, and nullable types for optional values:
  https://docs.vllm.ai/en/latest/features/tool_calling/
- vLLM documents that auto choice still permits no call. Strict mode constrains
  arguments when a call is selected; it does not replace the existing
  exactly-one-call application check.
- The official OpenAI Python SDK exposes `pydantic_function_tool()` and its
  source recursively converts Pydantic schemas to the strict contract,
  including nested definitions, arrays, unions, object closure, and required
  properties:
  https://github.com/openai/openai-python/blob/main/src/openai/lib/_tools.py
  https://github.com/openai/openai-python/blob/main/src/openai/lib/_pydantic.py
- NVIDIA's Nemotron 3 Super model card continues to prescribe vLLM
  `--enable-auto-tool-choice --tool-call-parser qwen3_coder`, temperature 1.0,
  and top-p 0.95:
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
- Upstream vLLM reports show that MTP and reasoning/tool parsing can still
  affect tool-call extraction. Strict arguments do not prove that forced calls
  or MTP are reliable:
  https://github.com/vllm-project/vllm/issues/38106

Local and reference-code evidence:

- Pilot 012 completed every path-stage request after retry, but initial or
  permanent attempts included stringified arrays, invalid enum values, empty
  objects, and absent calls.
- Curator's current `_auto_tool_request()` passes raw
  `model_json_schema()` and does not set strict mode.
- `PropositionBatch` and `CandidateBatch` contain nested objects and defaulted
  fields. Their raw Pydantic schemas do not mark every property required and
  do not close every object with `additionalProperties: false`; merely adding
  the boolean flag would violate the documented strict-schema contract.
- Installed OpenAI SDK 2.30.0 exposes the public
  `openai.pydantic_function_tool` converter. Curator already has this SDK in
  its LiteLLM runtime dependency graph.
- The reference project contains no stricter auto-tool implementation; its
  bundled upstream Curator code does not solve this capability.

Alternatives:

- Add only `strict: true`: rejected because the raw schema is not strict.
- Maintain a Curator-specific recursive schema converter: rejected because it
  duplicates protocol logic already maintained by the official SDK.
- Switch immediately to named/required tools: deferred because the current
  private server returned HTTP 500 for both forced modes.
- Use `openai.pydantic_function_tool()`: selected because it produces the
  official strict function shape while preserving Pydantic as the final
  application validator.

Decision and validation:

- Build the single auto tool with the official SDK converter.
- Preserve `tool_choice: "auto"`, the exactly-one expected-name check, JSON
  argument parsing, and final Pydantic validation.
- Add unit assertions for `strict: true`, recursively closed objects, and
  required defaulted properties, plus the existing tool-name/argument failure
  tests.
- Treat improved model yield as provisional until the next user-run pilot; do
  not infer it from schema inspection or unit tests.

Implementation result:

- `_auto_tool_request()` now uses the official
  `openai.pydantic_function_tool()` converter, producing `strict: true` and a
  recursively strict schema while leaving `tool_choice: "auto"` unchanged.
- The parser still rejects zero calls, multiple calls, unexpected tool names,
  malformed JSON arguments, and arguments that fail the original Pydantic
  model.
- Focused validation passed: 67 tests, Ruff, and Python bytecode compilation.
- [ ] Run the next bounded Nemotron pilot and compare initial validation
  retries, permanent failures, and accepted yield with Pilot 012.

## Pilot 014 core-dataset release remediation (2026-07-30)

Status: implemented and locally verified; the next model-backed run remains
user-controlled.

- Permanent `failed_requests.jsonl` construction now uses the terminal
  `GenericResponse` state rather than application parse-row presence. Valid
  zero-row responses and recovered retries are not mislabeled as provider
  failures.
- Single- and cross-document generators materialize an explicit
  `empty_generation` terminal lineage row when a schema-valid batch contains
  no examples.
- The final manifest now reports `terminal_request_completeness` separately
  from `quality_acceptance`. Deterministic and judge rejections reduce yield
  without masquerading as missing requests.
- QA-with-CoT and cross-document QA-with-CoT reject exact repeated reasoning
  steps and cases where every step reuses the identical evidence set. Ordinary
  QA behavior is unchanged.
- Real source-ablation adjudication measures proposition-level material-claim
  coverage over full, A-only, and B-only outputs. Passing path records still
  require independent review of the actual trials before export.
- Local verification passed 83 focused tests and Ruff checks without model
  calls. A user-run full pipeline must confirm all four required task types,
  terminal completeness, model yield, and human-reviewed quality.

### Stale-checkbox reconciliation (2026-07-30)

The following previously unchecked requirements are now marked complete from
direct implementation and regression-test evidence rather than pilot yield:

- Configuration-backed generation/judge profiles, secret-free manifests, and
  enforced generator/judge independence.
- Safe generated or explicit run IDs, private-endpoint enforcement, disabled
  viewer/telemetry, project-local caches, and immutable run output layout.
- Deterministic separate planning for QA versus QA-with-rationale, including
  cross-document variants.
- Raw single/cross generation audits, deterministic rejection files,
  source-ablation rejection/adjudication files, judge rejection files, and
  accepted canonical audit records.
- Expected request lineage through generation, validation, judging, and
  export, including path-question, path-answer, ablation-trial, and
  ablation-judge terminal accounting.
- Content-addressed record/claim/evidence/step identities, canonical reasoning
  graphs, leakage audits, connected-component splitting, task-specific export
  reconciliation, and post-retry failure classification.

Items deliberately left open because their wording exceeds the current
implementation:

- Representative selection is deterministic and diversity-first, but is not
  the requested strict round-robin algorithm and does not yet guarantee every
  section/content stratum.
- Proposition, reasoning-path, and ablation artifacts exist, but a dedicated
  persisted source-bundle file and one complete forward-lineage table still do
  not.
- Duplicate removal is implemented, but best-of-N ranking and its rejection
  artifact are not.
- Connected split assignment and leakage auditing exist, but authored held-out
  manuals, all derived RAG/distractor variants, and multiple folds do not.
- Terminal-lineage code is complete, while the acceptance checkbox requiring
  every request to be terminal remains an empirical run gate.

### Same-run model/endpoint resumption implementation (2026-07-30)

Status: implemented and locally verified; a bounded user-run interruption and
resume remains the final operational check.

- [x] Added logical, completed-stage checkpoints whose reuse key covers the
  stage inputs, configuration, pipeline source, and output contract, but not
  the current generator or judge identity. A completed stage therefore remains
  reusable when either model is renamed or replaced, while retaining the
  original producer identity in checkpoint provenance.
- [x] Added model-aware outer Curator cache fingerprints for incomplete stages.
  Changing a model/deployment starts a new stage cache instead of silently
  combining partial responses from two deployments.
- [x] Separated stable deployment identity from transport address. When a
  profile's `*_DEPLOYMENT_ID` is set, changing only its forwarded URL/port
  retains the same incomplete-stage cache. Without that explicit identity,
  the endpoint remains part of the fingerprint and changes invalidate safely.
- [x] Applied the same rules independently to generation and judge stages.
- [x] Added atomic `run_state.json`, attempt history, prior-manifest snapshots,
  fail-closed running manifests, secret-free model provenance, and explicit
  `--refresh-stage` overrides that preserve replaced checkpoint history.
- [x] Added regression coverage for generator/judge swaps, transport-only
  changes, completed-checkpoint reuse, forced refresh history, and credential
  redaction. Focused verification passed 73 tests, Ruff, Python compilation,
  YAML parsing, and `git diff --check`.

### Nemotron strict-auto endpoint probe (2026-07-30)

- [x] Re-tested the live Nemotron endpoint after disabling the broken vLLM
  structural-tag enforcement path. With Curator's real `PropositionBatch`
  schema, `tool_choice: "auto"`, `strict: true`, and the production
  4,096-token output allowance, both bounded requests returned HTTP 200,
  exactly one tool call, and Pydantic-valid arguments.
- [x] Confirmed the active Curator `tools_auto` implementation sends
  `tool_choice: "auto"`.
- [ ] Treat `tool_choice: "required"` as unsupported on this deployment:
  bounded probes returned HTTP 200 but no usable tool call, sometimes with a
  false `finish_reason: "tool_calls"`. The procurement pipeline does not use
  this mode, but it must not be enabled without a separate server-side fix.

### Source-independent completed checkpoints and Nemotron `$ref` fix (2026-07-30)

Research basis:

- Nextflow-style resume semantics distinguish immutable completed task results
  from active work: downstream reuse follows input/task identity, while changed
  active work receives a new cache identity.
- W3C PROV models a generated artifact as an Entity attributed to the Activity
  and SoftwareAgent that produced it. A later software revision does not alter
  the already-generated entity; provenance must preserve the original producer.
  https://www.w3.org/TR/prov-o/
- vLLM documents that disabling `VLLM_ENFORCE_STRICT_TOOL_CALLING` prevents
  structural-tag constraints for auto tool calls. Arguments are then extracted
  from raw model output and parser/schema handling remains material.
  https://docs.vllm.ai/en/stable/features/tool_calling/

Implementation:

- [x] Removed the global pipeline source hash from completed logical-checkpoint
  identity. Completed artifacts now remain reusable across source revisions,
  and preserve their original producer source hash as provenance.
- [x] Retained source hashing in incomplete Curator stage-cache identity, so a
  restarted stage never mixes partial responses generated by different code.
- [x] Added backward-compatible discovery of completed `nrl-resume-v1`
  checkpoints by exact stage and logical-input hash. Existing pilot-015
  artifacts therefore survive the resume-contract upgrade.
- [x] Kept explicit `--refresh-stage` as the operator-controlled invalidation
  mechanism when a completed artifact must be regenerated after a semantic
  code change.
- [x] Added a profile capability, `dereference_tool_schema`, enabled only for
  Nemotron. The `tools_auto` request builder now deep-copies and fully inlines
  local non-recursive `$defs/$ref` schemas before transmission. Unsupported,
  missing, and recursive references fail rather than looping or degrading.
- [x] Confirmed the four production response schemas (`PropositionBatch`,
  `PathAnswerDraft`, `AblationTrialDraft`, and `CandidateBatch`) contain no
  `$defs` or `$ref` after transformation.
- [x] Fixed Curator's Arrow writer to normalize missing columns to null across
  heterogeneous valid rows, preventing empty-generation lineage rows from
  causing `KeyError: record_id`. Type conflicts remain hard failures.
- [x] Focused verification passed 91 tests, Ruff, Python compilation, and
  `git diff --check`.

## Capability research — pilot-020 release-audit code defects (2026-07-30)

Status: researched and approved for implementation; a bounded user-run smoke
test remains pending after these fixes.

Research questions:

- Why do 32 accepted-record evidence references fail
  `source_chunk[start_char:end_char] == quote`, and what is the minimal,
  precedent-consistent fix?
- Why does `leakage_audit()` report only two unique manuals when accepted
  records span roughly fifteen, and what is the correct fallback?
- Why does `pending_independent_judge` stay nonzero after every ablation
  candidate has a terminal judge decision, and why does
  `request_coverage.judged.missing_request_ids` list deterministically
  rejected requests as missing?

Verified local findings:

- `pipelines/nrl_procurement/propositions.py:67,85,131` already resolves
  proposition evidence against
  `row.get("source_passage", row["passage"])` — the raw registered chunk
  identified by `chunk_id` — and rejects `non_verbatim_evidence` /
  `ambiguous_evidence_occurrence` when the quote cannot be uniquely located
  there. `tests/nrl_procurement/test_pipeline.py::test_proposition_offsets_resolve_against_original_source_chunk`
  (line 940) locks this behavior in with an inserted-image-line fixture.
- `pipelines/nrl_procurement/generate.py::plan_single_document_requests`
  (line 307) stores the original chunk text as `source_passage` but
  overwrites `row["passage"]` with `row["generation_passage"]` (the
  image-line-stripped, whitespace-collapsed rendering built by
  `corpus.generation_text()`) for prompting. `ProcurementGenerator.parse`
  (line 546) then computes `start_char`/`end_char` via
  `row["passage"].find(quote)` — i.e. against `generation_passage` — while
  `citations[].chunk_id` still names the original chunk. `generation_text()`
  strips leading/inter-line content, so the returned offset is only valid in
  the cleaned string's reference frame, not the registered chunk's. This is
  the same bug class already fixed for propositions, not yet applied to QA
  generation.
- `pipelines/nrl_procurement/cross_document.py` and `cross_stage.py` compute
  prompt text, offsets, and citations from the same untransformed
  `document["passage"]` throughout (no `generation_passage` substitution
  exists on the cross-document path), so this defect is confined to
  single-document QA/QA-CoT generation.
- `pipelines/nrl_procurement/export.py` (lines 56, 121, 176, 262) uniformly
  resolves manual identity as
  `[doc["manual_id"] for doc in row.get("source_documents", [])] or [row["manual_id"]]`,
  because only cross-document records carry `source_documents`; every
  single-document record instead carries a top-level `manual_id`.
  `pipelines/nrl_procurement/provenance.py::leakage_audit` (lines 240-243)
  omits the `or [row["manual_id"]]` fallback used everywhere else in this
  pipeline, so single-document records contribute nothing to the `manual` and
  `section` leakage-collision fields. With only one accepted cross-document
  record contributing its two `source_documents` entries, `unique_values.manual`
  reports exactly 2 — matching the audit's observation precisely — while the
  35 single-document `qa`/`qa_cot` records covering ~15 manuals are silently
  excluded from manual/edition-level leakage detection, producing a false
  `passed: true`.
- `pipelines/nrl_procurement/generate.py:1470-1507` shows `ablation_judged` is
  the terminal-complete list for every entry in `ablation_judge_inputs`
  (accepted, judge-rejected, or `model_failure_after_retries` via
  `materialize_terminal_failures`), and `path_qa.build_ablation_judge_inputs`
  restricts `ablation_judge_inputs` to a subset of `ablation_passed_ids`
  (real-ablation-passed records with a complete `full`/`source_a_only`/
  `source_b_only` trial set). The neighboring, already-correct
  `independent_judge_rejected` field is `len(ablation_judged) -
  len(accepted_ablation_judgments)`, but `pending_independent_judge` (line
  1584) is `len(ablation_passed_ids) - len(accepted_ablation_judgments)` —
  the same subtrahend as the rejected count, so every judge-rejected record
  is double-counted as still "pending" even though it reached a terminal
  judge decision.
- `single_coverage["judged"]` and `cross_coverage["judged"]` (lines 1687,
  1787) call `request_coverage(planned_single, ...)` /
  `request_coverage(planned_cross, ...)` — comparing judged/generated output
  against *every* planned generation request. Deterministically rejected
  requests and deduplicated requests are correctly excluded from `judged`
  before ever reaching the judge, so they are indistinguishable in this
  computation from a genuinely missing judge response.
- [Scikit-learn common pitfalls: group leakage](https://scikit-learn.org/stable/common_pitfalls.html)
  (accessed 2026-07-30) and the documented `GroupKFold`/group-split pattern:
  related samples sharing a group identity (here, manual/edition lineage)
  must stay in one split, or evaluation leakage results. The ELI5 dataset
  precedent (81% of test questions found to be training paraphrases) is the
  same failure mode the audit observed for the OM-availability question pair.
  These sources confirm `leakage_audit`'s manual/section grouping is the
  intended mechanism for exactly this defense, so restoring its coverage
  (not adding a new mechanism) is the correct fix.
- [W3C PROV-O](https://www.w3.org/TR/prov-o/), already the basis for this
  pipeline's drafting citation-integrity work (see the addendum above),
  requires a quotation relation to identify the actual entity quoted from.
  An offset computed against a transformed rendering but attributed to the
  original registered chunk violates that relation; this is the same
  principle already applied to drafting and propositions, extended here to
  QA evidence.

Decision:

- [x] In `ProcurementGenerator.parse`, resolve each evidence quote's
  `start_char`/`end_char` against `row.get("source_passage", row["passage"])`
  instead of `row["passage"]`, matching `propositions.py`. Add a
  `citation_offset_unresolvable` deterministic-rejection reason when a quote
  cannot be found in the source chunk, so no unresolvable citation is ever
  exported. Do not add ambiguous-occurrence rejection in this pass — it is a
  distinct, non-required enhancement (`propositions.py` already has it for
  proposition evidence) and changing QA acceptance behavior beyond the
  audit's stated required rule is out of scope for this fix.
- [x] Add the same `or [row["manual_id"]]` fallback already used throughout
  `export.py` to `leakage_audit`'s `manual` and `section` fields in
  `provenance.py`, so single-document records participate in manual/edition
  leakage detection.
- [x] Change `pending_independent_judge` to `len(ablation_passed_ids) -
  len(ablation_judged)`, matching the terminal-lineage accounting already
  used for `independent_judge_rejected`.
- [x] Compute `single_coverage["judged"]` / `cross_coverage["judged"]` against
  the judge-eligible planned subset (planned requests whose generated
  candidate reached `generated`/`cross_generated`, i.e. passed deterministic
  checks and survived deduplication), not the full planned-request list.

Rejected alternatives:

- Rewriting `validate_record`'s signature to take `source_passage` directly:
  rejected because it is exercised by ~15 existing unit tests with a
  `(record, passage)` signature checking grounding against exactly what the
  model was shown; the citation-offset concern is separable and belongs next
  to where offsets are actually computed, matching how `propositions.py`
  keeps its own local resolution.
- Silently clamping unresolved offsets to `-1` without adding a rejection
  reason: rejected because the release rule is "anything else must be
  rejected before export," not merely flagged.
- Treating `judged` coverage gaps as a new terminal-audit row: rejected
  because deterministic rejection and dedup removal already have their own
  audit trails (`qa_rejected.jsonl`, `duplicates` counter); the bug is a
  wrong comparison base, not a missing audit artifact.

Known risks:

- The corrected `judged` coverage still depends on `generated`/`cross_generated`
  already being the authoritative judge-eligible set; if a future stage adds
  another filter between generation and judging without updating this
  eligible-set computation, coverage could again misclassify exclusions as
  missing.
- Restoring manual-level leakage detection will likely surface new
  collisions (same-edition rule families, near-duplicate cross-split
  questions) that pilot-020 already found by manual inspection. This fix
  makes the automated audit agree with that manual finding; it does not by
  itself implement edition-lineage-aware split assignment, which remains a
  separate, larger design task if the collisions require a different split
  algorithm rather than just being surfaced.

Validation plan:

- Add/adjust regression tests for: an inserted-image-line offset fixture on
  the QA path (mirroring the existing propositions test), an unresolvable
  QA evidence quote being rejected, `leakage_audit` catching a single-document
  manual collision it previously missed, `pending_independent_judge`
  reaching zero once every ablation-passed record has a terminal judge
  decision, and `judged` coverage no longer listing a deterministically
  rejected request as missing.
- Run the focused regression suite
  (`tests/nrl_procurement/test_pipeline.py`, `test_temporal.py`,
  `test_litellm_online_request_processor.py`) plus Ruff before commit.
- A bounded user-run pilot remains required before any conclusion about
  resulting yield or split composition; this fix corrects reporting/citation
  integrity, it does not itself re-run generation.

## Capability research — export-time reasoning-graph abort (2026-07-31)

Status: researched and implemented.

Observed incident: a full-corpus qa/qa_cot run (`qa-qacot-full-002`, 3,006
chunks, ~2,938 generation requests, 3,834 judge decisions, ~1.5 hours of live
Nemotron/Gemma calls) crashed at the very last step with `ValueError: 9
accepted records have invalid reasoning graphs` and produced no
`canonical.jsonl` at all.

Verified root cause:

- `export.py::export_records` (lines 140-172 before this fix) builds a
  `build_reasoning_graph(row)` for every accepted record, writes any
  structurally invalid ones to `reasoning_graph_rejected.jsonl`, and then
  unconditionally raised `ValueError` if that rejected list was non-empty —
  aborting before `canonical.jsonl` or any task-specific export was written,
  discarding every other accepted record along with the genuinely bad ones.
- `provenance.py::build_reasoning_graph` treats a `qa_cot` record's
  `terminal_claim_ids` ancestry as reachable only through the linear
  `reasoning_steps` chain (`previous_outputs` carries forward only the most
  recent step's output claim) plus each step's own directly cited evidence.
  A top-level `claims` entry whose evidence is never cited by any
  `reasoning_steps` entry has no outgoing adjacency edge and is therefore
  unreachable backward from the terminal claim, firing
  `disconnected_claims` + `unused_source_claim`.
- All 9 rejected records in the incident matched this exact pattern (8x
  `disconnected_claims`/`unused_source_claim`, 1x
  `invalid_or_duplicate_claim_ids`): the model's parallel `claims` and
  `reasoning_steps` outputs disagreed about which atomic claim was actually
  walked through, a genuine per-record data defect, not a false positive.
  Reproduced locally with `_exportable_record(unused_claim=True)` in
  `tests/nrl_procurement/test_pipeline.py`.
- Confirmed via the crashed run's own `manifest.json` `resume` section that
  the generation (7,299 records) and judge (3,834 decisions) stages were
  both checkpointed (`status: "executed"`) before the crash, so a re-run
  with the same `--run-id` reuses the cached model responses and only
  re-executes export — the fix does not require repeating the live calls.

Decision:

- [x] Change `export_records` to exclude only the graph-invalid records
  (still logged to `reasoning_graph_rejected.jsonl` with their issues) and
  continue exporting every other accepted record, matching the existing
  pipeline convention of excluding bad records rather than discarding a
  whole run (mirrors deterministic-rejection and dedup handling elsewhere
  in `generate.py`).
- [x] Mutate the caller's `records` list in place (`records[:] =
  graph_valid`) so `main()`'s downstream task-type counts and manifest
  statistics, computed from the same list object after `export_records`
  returns, agree with what was actually written to `canonical.jsonl`.
- [x] Add `stats["reasoning_graphs_rejected"]` alongside the existing
  `reasoning_graphs_valid` counter for manifest visibility.

Rejected alternatives:

- Keeping the hard abort and only fixing it operationally (e.g. asking
  operators to manually filter and re-export): rejected because the
  pipeline's own resumability design already treats completed
  generation/judge checkpoints as reusable; the export step should honor
  that same philosophy instead of requiring a full manual recovery workflow
  for a handful of bad records.
- Also adding a generation-time deterministic check that every top-level
  claim's evidence is cited by some reasoning step (catching this before
  judging, not just before export): deferred as a separate, worthwhile
  follow-up rather than bundled into this incident fix, since the acute
  problem is the crash's blast radius, not that the defect exists at all.

Validation:

- Added `test_export_drops_only_graph_invalid_records_instead_of_aborting`,
  reproducing the exact `disconnected_claims`/`unused_source_claim` pattern
  from the incident and asserting export no longer raises, the bad record
  is excluded from `canonical.jsonl` and the mutated `records` list, and it
  still appears in `reasoning_graph_rejected.jsonl`.
- Full focused suite (101 tests) and Ruff pass.
- Operational follow-up: re-run `qa-qacot-full-002` with the same run ID to
  resume from the checkpointed generation/judge stages and complete export
  without repeating the live model calls.

## Capability research — train/eval file split leakage in export (2026-07-31)

Status: researched and implemented.

Observed incident: manual review of `qa-qacot-full-002`'s exported files
found `qa_sft.jsonl` (2,163 records) spanning all three splits (`train`
1,676 / `validation` 254 / `test` 233) and `eval.jsonl` (2,493 records,
1,930/305/258) sharing **100%** record_id overlap with `qa_sft.jsonl` and
`qa_cot_sft.jsonl`. `assign_splits`/`leakage_audit` correctly assign and
audit the `split` field upstream, but `export_records` never applied it
when writing the per-task-type files.

Verified root cause:

- `export.py::export_records` (before this fix) built `qa`/`cot`/`cross_qa`/
  `cross_cot`/`rag`/`evaluation` from every accepted record regardless of
  `row["split"]`, then wrote each straight to a single flat file. The
  `split` field was present per-row inside each exported object, but
  nothing partitioned the files themselves on it. A consumer training on
  `qa_sft.jsonl` "as-is" and evaluating on `eval.jsonl` "as-is" would
  silently train on part of what they believe is held-out evaluation data.

- [Hugging Face `datasets` repository-structure convention](https://huggingface.co/docs/datasets/repository_structure)
  (accessed 2026-07-31): splits are conventionally separated either by
  filename (`train.csv`/`validation.csv`/`test.csv`) or by directory, and
  the library infers split membership from the file itself — never from a
  field inside a shared file. Manual-split guidance further recommends
  auditing for exact and near-duplicate overlap between split files as a
  release gate, which is exactly the check that was missing here at the
  file level despite already existing at the record-assignment level
  (`leakage_audit`).

Decision:

- [x] Filter every `*_sft.jsonl` export (`qa_sft`, `qa_cot_sft`,
  `cross_document_qa_sft`, `cross_document_qa_cot_sft`) and `rag.jsonl` to
  `split == "train"` only.
- [x] Filter `eval.jsonl` to non-train splits (`validation` + `test`) only.
- [x] Leave `canonical.jsonl` covering every split unchanged — it is the
  full audit/lineage record, not a ready-to-train artifact, and collapsing
  it to one split would break `leakage_audit`/manifest statistics that
  legitimately need the whole accepted population.
- [x] Add explicit per-file record counts (`qa_sft_records`,
  `qa_cot_sft_records`, `cross_document_qa_sft_records`,
  `cross_document_qa_cot_sft_records`, `rag_records`, `eval_records`) to
  manifest statistics so a future train/eval mismatch is visible without
  manually diffing record_ids across files by hand, the way this one was
  found.

Rejected alternatives:

- Splitting into per-split files (`qa_sft_train.jsonl`,
  `qa_sft_validation.jsonl`, `qa_sft_test.jsonl`, matching the HF filename
  convention exactly): rejected for now as unnecessary complexity — the
  existing single-file-per-purpose naming (`*_sft.jsonl` = train,
  `eval.jsonl` = non-train) is unambiguous once the invariant holds, and
  the `split` field remains in every exported row for anyone who wants
  finer partitioning (e.g. separating validation from test within
  `eval.jsonl`).
- Leaving `rag.jsonl` all-split (treating it as a distinct, non-training
  artifact): rejected because it is exported alongside the SFT files as a
  ready-to-use artifact with the same messages/answer shape, and nothing
  in this pipeline documents `rag.jsonl` as an evaluation-only file; the
  same train/eval overlap risk applies to it.

Validation:

- Added `test_export_never_mixes_splits_between_sft_and_eval_files`,
  constructing train/validation/test records across distinct manuals
  (avoiding `leakage_audit` collisions unrelated to this fix) and asserting
  `qa_sft.jsonl`/`rag.jsonl` contain only train record_ids, `eval.jsonl`
  contains only non-train record_ids, and the two sets never intersect.
- Full focused suite (102 tests) and Ruff pass.
- A bounded user-run pilot remains required to confirm the fix against a
  real multi-split run before treating any prior run's `qa_sft.jsonl` as
  safe to train on; `qa-qacot-full-002`'s existing exported files predate
  this fix and should be re-exported (resume from checkpoint, same run ID)
  rather than trusted as-is.

## Capability research — corpus-level question-opener template collapse (2026-07-31)

Status: researched and implemented; supersedes an inadequate first attempt
from the same session (see "Rejected alternatives").

Observed defect: manual review of `qa-qacot-full-002`'s canonical export
found 2,111/2,493 (84.7%) accepted questions begin with the literal phrase
"According to." Verified root cause: `ProcurementGenerator.prompt()`
(`generate.py`, CONSTRAINTS section) requires every question to "identify
the organization, manual, domain, or date needed to make its authority and
temporal scope unambiguous" — a genuinely necessary requirement, since
`qa_sft.jsonl` flattens questions from 16 manuals into one file — but gives
the model no guidance on *how* to satisfy it and no pressure against a
single stereotyped realization. Each of the ~2,938 generation requests is
independent and stateless (one isolated passage per call, no shared memory
across calls), so the model reliably converges on its single most
statistically default construction absent any counter-pressure.

First attempt (rejected): a within-one-response check
(`_question_opener_key` comparing the `examples` returned by a single
`CandidateBatch` call) plus a prompt line forbidding repeated openers
*within one batch*. Identified as structurally inadequate: a single
generation call returns at most `examples_per_chunk` (3) records, so this
mechanism could only ever catch a same-call collision — a negligible
fraction of a defect that spans thousands of independent calls. Removed in
favor of a corpus-level mechanism.

Verified findings from primary sources:

- [Self-Instruct (Wang et al., ACL 2023)](https://aclanthology.org/2023.acl-long.754.pdf)
  (accessed 2026-07-31): a newly generated instruction is added to the task
  pool only when its ROUGE-L similarity with every existing pool instruction
  is below 0.7. This is explicitly a growing-pool, corpus-level filter — it
  rejects a candidate based on its similarity to everything generated
  *so far*, not based on anything visible to the single call that produced
  it. This is the established, primary-source-backed technique for exactly
  this class of problem (template/instruction collapse across many
  independent generations).
- [The LLM Data Auditor](https://arxiv.org/abs/2601.17717) (survey) and
  [The Price of Format](https://arxiv.org/abs/2505.18949) (accessed
  2026-08-02): the former motivates intrinsic synthetic-data diversity
  measurement, while the latter provides evidence that rigid formatting can
  reduce output diversity even at high sampling temperature. Neither paper is
  the source of the term "cross-batch mode collapse."
- [Dynamic Context Evolution](https://arxiv.org/abs/2604.07147) (accessed
  2026-08-02) specifically defines cross-batch mode collapse for repeated
  stateless generation. Its experiments support combining semantic-memory
  deduplication with adaptive prompt evolution; a simple opener quota is only
  a narrow observable guardrail, not an equivalent guarantee of semantic
  diversity.
- Local precedent: `validation.py::deduplicate` already implements exactly
  this shape of corpus-level filter — a deterministic, single-pass,
  growing-pool comparison (`rapidfuzz.fuzz.token_set_ratio` against
  already-accepted questions) — for near-duplicate *full-text* removal.
  This defect is the same failure class at a different granularity (shared
  opening template rather than near-identical full question), so the
  correct fix reuses the same architectural position and determinism
  guarantees rather than inventing a new mechanism.

Decision:

- [x] Keep the prompt wording improvement (the identifying detail does not
  have to open the sentence; phrase the question in the assigned persona's
  authentic voice) as a source-side complementary signal — grounded in the
  "vary prompts via persona/style" finding from general synthetic-data
  diversity research — but remove the batch-scoped "no two of your own
  returned questions" instruction and its matching code-level check, since
  neither can address a phenomenon that spans independent calls.
- [x] Add `validation.py::enforce_question_opener_diversity`, a
  deterministic, single-pass, growing-pool cap on how much of the surviving
  pool may share one normalized 4-word opening n-gram (default
  `max_share=0.15`, configurable via `quality.max_question_opener_share`),
  matching `deduplicate`'s exact signature/return shape
  (`(records, removed_count)`) and processing order.
- [x] Call it in `generate.py::main()` immediately after `deduplicate()`,
  before judging — matching this pipeline's established cost-conscious
  ordering (deterministic rejection always precedes the paid judge call for
  a candidate that would be discarded anyway), and before the pool sees
  content this run has no way to un-generate.
- [x] Add `question_opener_diversity` (unique openers, top opener, top
  opener's share) to `export.py`'s manifest statistics as a passive,
  non-rejecting corpus-level visibility signal — independent of the active
  enforcement above, since the enforcement only sees the pre-judge pool and
  a small residual concentration could still remain after judging/dedup.

Rejected alternatives:

- The within-batch check from the first attempt: rejected as described
  above — wrong scope, can only ever address a same-call collision.
- Hardcoding a fixed set of alternative question-opening phrases (e.g. "use
  one of these 5 templates"): explicitly rejected per user instruction —
  this only relocates the collapse from 1 template to N hardcoded ones and
  does not generalize to any phrasing pattern the model might otherwise
  produce.
- Enforcing the quota only after judging (on `accepted` rather than
  `generated`): rejected because it wastes a paid judge call on a candidate
  that the quota would discard anyway, inconsistent with every other
  deterministic-before-judge ordering already established in this pipeline.
- Feeding a rolling sample of prior questions back into each subsequent
  generation prompt (true shared-context mitigation, closer to how
  Self-Instruct also varies its seed pool): deferred as a larger
  architectural change requiring per-request state threading through
  Curator's stateless, concurrently-dispatched request model. The post-hoc
  pool cap enforces only its declared opening-ngram metric and must not be
  described as providing the same semantic-diversity guarantee.

Validation:

- Unit tests for `enforce_question_opener_diversity`: caps a dominant
  opener to its configured share, preserves genuinely diverse questions
  untouched, and is deterministic given the same input order.
- Regression test confirming the removed within-batch mechanism's absence
  does not reintroduce the earlier `citation_offset_unresolvable`/other
  deterministic checks (no interaction expected, but re-run full suite).
- A bounded user-run pilot remains required to measure the actual resulting
  opener-share distribution and yield impact; `max_share=0.15` is a
  reasoned default, not empirically calibrated against this corpus yet.

## Capability research — answer-copy concentration and QA/CoT balance (2026-08-02)

Status: researched and implemented locally; a fresh model-backed run is required.

Observed defects in `qa-qacot-full-002` (independently recomputed from
`canonical.jsonl` rather than copied from the manifest):

- 2,111/2,493 questions (84.68%) begin with literal `According to`.
- 1,311/2,493 answers (52.59%) are normalized contiguous substrings of one
  evidence quote; 841 (33.73%) exactly equal a normalized evidence quote.
- Only 330/2,493 accepted records (13.24%) are `qa_cot`, despite a 25%
  pre-generation request allocation, confirming materially lower CoT survival.
- The historical exports overlap on all 2,493 record IDs between training SFT
  files and `eval.jsonl`; those files predate the split-safe exporter commit.

Primary-source findings:

- [Explicit Diversity Conditions for Effective Question Answer Generation
  (LREC-COLING 2024)](https://aclanthology.org/2024.lrec-main.601/) reports
  that redundant synthetic QA generation harms downstream QA and that explicit
  diversity conditions outperform sampling-only diversity. It supports an
  explicit portfolio constraint, but does not prescribe this pipeline's 35%
  answer-copy ceiling.
- [Self-Instruct (ACL 2023)](https://aclanthology.org/2023.acl-long.754/)
  filters invalid and similar generated instructions before fine-tuning,
  supporting deterministic pool-level selection rather than prompt wording
  alone.
- [The Flan Collection (ICML 2023)](https://research.google/pubs/the-flan-collection-designing-data-and-methods-for-effective-instruction-tuning/)
  finds task balancing and mixing ordinary and chain-of-thought prompt settings
  important for instruction-tuning performance. It supports an explicit QA/CoT
  mixture gate, but does not establish a universal optimal CoT percentage.
- [Hugging Face dataset repository structure](https://huggingface.co/docs/hub/main/datasets-data-files-configuration)
  separates train/validation/test by files or declared split configuration,
  reinforcing the already-landed train-only SFT and non-train eval exports.

Decision:

- [x] Preserve short exact names, values, and labels, where paraphrasing would
  reduce precision. Classify only answers of four or more normalized words that
  occur contiguously inside one evidence quote as substantial span copying.
- [x] Cap substantial extractive answers at 35% of the resulting pool before
  judging and again after judging. The 35% ceiling is a conservative project
  policy chosen to retain some useful extractive QA; it is not a literature-
  derived optimum and must be calibrated with downstream and human evaluation.
- [x] Correct opener enforcement to calculate share against the resulting pool,
  not the original pool, and re-apply it after judge attrition.
- [x] Raise planned `qa_cot_fraction` from 0.25 to 0.40 and require at least
  20% accepted QA-CoT for a complete manifest. These are explicit operational
  targets, not claims of universal optimality.
- [x] Add accepted-pool answer-style metrics and portfolio-removal counts to the
  manifest; portfolio removals remain auditable in `qa_rejected.jsonl`.
- [x] Make `validate_run.py` verify train-only SFT, non-train eval, zero ID
  overlap, split-safe cardinality reconciliation, and portfolio-quality status.

Rejected alternatives:

- Reject every extractive answer: rejected because precise thresholds, names,
  titles, and prescribed wording are valid extractive QA targets.
- Relabel direct QA as CoT or discard QA until a target ratio appears: rejected
  because it fabricates reasoning shape or hides yield. Increase preassigned CoT
  opportunities and fail the release gate transparently if accepted yield stays
  below target.
- Treat opener diversity as semantic diversity: rejected; the opener metric is
  deliberately narrow and must remain separately reported from full-question
  deduplication and future embedding/cluster analysis.

## Capability research — intent-planned, grounded diversity portfolio (2026-08-02)

Status: deterministic first production slice and the disabled-by-default
embedding/calibration infrastructure are implemented. Human threshold validation,
adversarial abstentions, and downstream threshold calibration remain staged work.

Reference implementation audit:

- `/home/abhishek/nrl_curator_native_glm52` is comparative evidence, not the
  target repository. Its useful separation is topic -> grounded QA blueprint ->
  one final QA record. Its `QABlueprint` fixes question type, task, persona,
  answer style, goal, required facts, and exact evidence before final wording.
- The reference's open-ended "create ALL" multi-pass saturation is not copied.
  It adds another model-selected taxonomy surface, its stopping quality depends
  on fuzzy wording thresholds, and its working tree is materially dirty. Curator
  instead assigns feasible intent/format contracts deterministically, requests
  one compact grounded blueprint, and requests one final record from that plan.
- In the interrupted `qa-qacot-full-003` generation cache, 215/2,494 requests
  (8.62%) have response errors. Failures include list containers serialized as
  strings, swapped task/persona fields, invented enum values, empty evidence,
  and arbitrary eight-character evidence rejection. Separately, successful
  schema responses generated page-number questions from contents pages. These
  are distinct structural-output and source-selection defects.

External benchmark used only for aggregate comparison:
`/home/abhishek/review.jsonl` contains 200 records supplied by another user. It
is not approved for copying into prompts, fixtures, or training exports. Its
useful aggregate signals are 3% literal `According to` openings, 199/200 unique
normalized questions, broad question/answer forms, and very low whole-answer
source-span copying. Its defects (empty human-review fields, systematic missing
Consultancy citation metadata, false abstentions, unsupported teaching examples,
and one exact duplicate) are explicit reasons not to ingest it.

Primary-source findings:

- [Self-Instruct (ACL 2023)](https://aclanthology.org/2023.acl-long.754/)
  supports invalid/similar-example filtering, but does not make lexical filters
  a substitute for semantic skill coverage.
- [InsTag (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9dae2a90bae49dc874ce1ca8fcc20879-Abstract-Conference.html)
  supports fine-grained intent tagging and diversity-aware selection. This
  pipeline already has the necessary `question_type`, task, and persona labels;
  adding a duplicate intent schema would create drift.
- [Measuring Data Diversity for Instruction Tuning / NovelSum (ACL 2025)](https://aclanthology.org/2025.acl-long.908/)
  finds that useful diversity measurement must consider both inter-sample
  difference and local information density. Lexical uniqueness and opener
  entropy are therefore monitoring signals, not semantic-diversity proof.
- [SemDeDup](https://arxiv.org/abs/2303.09540) supports embedding-based semantic
  duplicate detection as a second-stage selector after cheap exact/lexical
  filtering.
- [FActScore (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.741/)
  motivates atomic-fact support; the existing claim/evidence model is retained
  and extended rather than replaced with answer/source similarity.
- [QAFactEval (NAACL 2022)](https://aclanthology.org/2022.naacl-main.187/)
  finds question filtering and answerability classification critical and shows
  QA- and entailment-style signals are complementary.
- [SQuAD 2.0](https://nlp.stanford.edu/pubs/rajpurkar2018squad.pdf) constructs
  unanswerable questions deliberately around plausible distractors. Retrieval
  failures, contents pages, and empty tables are not acceptable negative QA.
- [Judging LLM-as-a-Judge (NeurIPS 2023)](https://papers.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)
  documents verbosity and other judge biases. Deterministic format and length
  checks must complement the independent judge.
- [NVIDIA llama-nemotron-embed-1b-v2 API reference](https://docs.api.nvidia.com/nim/re/reference/nvidia-llama-nemotron-embed-1b-v2-infer)
  defines the OpenAI-compatible embeddings request, including the required
  `input_type`, float encoding, truncation policy, and optional dimensions.
- [NVIDIA llama-nemotron-embed-1b-v2 model card](https://build.nvidia.com/nvidia/llama-nemotron-embed-1b-v2/modelcard)
  documents an 8,192-token maximum, a 2,048 base dimension, and supported
  reduced dimensions. Curator uses `input_type=query` and a configurable 1,024-
  dimensional profile for generated-question comparison; this is an operational
  choice, not a claim that 1,024 is universally optimal.
- [NVIDIA NeMo Retriever embedding NIM reference](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/2.2.0/reference.html)
  confirms the current query/passage distinction. Only generated question text
  may leave the Curator host; answers, evidence, source passages, and credentials
  are excluded from the embedding payload.

Implemented decisions:

- [x] Add high-confidence `corpus.py::source_quality_issues` screening for
  front matter, contents-only chunks, abbreviation-only chunks, effectively
  blank forms/tables, and undersized text. Preserve mixed prose/tables by
  failing open. Persist every exclusion in `source_quality_rejected.jsonl` and
  expose reason counts in `corpus_quality.json`.
- [x] Reuse the existing `QuestionType` schema. Add deterministic
  `eligible_question_types` source-signal classification and deficit-balanced
  `plan_question_types`; attach `planned_question_type` to every single-source
  request and reject model substitutions.
- [x] Derive `answer_format` deterministically from the planned question type:
  concise direct, ordered steps, audit check, compact comparison,
  rule-and-exception, responsibility summary, or dated-scope summary. Do not add
  another model-selected response-schema enum.
- [x] Add a real `qa_blueprints` stage between source planning and final
  generation. It produces exactly one compact, evidence-grounded goal with
  `must_cover`, task, and persona. Persist every result or terminal failure in
  `qa_blueprints_audit.jsonl` and expose blueprint request coverage.
- [x] Make final generation singular (`GroundedCandidateDraft`), not
  `CandidateBatch.examples`. Inject task type, question type, answerability,
  task, persona, answer format, and blueprint identity from the fixed plan; the
  final model emits only question, answer, atomic claims, and auditable rationale
  steps. Derive top-level evidence from claim evidence instead of asking the
  model to duplicate the same list.
- [x] Permit short exact evidence such as `Buyer` or `Seller`; validate exact
  grounding and claim support rather than imposing an unrelated eight-character
  schema minimum.
- [x] Add per-format answer-length limits and high-confidence rejection of
  unsupported lecture, role-play, discussion, and case-study embellishments.
- [x] Reject unsupported answer acronyms in addition to the existing unsupported
  quantity and legal-modality checks.
- [x] Lower the provisional four-word opener ceiling from 15% to 8%. This is a
  project pilot target informed by the external benchmark, not a universal
  literature-derived optimum.
- [x] Cap any one accepted single-document `question_type` at 30% before and
  after judge attrition, with auditable portfolio rejection reasons.
- [x] Report question-type and answer-format concentration/entropy, answer
  lengths by format, eight-gram source coverage, and copied-sentence fraction in
  the export manifest.
- [x] Keep exact structured evidence offsets as citation truth. Human-readable
  citations, if added later, must be rendered from structured provenance rather
  than parsed back from answer prose.

Deferred staged work:

- [x] Add an optional OpenAI-compatible NVIDIA embedding profile, mandatory
  capability probe, and persistent record-ID plus question-hash keyed cache.
  Keep it disabled by default and do not add Torch/`sentence-transformers`.
  Emit nearest-neighbor review pairs to `semantic_calibration.jsonl`; never send
  source passages, answers, or evidence to the public endpoint.
- [ ] Hand-label a calibration set of duplicate/related/distinct question pairs
  before selecting a cosine threshold; no universal embedding threshold is
  accepted without calibration for the deployed model. The implemented
  `semantic_calibration.py calibrate` command requires both classes and a minimum
  sample before producing a development recommendation; validate it on a
  separate reviewed holdout before enabling deletion.
- [x] Replace order-based retention inside semantic clusters with a
  quality-aware selector using deterministic validity, independent judge score,
  qualification preservation, grounded-claim/evidence completeness, and stable
  record-ID tie-breaking. It is unreachable while `selection_enabled: false` and
  refuses to start without an explicit calibrated threshold.
- [ ] Implement adversarial unanswerable generation as an isolated stage with a
  plausible same-type distractor and an independent full-passage answerability
  judge. Keep `quality.unanswerable_fraction=0.0` until that stage exists.
- [ ] Calibrate the 8% opener, 30% question-type, answer-length, and extraction
  targets using a 200-500 record model-backed pilot plus stratified human review
  and downstream evaluation. A manifest passing provisional thresholds is not
  a substitute for that evidence.

Rejected alternatives:

- Copy the externally supplied records or their wording into this dataset:
  rejected for ownership/provenance and because their review annotations are
  empty.
- Force uniform question types or manuals regardless of source support:
  rejected because it creates fabricated questions. Planning operates only on
  types for which the passage has observable support signals.
- Treat zero source overlap as the goal: rejected because exact terms,
  thresholds, authorities, and prescribed language legitimately require exact
  wording. Atomic support and controlled extraction are separate dimensions.
- Expand the already failure-prone `CandidateBatch` tool schema with cosmetic
  style labels: rejected. The batch was removed from final single-document
  generation; answer format is derived outside the model response.
- Copy the reference's "all distinct blueprints" saturation loop: rejected for
  the first production slice because it multiplies calls and schema failures
  without a calibrated semantic stopping rule. A later saturation feature must
  stop on accepted semantic novelty, not model assertion or a raw pass count.
- Train on ordinary TOC/retrieval failures as abstentions: rejected; these teach
  retrieval accidents, not calibrated refusal behavior.

## Smoke latency, Gemma mode A/B, and instruction diversity (2026-08-08)

- [x] Attribute the 38-minute smoke from recorded stage timings. The two GLM
  cross-document passes consumed about 1,303 seconds (~22 minutes); Gemma
  thinking judgments then generated long traces, invoked output rescue, and the
  final all-failed answerability batch crashed before rescue.
- [x] Compare `/home/abhishek/nrl_curator_native_glm52`. Carry over its relevant
  operational pattern: `require_all_responses=false`, explicit missing-row
  reconciliation, one larger output rescue, a 5,000-token ordinary GLM reserve,
  and thinking disabled for schema-constrained stages.
- [x] Fix Curator's all-failed edge case. When partial results are explicitly
  allowed, persist `failed_requests.jsonl` and return an empty Dataset so the
  pipeline can invoke recovery; retain fail-fast behavior when
  `require_all_responses=true`.
- [x] Route answerability judgment through the same separately checkpointed,
  context-preflighted 4,096-token rescue as other singular judge stages.
- [x] Add `gemma_structured` on the same shared-gateway Gemma deployment with
  temperature 1.0, top-p 0.95, top-k 64, and thinking disabled. Keep
  `gemma_thinking` as an explicit opt-in profile.
- [x] Run a live same-deployment mode comparison. The schema probe passed in
  both modes: thinking used 401 output tokens and 6.87s; non-thinking used 73
  tokens and 1.71s. On the same three real procurement judge records,
  non-thinking reproduced two score-5 acceptances and one score-3 rejection,
  completed all three in 4.91s with 644 output tokens, and needed no rescue.
  Thinking required 76.43s plus 46.68s rescue and at least 5,621 successful
  output tokens; two ordinary responses also truncated, whose consumed tokens
  Curator did not include in that total.
- [x] Reduce ordinary GLM `max_tokens` from 8,192 to 5,000 while preserving the
  12,000-token missing-row rescue.
- [x] Add regression coverage for an all-failed Curator batch and direct
  answerability rows entering judge rescue.
- [ ] Validate the Gemma mode choice on at least 100-200 stratified, human-
  labeled judge cases (accept/reject, direct/CoT, single/cross-document,
  answerable/unanswerable, boundary/adversarial). Compare precision, recall,
  agreement by failure class, schema success, truncation, tokens, and latency;
  the three-row A/B is strong operational evidence, not a quality proof.
- [ ] Run a fresh new-ID end-to-end smoke and compare per-stage wall time,
  output tokens, truncation/rescue yield, and accepted records against
  `run-20260808T122041-362263Z`.
- [ ] Ask the vLLM deployment owner to capture queue/TTFT/TPOT/KV-cache metrics
  and validate `--max-num-seqs`, chunked prefill, and
  `--max-num-batched-tokens`; client concurrency cannot improve a stage that
  contains fewer requests than its ceiling.
- [ ] Add an explicit, source-feasible difficulty/operation coverage matrix for
  instruction synthesis. Cross product only supported axes: procurement task,
  authentic persona need, question intent, reasoning operation, answer format,
  single/multi-hop context, and basic/intermediate/advanced difficulty. Never
  manufacture a scenario solely to fill a quota.
- [ ] Add materially distinct scenario planning for CoT: each variation must
  change the governing condition, exception, threshold boundary, stakeholder
  decision, evidence requirement, temporal state, or reasoning path—not merely
  wording. Generate multiple candidate rationales only for genuinely multi-step
  records, verify every step against exact evidence, and select by grounded
  correctness before semantic diversity.
- [ ] Human-label the existing semantic-neighbor calibration artifact and set a
  model-specific threshold before enabling embedding-based deletion. Lexical
  opener/type/style balancing is not sufficient evidence of semantic diversity.

Research basis:

- Self-Instruct generates instruction/input/output candidates and filters
  invalid or similar items: <https://aclanthology.org/2023.acl-long.754/>.
- Evol-Instruct increases complexity by controlled instruction evolution, but
  procurement evolution must remain source-feasible:
  <https://arxiv.org/abs/2304.12244>.
- DEITA evaluates instruction data jointly on complexity, quality, and
  diversity rather than maximizing raw volume:
  <https://arxiv.org/abs/2312.15685>.
- Bonito conditions task generation on unannotated domain text and explicit
  task attributes: <https://aclanthology.org/2024.findings-acl.748/>.
- RAFT trains open-book domain QA with relevant evidence, distractors, and
  rationale-style answers: <https://arxiv.org/abs/2403.10131>.
- CoT prompting supports genuinely complex multi-step tasks, while STaR keeps
  rationales that lead to verified answers and self-consistency selects among
  diverse paths: <https://proceedings.neurips.cc/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html>,
  <https://papers.nips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html>,
  <https://openreview.net/pdf?id=1PL1NIMMrw>.
- Google documents Gemma 4's explicit `enable_thinking` switch; vLLM documents
  how the same flag activates reasoning output:
  <https://ai.google.dev/gemma/docs/capabilities/thinking>,
  <https://docs.vllm.ai/en/stable/features/reasoning_outputs/>.
- vLLM's optimization and serving guides define chunked-prefill tuning,
  `max_num_batched_tokens`, and `max_num_seqs`:
  <https://docs.vllm.ai/en/stable/configuration/optimization/>,
  <https://docs.vllm.ai/en/stable/cli/serve/>.
