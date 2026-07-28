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
