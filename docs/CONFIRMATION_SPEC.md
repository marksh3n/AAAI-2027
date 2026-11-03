# Confirmation spec

Frozen design for the confirmation run: monitors, policies, cost accounting, tests. Constants live in
`src/confirm/spec.py`; if this file and `spec.py` disagree, `spec.py` is what runs and the mismatch is a
bug to fix before any confirmation data exists.

| Hash | Value |
|---|---|
| `spec.spec_sha()`, frozen design | `f1e4f6ef4c510e738e04b8fd9937a8de17f1305351a51e04d8d8320d19d75df2` |
| `spec.monitor_sha()`, monitor fit only | `22612aaffbf5fc8d26fda3a450363e2d9127062f1b40530d94a4db8b1d2a92a5` |
| `monitors/manifest.json` sha256 (refit 2026-10-27; `.joblib` bytes identical to the 2026-10-25 fit, manifest `abcd7a2f…37f1`) | `344b7fa22399076799a71bc8b83cfef81d80eb01d4568091d95de065f35c5b83` |
| `spec.slots_sha()`, team slots (status `pending_team_choice`) | `c9cdcf5a4f65b8b8b8701880683bd4e99cf56c4d92a15191c6b0b83c43f0e9ab` |

**Status: draft.** Freezes once the team slots and the open decisions in §11 are settled; the primary
test (§11 row 4) may still change, which changes `spec_sha()`.

**Pre-registration.** After the freeze and before any confirmation GPU step: push this file, `src/confirm/`
and the monitors manifest, or send the mentor the commit hash and the hashes above. Nothing here changes once
confirmation outcomes exist; any change goes in the decision log (§10) with date and reason.

## 1. Questions

| # | Question | Answered by |
|---|---|---|
| 1 | Do sampled continuations separate outcomes within problem better than cheap monitors? | AUROC table: r̂ vs each monitor, mixed problems, paired deltas; diagnostic, rollout cost stated (§5) |
| 2 | Does a cheap monitor become informative around 6144, before the answer, while intervention still helps? | S7, matched-cohort timing table, S4/S5 |
| 3 | Does a monitor improve accuracy at equal total generated tokens? | H1 (§7) |

## 2. Engine and data

Sources and rollouts come from one engine with one config, both through `src/recovery/run_rollouts.py`;
a source trace is a continuation of the empty prefix.

| Setting | Value |
|---|---|
| Model / tokenizer | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` @ `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| Engine | vLLM 0.10.2, transformers 4.56.2 |
| dtype | float16 (generation and extraction) |
| Sampling | T 0.7, top_p 0.95, no top_k, per-request seed `prefix_seed(0, prefix_id, round)` |
| BOS | one (the chat template's), prompt ids passed as ids; runner asserts one leading BOS and an unchanged echo |
| Budget | 32768 tokens per trace; a rollout from cut c gets `max_tokens = 32768 − c` |
| Saved | sources: prompt ids, every token id, n_gen, finish reason, seed, text, answer, grade. bank: sha256 of token ids, n_gen, finish reason, seed, text, answer, grade. per prefix: cut, prefix ids + sha, max_tokens. run configs and every file's sha256 in the bundle |

Hidden states: HF forward pass, fp16, same revision, on the exact prompt + generated ids; state at
index c − 1, layer 20 = `hidden_states[21]`. Never recomputed from retokenized text.

Cuts `CHECKPOINTS = (2048, 4096, 6144)`. A source gets rollouts at a cut only when live there (§4),
`K_PREFIX = 8` continuations. Each problem gets `K_T0 = 8` fresh samples (t0 bank), round 1 of the same
prompt records, so seeds differ from the sources' round 0.

## 3. Monitors

Fit on all development data by `fit_monitors.py`; thresholds from dev only; nothing refit on
confirmation data.

| Name | Model | Features | Cuts | Role |
|---|---|---|---|---|
| `lr_raw_L20` | logistic regression, C by grouped inner CV | layer-20 state, standardized, 1536 dims | 2048, 4096, 6144 | primary |
| `hgb_topk50_L20` | HistGradientBoosting: depth 3, 150 rounds, lr 0.05, min leaf 20, l2 1.0, no early stopping | 50 L20 columns with largest within-problem \|AUROC − .5\| on training rows | 6144 | secondary |
| `len` | logistic regression | log1p(tokens seen, `\n\n` paragraphs inside `<think>`) | 2048, 4096, 6144 | baseline |
| `tfidf` | logistic regression | word 1–2-gram tf-idf, min_df 2, sublinear tf | 2048, 4096, 6144 | baseline |

- Definitions follow probe_cmp `models.py`; `tests/test_confirm.py` checks identical out-of-fold
  predictions and C for all four.
- Training rows at c: untruncated traces reaching c, in problems with a correct and a wrong such trace
  (342 traces / 74 problems at 6144). Truncated traces stay out of training; in confirmation they are
  live until the cap, scored, and count as wrong.
- Operating point: flag when score > τ_c, τ_c = 0.75 quantile of dev scores over live traces at c in all
  problems (`Q_DEV = 0.25`). Training traces enter with out-of-fold scores (5 grouped folds, seed 0).
- Saved: `<name>@<c>.joblib` and `manifest.json` (C, τ_c, dev OOF AUROC + CI, training-set hash, input
  and code sha256, library versions). Scoring refuses a monitor whose hash doesn't match.

**6144 tree provenance** (`provenance_6144.py`, 2026-10-25). The "depth-3 tree" is probe_cmp's
`hgb/topk50`: a 150-round boosted ensemble of depth-3 trees on 50 training-selected L20 neurons.
- PR #5 memo: 197 traces / 53 problems, AUROC .642 / .679 / .642 over three fold assignments vs
  observable .553 / .572 / .553.
- Planning exports 342/74 and 227/59 are the probe_cmp fixed populations of the exploratory runs to
  6144 and 8192; refit gives tree .523 / .573, len .554 / .568.
- None of 12 candidate cohort rules gives 197/53 (closest: reach 8192, no `</think>` at 8192, 206/55).
  The memo number does not reproduce from the repo.
- Picked as best cell after ~324 comparisons. Frozen dev OOF at 6144 on 342/74: .560 [.487, .631], len
  .554, lr/raw .523.
- Decision: secondary only (S4, S5, S7); its dev effect is selected and cohort-dependent, so it is not
  used for power planning.

## 4. Definitions

- **Attempt**: one source trace; one run of a policy on its problem.
- **Live at c**: still generating after c tokens (n_gen > c) and no `</think>` in the first c tokens.
  Policies act only on live traces; every other attempt keeps its own outcome and cost.
- **Pre-answer at c** (S7), primary rule "stated" (`bundle_lib.cut_rules`): live, no `\boxed`, and the
  prefix does not state the trace's own final answer, i.e. no
  `(?i:final answer|answer is|answer:)[^\n]{0,40}?(?<![\w.])VALUE(?!\w|\.\d)` with VALUE and prefix
  normalized (`\left`/`\right` dropped, `\dfrac`/`\tfrac` → `\frac`). Sensitivities, reported, not tested:
  **strict** (no `\boxed`, value nowhere as a standalone number) and **think_boxed** (no `\boxed`).
- **Populations**: available = every live trace at c; matched cohort = live at every cut (timing only).
- **Outcome**: 1 if the final answer is correct under math_verify; truncated = 0.

## 5. Arms and cost

Cost = total generated tokens per attempt, including discarded tokens, replacements, extra votes and
selection rollouts. Hidden-state monitors add none.

| Arm | Outcome if flagged | Tokens if flagged |
|---|---|---|
| no intervention | y_i | n_i |
| restart | r0, problem's t0 bank mean | c + L0 |
| backtrack to c′ (2048 for c = 4096, 4096 for c = 6144) | r(c′), that prefix's bank mean | c + L(c′) |
| aggregation, m extra samples + plurality vote (per attempt, not flagged) | A_m, exact over m-subsets of the t0 bank plus own answer, ties split | n_i + m·L0 |

- Replacement outcome and length = bank means (expected value of one fresh draw). Unbiased because the
  decision reads the source prefix only and the bank is sampled independently by the same engine;
  checked in §8.
- Per cut and action: fixed (every live trace), random (probability π), monitor (score > τ_c).
- Aggregation m = 2 (maj@3) for tests; m = 4 reported.
- Rollout-ranked arm (cross-fitted r̂ halves, q 0.25): diagnostic for question 1, not a tested policy;
  its tokens include 4 selection rollouts per live trace.

## 6. Token matching

Monitor policy P spends `S_P = Σ_i (tok_i(P) − n_i)` extra tokens. A comparator linear in its rate
(random, aggregation) gets `π = S_P / S_C(1)`; accuracy `mean(y) + π·Σ gain_i / N`, expected tokens
equal to P's.
- Rematched inside every bootstrap draw, so CIs include matching uncertainty.
- π > 1: capped at 1, flagged. No valid rate: comparator falls back to no intervention, flagged. Share of
  failed matches reported per row.
- Equal fractions flagged are never a comparison.
- S6 compares through value over the token-matched random line: `V(P) = acc(P) − acc_random(T_P)`.

## 7. Endpoint and tests

- **Endpoint** [slot]: accuracy per attempt at equal total generated tokens, all attempts of all
  problems (restricting to mixed problems conditions on outcomes). Within-problem AUROC on mixed problems
  among scored rows.
- **Uncertainty**: problem-cluster bootstrap, B 2000, seed 0, draws shared by every arm at a cut
  (paired deltas), 95% percentile CIs.
- **H1 (primary)**: at 4096, `lr_raw_L20`-triggered restart at frozen τ beats both token-matched random
  restart and token-matched maj@3. Holds only if both lower 95% bounds exceed 0 (intersection-union, no
  multiplicity adjustment). No verdict while slots are unfrozen.
- **Rule-out**, per comparator, reported with H1: a gain ≥ margin [slot, 5 pp] is excluded when the
  one-sided 95% limit (95th bootstrap percentile) is below it. Stands in for the planning protocol's Tango
  score test, which needs one binary outcome per arm per problem; here outcomes are cluster means over
  several sources, and replacements are bank means.
- **Secondary**: Holm at α 0.05 over S1–S7; Wald p from the bootstrap SE.

  | ID | Cut | What | Compared with |
  |---|---|---|---|
  | S1 | 4096 | primary-triggered backtrack | random backtrack |
  | S2 | 2048 | primary-triggered restart | random restart |
  | S3 | 6144 | primary-triggered restart | random restart |
  | S4 | 6144 | tree-triggered restart | random restart |
  | S5 | 6144 | tree-triggered backtrack | random backtrack |
  | S6 | 4096 | primary's value over random | `len`'s value over random |
  | S7 | 6144 | tree AUROC on live pre-answer cuts ("stated") | `len` |

- **S7 branch** (`S7_MIN_MIXED = 500`, pre-registered 2026-10-27): count mixed problems among live stated
  pre-answer cuts at 6144 (labels only, no scores). Below 500: S7 reported as estimate + CI, no claim,
  out of the Holm family (then S1–S6). Dev: 61 mixed problems, half-width ≈ ±0.13; 80% power for +0.065
  needs ≈ 9× the clusters.
- **Descriptive**: every AUROC (live, pre-answer, matched cohort), r̂ vs monitors, fixed arms,
  rollout-ranked diagnostic, m = 4, baseline-triggered policies.
- **Dev dry run** (2026-10-25, 87 problems, 522 attempts, not evidence): primary vs random +1.1 pp
  [−0.6, +3.1], vs maj@3 +1.6 [−0.1, +3.4]; primary flags 16% of live traces, ≈ +400 tokens per attempt
  over 11,964; half-width ≈ 1.8 pp.

## 8. Validity: resample check and sampler gap

- Resample check per cut: replace every live trace (and separately the primary's flags) with a fresh
  continuation of its own prefix. If rollouts continue a prefix the way the source sampler did, this is ≈ 0.
- If the check's CI at the decision cut excludes 0: primary reported with a flag, plus the pre-specified
  sensitivity `analyze.py --keep rollout` (kept live traces valued at their prefix's bank mean).

**Sampler gap on dev.** Measured: continuations of a source prefix end correct more often than the source
trace itself did. Ceiling cohort, source accuracy 0.527 vs mean prefix r̂ 0.600 at 512 (7.3 pp); 5.1 and
3.0 pp at 2048 and 4096. Fresh t0 samples 0.593 vs source pass rate 0.540 on the same problems. Dry-run
resample check +4.4 pp [+1.3, +7.6] at 2048, +2.6 [−0.3, +5.5] at 4096.
- Same size in both dev blocks (512: 0.585 vs 0.658 main, 0.443 vs 0.515 pilot), so not the pilot's BOS
  mismatch or its vLLM 0.28.0.
- Not mixed selection: modelling it predicts source accuracy 0.566, observed 0.527.
- Not length selection: length-matched resample control −2.3 / −2.5 pp at 512 / 2048 vs −3.1 / −2.3 raw.
- Not grading: regrading reproduces 448/448 source labels and 400/400 rollouts.
- Not temperature: 0.7 in sources and rollouts.
- Not the larger rollout budget (32768 past the cut, not 32768 − c): 67 of 10,997 finished continuations
  ran past the source's budget, 4 of them correct; too few to explain the gap.

Open: the main block's engine version was never recorded; dev rollouts were fed retokenized prefix text,
not the sampled ids; other run differences are unlogged. Cause unknown. Until it is explained, any gain
that scores a replacement by rollouts against kept traces scored by their own outcome may carry the gap.

**Why the confirmation run avoids it.** Sources and rollouts come from one engine and config, prefixes are
fed as the exact sampled ids with one asserted BOS, and budgets are 32768 − c (§2). This removes every
known dev difference; it does not show which one caused the gap. The resample check measures whether a gap
remains, and the fallback above covers the case where it does.

## 9. Wording

- r̂ is "rollout-estimated" (K 8; 4 per half when cross-fitted), not a perfect monitor or a ceiling.
- Rollout-ranked plots are offline estimates: selector rollouts are not in the plotted cost, and equal
  shares flagged are not equal tokens.
- Token changes carry CIs. oracle_v4, all problems, before the ~35–47k selection tokens per trace:
  backtrack at 2048 −24 [−233, +163], restart at 512 −38 [−279, +199].
- A null means the tested monitors showed no intervention value at equal compute on these cohorts, not
  that hidden states lack the information.

## 10. Decision log (changes after dev outcomes were seen)

- **2026-10-25, S7 pre-answer rule.** The strict value-anywhere rule dropped many cuts for coincidental
  short numbers (193 of 357 live cuts and 30 of 70 mixed problems left at 6144); a hand audit at 512 found
  0 of 20 removed cuts stated an answer. "stated" made primary; strict and think_boxed pre-registered as
  sensitivities. Seen before the choice: strict pre-answer tree − len at 6144 +0.096 [−0.092, +0.285]; all
  live cuts tree .593, +0.065 over len. No S7 number under "stated" existed then; counts below come from
  labels only (`count_pre_answer.py`).

  | Rule (cuts kept / mixed problems) | all dev 2048 | 4096 | 6144 | dry-run 87: 2048 | 4096 | 6144 |
  |---|---|---|---|---|---|---|
  | live (`</think>` only) | 1456 / 99 | 853 / 86 | 611 / 73 | 512 / 87 | 442 / 82 | 357 / 70 |
  | stated (primary) | 1376 / 98 | 797 / 82 | 565 / 63 | 505 / 87 | 419 / 78 | 333 / 61 |
  | strict | 745 / 65 | 427 / 45 | 315 / 31 | 346 / 60 | 249 / 44 | 193 / 30 |
  | think_boxed | 1450 / 99 | 848 / 85 | 605 / 72 | 511 / 87 | 440 / 81 | 353 / 69 |

- **2026-10-25, monitor hash.** Monitors are checked against `monitor_sha()`, not the full spec hash, so
  analysis-only rules can change without a refit. Refit 2026-10-27: same `.joblib` bytes, new manifest.
- **2026-10-27, S7 estimate-only branch**, added before any S7 number under "stated" existed. Dry run
  under the frozen rule afterwards: S7 +0.090 [−0.029, +0.211], 61 mixed problems, estimate-only; primary
  contrasts unchanged.
- **2026-10-27, team slots** filled from the accuracy planning package (zip sha256 `7aef6367…aa25`),
  status `pending_team_choice`. `analyze.py` gives no verdict and `gpu_prep.py t0` refuses to start until
  `frozen`.
- **2026-10-27, renames.** Slot dict → `spec.SLOTS`, its hash → `slots_sha`, provenance flag →
  `--exports`. Values and all hashes unchanged.
- **2026-11-02, win rule (§11 row 7).** H1 now needs both lower 95% bounds above 0, not above the 5 pp
  margin; the margin becomes a rule-out bound reported per comparator. Seen before the choice: dev dry-run
  primary +1.1 [−0.6, +3.1] vs random and +1.6 [−0.1, +3.4] vs maj@3, half-width ≈ 1.8 pp, so the old rule
  needed an estimate ≳ 6.8 pp. No confirmation data exists. `spec_sha`, `monitor_sha`, `slots_sha` unchanged.
- **2026-11-02, problems file (§11 row 8).** `build_problems.py` writes the jsonl `gpu_prep.py` reads from the
  frozen id list; its sha256 goes into `SLOTS.problems_file_sha256` once run on Colab.
- **2026-11-02, slot source.** `SLOTS.source` names the planning package by its zip sha256, not its file name.
  `slots_sha` f1d63e60… → c9cdcf5a…; `spec_sha`, `monitor_sha` unchanged.

## 11. Open: planning protocol vs this spec

| # | Topic | Planning protocol | This spec | Recommendation |
|---|---|---|---|---|
| 1 | Comparator | aggregation only | random restart and maj@3, token-matched | keep random restart: at equal tokens maj@3 acts on 1.6% of attempts, so alone it is ≈ vs nothing |
| 2 | Discordance q | 0.25, effect +5 pp | dev at 4096: d +1.6 pp vs maj@3; q 0.062 shared source, 0.378 independent | pair arms on a shared source; at N 1000, McNemar power 0.51 vs 0.12, Tango rule-out of ≥ 5 pp 0.99 vs 0.54 |
| 3 | Samples per problem | one outcome per arm | several sources (dev: 6) | K = 6 with the t0 bank on all 1000 (~169M tokens); one sample per problem makes WP AUROC and S7 impossible |
| 4 | Scope | probe-triggered intervention | same | make probe value beyond length (S6) confirmatory; dev S6 +0.6 pp [−2.0, +3.3]; changes `spec.PRIMARY`, team call |
| 5 | Population | MATH L3–5, train and test | 367 MATH test problems | test only: 632 of 1000 planned are MATH train; pool has 3,301 eligible test problems |
| 6 | Equal compute | equal caps, weighted token equivalents | realized tokens, rate matched per draw | keep realized tokens; report weighted totals as audit |
| 7 | 5 pp margin | superiority vs 0 + Tango rule-out of ≥ 5 pp | superiority vs 0 + bootstrap rule-out (§7) | adopted 2026-11-02 (§10) |
| 8 | Problem file | id CSVs, no text | `gpu_prep.py` needs jsonl with problem and gold | `build_problems.py` added 2026-11-02; sha pending a Colab run and row 5 |

## 12. Files

| Path | What |
|---|---|
| `src/confirm/spec.py` | frozen constants and team slots |
| `src/confirm/monitors.py`, `fit_monitors.py` | frozen monitors; fit, thresholds, manifest |
| `src/confirm/provenance_6144.py` | where the dev 6144 numbers come from |
| `src/confirm/policy.py`, `analyze.py` | arms, cost, token matching, tests, report |
| `src/confirm/build_problems.py` | problems file (text + gold) from the frozen id list |
| `src/confirm/gpu_prep.py`, `bundle.py`, `scripts/confirm_modal.py` | confirmation data run (`docs/RUN_CONFIRM_GPU.md`) |
| `src/confirm/dev_bundle.py` | dev data in the same schema, for the dry run |
| `src/confirm/power.py`, `discordance.py` | power of the planning design's tests; dev-implied q and effect |
| `tests/test_confirm.py`, `tests/test_confirm_pipeline.py` | synthetic checks, including equality with probe_cmp fits |
