# E0 activation audit: decision memo

This is an exploratory audit of previously inspected data, not a confirmatory result.

## Findings

- A 4,096-only experiment cannot support a timing or intervention-window claim. It
  can only evaluate a monitor at that decision point.
- On one fixed long-trace cohort (197 traces, 53 mixed-outcome problems), a
  depth-3 tree using 50 training-selected layer-20 neurons reached within-problem
  AUROC .642/.679/.642 at 6,144 tokens across three grouped fold assignments.
  The observable-prefix baseline was .553/.572/.553. Paired gains were about
  .09-.11, but every interval against that baseline included zero. The pattern was
  weaker at 4,096 and 8,192 and therefore needs independent replication.
- Early estimates depend on retrospective population selection. At 512 tokens,
  top-50 linear AUROC was .575-.613 among traces later reaching 4,096, versus
  .510-.588 in the broader 512-token cohort. These are different training and
  evaluation populations, so this is a sensitivity warning rather than a causal
  estimate.
- Reconstructed output-head predictions matched the re-tokenized next token about
  86-94% of the time, while neighboring offsets matched poorly. This argues against
  a systematic one-token activation offset. All examined end states predicted EOS;
  end-state results are after-answer controls, not pre-answer warnings.
- Next-token entropy/confidence did not reliably add early outcome prediction.
  Some early high-dimensional logistic models also had worse Brier/log loss than a
  training-only prevalence predictor.
- Across 324 exploratory seed-0 comparisons against observable and TF-IDF
  baselines, nine had positive nominal lower bounds and none retained a positive
  simultaneous lower bound. This does not prove equivalence or universal absence.

## Interpretation

Do not claim that the model has no early error signal, that commitment occurs at a
specific token, or that intervention cannot work. Final-outcome prediction, error
localization, continuation success, and intervention utility are different targets.
The current 6,144 result is a development hypothesis—not a paper headline.

## Next confirmation protocol

1. Treat all current problems as development data. Freeze the representation,
   checkpoints, useful-effect margin, grouped folds, and analysis before collecting
   new problems.
2. Re-extract from exact generated token IDs with pinned model/tokenizer revisions
   and one BOS convention. Store token IDs, positions, hashes, sampling settings,
   and a genuine pre-answer boundary.
3. Evaluate all prefixes alive at 512, 2,048, 4,096, 6,144, and 8,192 without
   selecting them for survival to a future checkpoint. Separately report a fixed
   matched cohort for timing comparisons.
4. Confirm only a small frozen family: observable and TF-IDF baselines; regularized
   linear hidden-state probe; the depth-3 top-50 tree; and a trailing-window pooled
   representation plus checkpoint-to-checkpoint change.
5. At selected checkpoints, sample eight matched-temperature continuations using
   the actual remaining-token budget. Compare no intervention, fixed/random timing,
   monitor-triggered resampling, and an equal-compute continuation baseline.
6. Run the proposed consequential-error versus matched harmless-edit experiment as
   a separate controlled-onset study.

The strongest defensible paper question is whether a cheap prefix monitor improves
correctness under equal inference compute, beyond observable reasoning behavior.
