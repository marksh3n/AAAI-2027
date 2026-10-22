# Jaden confirmation preparation and sample-size planning

This workflow prepares a candidate problem pool, checks supplied development predictions,
and estimates confirmation sample sizes. It also includes exact-token generation and
extraction code whose real GPU execution still needs validation.

## Current status

- Prepared 8,881 MATH levels 3–5 candidates after MATH-500 overlap and duplicate/quality exclusions.
- The team reports that previous generation used only MATH-500; recheck exclusions if that changes.
- Checked two development exports: 342 traces / 74 mixed problems and 227 traces / 59 mixed problems.
- Calculated four provisional comparisons at checkpoint 6,144. The smaller cohort overlaps the larger one and has different predictions; do not pool them.
- All 60 local CPU tests passed when this workflow was checked. GPU validation, a frozen final protocol, and fresh confirmation generation are pending.

## Reproduce the CPU calculations

From the repository root:

```bash
python -m pip install -r experiments/jaden_confirmation/requirements-cpu.txt
python -m unittest discover -s experiments/jaden_confirmation/tests
python experiments/jaden_confirmation/scripts/wide_prediction_planning.py --work experiments/jaden_confirmation
```

Outputs go to `experiments/jaden_confirmation/outputs/development_planning/`.
The two small prediction CSVs are included unchanged; their hashes are in
`reports/input_provenance.json`. No model training or GPU run occurs in these commands.

## Colab

Upload `notebooks/Jaden_Confirmation_Setup.ipynb` to Colab. Section 1 asks for the
matching `Jaden_Confirmation_Data.zip` supplied separately with the notebook.
That bundle contains the public source problem data and prepared candidate pool;
it is not committed here. The notebook verifies its SHA-256 before unpacking.
Run sections 1–6 on CPU, then stop for review. The notebook alone is not self-contained.

## Interpretation

The sample-size tables use assumed true AUROC gains of .02/.03/.05/.10 to plan
detection of a difference from zero, with 80%/90% power and two-sided alpha .05.
These are illustrative scenarios, not approved thresholds or guarantees. Counts
refer to NEW independent eligible mixed problems. Total problems to generate
remain unknown until the team supplies an eligibility-yield assumption.

The CSV names and team message identify checkpoint 6,144. The files do not prove
which folds, seed, configuration, or training-only feature selection produced the
scores. The original source's `base_lr_len` uses observed prefix tokens and
paragraphs, and `hgb` is a gradient-boosted ensemble. Actual export provenance
still needs confirmation. Shared CV training, method selection and a changed
future population limit the variance-based planning approximation.

The exact-token helper pins model/tokenizer revisions, records generated token IDs,
checks the BOS convention, and validates cache/resume provenance. CPU tests do not
certify the historical activation arrays or replace a real GPU smoke test.

Before generating confirmation data, agree on the primary model/baseline, useful
effect, exclusion/eligibility policy, sample target and compute budget. The notebook
keeps GPU and confirmation runs disabled by default. No historical results are
relabelled as confirmed by this PR.
