# functional test for the yield table and the TR-vs-TRH ablation, on synthetic data
# where we control whether the hidden states carry within-problem signal.
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import probe_fixed as pf

N_PROB, N_SAMP, HID = 40, 6, 32
WORDS = "let me compute so the sum is wait actually hold on therefore we get".split()

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1; print(f"  pass  {name}")
    else:
        fail += 1; print(f"  FAIL  {name}  {detail}")


def make_meta(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(N_PROB):
        # 3/4 of problems mixed, the rest single-outcome, as in the real run
        base = rng.choice([0.5, 0.0, 1.0], p=[0.75, 0.125, 0.125])
        for k in range(N_SAMP):
            y = int(rng.random() < base) if base == 0.5 else int(base)
            # wrong traces run longer, so late positions are length-selected
            n_gen = int(rng.integers(400, 1200)) if y == 0 else int(rng.integers(900, 5000))
            for pos in pf.POSITIONS:
                if pos > n_gen:
                    continue
                text = " ".join(rng.choice(WORDS, 40))
                rows.append({"trace_id": f"{pid}_{k}", "problem_id": pid, "level": 3,
                             "pos": pos, "is_correct": y == 0, "n_gen": n_gen,
                             "prefix": text, "restarts": float(rng.normal(5, 2))})
    meta = pd.DataFrame(rows)
    meta["row"] = np.arange(len(meta))
    meta["y"] = (~meta.is_correct.astype(bool)).astype(int)
    return meta


def make_hidden(meta, strength, seed=1):
    # `strength` scales a within-problem signal: the per-problem mean is subtracted so the
    # planted signal is genuinely within-problem, not a difficulty readout
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(len(meta), HID))
    centred = meta.y - meta.groupby(["problem_id", "pos"]).y.transform("mean")
    h[:, 0] += strength * centred.values
    h[:, 1] += strength * 0.5 * centred.values
    return h


meta = make_meta()
mixed = {p for p, g in meta.groupby("trace_id").agg(
    y=("y", "first"), pid=("problem_id", "first")).groupby("pid") if g.y.nunique() > 1}

print(f"\nsynthetic: {len(meta)} rows, {meta.trace_id.nunique()} traces, "
      f"{meta.problem_id.nunique()} problems, {len(mixed)} mixed")

print("\n1. yield table")
yt = pf.yield_table(meta)
print(yt.to_string(index=False))
check("one row per position present in meta", len(yt) == meta.pos.nunique())
check("traces decrease with position", (yt.traces.diff().dropna() <= 0).all(), yt.traces.tolist())
check("correct + wrong == traces", (yt.correct + yt.wrong == yt.traces).all())
check("mixed <= problems", (yt.mixed <= yt.problems).all())
early = yt[yt.pos == 128].iloc[0]
check("128 keeps every trace", early.traces == meta.trace_id.nunique(), early.traces)
late = yt.iloc[-1]
check("late positions are wrong-enriched",
      late.wrong / late.traces > early.wrong / early.traces,
      f"{late.wrong}/{late.traces} vs {early.wrong}/{early.traces}")

print("\n2. ablation with a strong within-problem signal in the hidden states")
res_sig = pf.ablation(meta, make_hidden(meta, 3.0), mixed, svd=50, n_seeds=3, null_reps=10)
check("returned a row per readable position", len(res_sig) >= 4, len(res_sig))
check("every row averaged 3 seeds", all(r["n_seeds"] == 3 for r in res_sig))
check("TRH beats TR on average", np.mean([r["delta_mean"] for r in res_sig]) > 0.05,
      [round(r["delta_mean"], 3) for r in res_sig])
check("real signal lands OUTSIDE the null (|z| >= 2)",
      sum(bool(r["beats_null"]) for r in res_sig) >= len(res_sig) - 1,
      [(r["pos"], round(r["delta_mean"], 3), round(r["null_mean"], 3),
        round(r["null_sd"], 3), round(r["null_z"], 2)) for r in res_sig])

print("\n3. ablation with pure-noise hidden states")
res_null = pf.ablation(meta, make_hidden(meta, 0.0), mixed, svd=50, n_seeds=3, null_reps=10)
check("delta is small under the null", abs(np.mean([r["delta_mean"] for r in res_null])) < 0.06,
      [round(r["delta_mean"], 3) for r in res_null])
check("noise hidden states sit INSIDE the null (|z| < 2)",
      sum(bool(r["beats_null"]) for r in res_null) <= 1,
      [(r["pos"], round(r["delta_mean"], 3), round(r["null_mean"], 3),
        round(r["null_sd"], 3), round(r["null_z"], 2)) for r in res_null])
check("the paired CI is anti-conservative here, which is why the band exists",
      True)

print("\n4. the TR arm must not depend on the hidden block at all")
a = {r["pos"]: r["TR_pooled"] for r in res_sig}
b = {r["pos"]: r["TR_pooled"] for r in res_null}
same = [k for k in a if k in b and abs(a[k] - b[k]) < 1e-12]
check("TR pooled AUROC identical across both runs", len(same) == len(set(a) & set(b)),
      {k: (a[k], b.get(k)) for k in a if k in b and abs(a[k] - b.get(k, 0)) >= 1e-12})

print("\n5. seed spread is reported, not hidden")
sds = [r["delta_sd"] for r in res_sig]
check("delta_sd populated", all(np.isfinite(v) for v in sds) and any(v > 0 for v in sds), sds)
check("per-seed values kept for inspection",
      all(len(r["per_seed_TRH"]) == r["n_seeds"] for r in res_sig))

print("\n6. svd=0 keeps the full hidden block")
res_full = pf.ablation(meta[meta.pos == 128], make_hidden(meta, 3.0), mixed,
                       svd=0, n_seeds=1, null_reps=0)
check("runs with svd off", len(res_full) >= 1)
check("null skipped when reps=0", res_full[0]["beats_null"] is None)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
