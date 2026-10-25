# development data -> the bundle analyze.py reads, for the dry run. CPU, needs the frozen monitors.
#   sources   every tag-B trace of the problems with rollouts (the 87 ceiling problems), truncated
#             ones included as wrong; n = vLLM's n_gen
#   cuts      spec.CHECKPOINTS; a trace is live at c when meta_B has its row at c (it runs that far)
#             and the stored prefix holds no </think>
#   bank      recovery_ceiling: t0 and the 512 / 2048 / 4096 cuts, the 32768 source budget applied
#             as oracle.py does. dev has no rollouts at 6144 and none outside the ceiling cohort
#   scores    the frozen monitors applied to meta_B / acts_B rows; a monitor's own training traces get
#             their out-of-fold score from fit_monitors.py (dev_oof.parquet), so no score is in-sample
# dev token ids were never saved: everything is keyed by the retokenized text, as probe_cmp is.
#   python src/confirm/dev_bundle.py --data data --rollouts <recovery_ceiling> \
#       --monitors data/confirm/monitors --out data/confirm/dev_bundle
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "probe_cmp"))
import spec  # noqa: E402
import bundle_lib as B  # noqa: E402
import data as D  # noqa: E402
from oracle import load_samples  # noqa: E402

CUT = {"t0": 0, "p512": 512, "p2048": 2048, "p4096": 4096}


def t0_answers(path):
    # answers of the t0 continuations, which load_samples does not keep
    last = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("checkpoint") != "t0":
            continue
        for s in r["samples"]:
            last[(r["problem_id"], r["round"], s["sample_idx"])] = s.get("answer")
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--registry", default=D.REGISTRY)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--monitors", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    samples, torn = load_samples(a.rollouts)
    samples = samples[samples.checkpoint.isin(CUT)].copy()
    samples["checkpoint"] = samples.checkpoint.map(CUT).astype(int)
    samples["trace_id"] = samples.trace_id.fillna("").astype(str)
    samples.loc[samples.checkpoint == 0, "trace_id"] = ""
    problems = sorted(samples.problem_id.unique())
    print(f"rollouts: {len(samples)} continuations, {len(problems)} problems, torn lines {torn}")

    store = D.Store(a.data, reg=D.load_registry(a.registry), names=["B_fixed"])
    tr = store.traces[store.traces.problem_id.isin(problems)].copy()
    tr["correct"] = (tr.is_correct.astype(bool) & ~tr.trunc.astype(bool)).astype(int)
    tr["answer"] = tr.answer.where(tr.answer.notna(), None)

    # answer keys over each problem's sources and t0 votes together, so equal answers share a key
    ans = t0_answers(a.rollouts)
    t0 = samples[samples.checkpoint == 0].copy()
    t0["answer"] = [ans.get((p, r, i)) for p, r, i in zip(t0.problem_id, t0["round"], t0.sample_idx)]
    src_keys, vote_rows = {}, []
    for p in problems:
        s = tr[tr.problem_id == p]
        v = t0[t0.problem_id == p].sort_values(["round", "sample_idx"])
        keys = B.cluster_keys(list(s.answer) + list(v.answer), list(s.correct.astype(bool)) + list(v.correct))
        src_keys.update(dict(zip(s.trace_id, keys[: len(s)])))
        vote_rows += [{"problem_id": p, "sample_idx": i, "answer_key": k, "correct": int(c), "n_gen": int(n)}
                      for i, (k, c, n) in enumerate(zip(keys[len(s):], v.correct, v.n_gen))]
    sources = pd.DataFrame({"trace_id": tr.trace_id.values, "problem_id": tr.problem_id.values,
                            "n": np.minimum(tr.n_gen.values, spec.MAX_NEW), "correct": tr.correct.values,
                            "answer_key": [src_keys[t] for t in tr.trace_id]})

    monitors, man = B.load_monitors(a.monitors)
    cut_rows, score_parts = [], []
    for c in spec.CHECKPOINTS:
        ck = str(c)
        if ck not in store.ckpts:
            sys.exit(f"checkpoint {c} not in the dev activations")
        tab = store.table[(store.table.ckpt == ck) & store.table.trace_id.isin(set(tr.trace_id))]
        ids = tab.trace_id.tolist()
        texts = store.text(ck, ids)
        for tid, txt in zip(ids, texts):
            live, pre = B.cut_flags(txt, tr.at[tid, "answer"], True)
            cut_rows.append({"trace_id": tid, "checkpoint": c, "live": live, "pre_answer": pre})
        X = store.hidden(ck, ids, spec.LAYER)
        score_parts.append(B.score_rows(monitors, c, ids, X, texts, tab.tokens.values.astype(float)))
    scores = pd.concat(score_parts, ignore_index=True)
    oof = pd.read_parquet(os.path.join(a.monitors, "dev_oof.parquet"))
    key = ["trace_id", "checkpoint", "monitor"]
    scores = scores.merge(oof[key + ["p_oof"]], how="left", on=key)
    n_oof = int(scores.p_oof.notna().sum())
    scores["score"] = scores.p_oof.fillna(scores.score)
    scores = scores.drop(columns="p_oof")
    print(f"scores: {len(scores)} rows, {n_oof} out-of-fold (training traces), the rest out-of-sample")
    cuts = pd.DataFrame(cut_rows)
    seen = set(zip(cuts.trace_id, cuts.checkpoint))
    cuts = pd.concat([cuts, pd.DataFrame([{"trace_id": t, "checkpoint": c, "live": False, "pre_answer": False}
                                          for t in tr.trace_id for c in spec.CHECKPOINTS if (t, c) not in seen])],
                     ignore_index=True)
    bank = B.bank_table(samples[["problem_id", "trace_id", "checkpoint", "sample_idx", "round", "correct", "n_gen"]]
                        .assign(sample_idx=lambda d: d["round"] * 1000 + d.sample_idx))
    prov = {"traces": spec.file_sha(os.path.join(a.data, store.reg["traces"])), "rollouts": spec.file_sha(a.rollouts),
            "monitors_manifest": spec.file_sha(os.path.join(a.monitors, "manifest.json")),
            "note": "dev: retokenized text, sources sampled by vLLM 0.28.0 (pilot, one BOS) or unrecorded (main, two BOS), "
                    "rollouts by vLLM 0.10.2 with two BOS; training traces scored out of fold"}
    meta = B.write(a.out, "development", prov, sources, cuts, bank, pd.DataFrame(vote_rows), scores)
    print(json.dumps({k: v for k, v in meta.items() if k != "files"}, indent=1, default=str))


if __name__ == "__main__":
    main()
