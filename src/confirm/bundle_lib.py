# shared by bundle.py (confirmation GPU outputs) and dev_bundle.py (development data): answer
# clustering for the vote, live / pre-answer flags per cut, rollout-bank summaries, applying the frozen
# monitors, and writing the bundle analyze.py reads.
import json, os, re, sys
import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "probe_cmp"))
import spec  # noqa: E402
from ceiling import answer_flags, norm, same  # noqa: E402


def cluster_keys(answers, correct):
    # one key per answer within a problem: "C" for every correct answer (all equal the gold),
    # "W<k>" for wrong answers grouped by equivalence (math_verify, string fallback), None for none
    reps, out = [], []
    for a, ok in zip(answers, correct):
        if ok:
            out.append("C")
        elif a is None or (isinstance(a, float) and np.isnan(a)) or str(a).strip() == "":
            out.append(None)
        else:
            k = next((i for i, r in enumerate(reps) if same(a, r)), None)
            if k is None:
                reps.append(a)
                k = len(reps) - 1
            out.append(f"W{k}")
    return out


def stated(prefix_text, final_answer):
    # the trace's final answer value written right after an answer cue, on the same line: "the answer
    # is 6", "Final answer: 3/4". value matching as ceiling.answer_flags (normalized, standalone)
    if final_answer is None or (isinstance(final_answer, float) and np.isnan(final_answer)) or not norm(final_answer):
        return False
    text = re.sub(r"\\(left|right)", "", prefix_text).replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    cue = "|".join(re.escape(c) for c in spec.ANSWER_CUES)
    a = re.escape(norm(final_answer))
    return bool(re.search(rf"(?i:{cue})[^\n]{{0,{spec.ANSWER_CUE_WINDOW}}}?(?<![\w.]){a}(?!\w|\.\d)", text))


def cut_rules(prefix_text, final_answer, still_generating):
    # live: generating after the cut and thinking not closed. the three pre-answer rules, each within live:
    #   pre_answer (primary, "stated"): no \boxed and no stated answer
    #   pre_answer_strict: no \boxed and the final answer value nowhere (ceiling.answer_flags)
    #   pre_answer_think_boxed: no \boxed
    live = bool(still_generating) and spec.THINK_END not in prefix_text
    boxed = "\\boxed" in prefix_text
    _, leaked = answer_flags(prefix_text, final_answer)
    return {"live": live, "pre_answer": live and not boxed and not stated(prefix_text, final_answer),
            "pre_answer_strict": live and not leaked, "pre_answer_think_boxed": live and not boxed}


def cut_flags(prefix_text, final_answer, still_generating):
    r = cut_rules(prefix_text, final_answer, still_generating)
    return r["live"], r["pre_answer"]


PRE_COLS = ("pre_answer", "pre_answer_strict", "pre_answer_think_boxed")


def bank_table(samples):
    # samples: one row per continuation with problem_id, trace_id ("" for t0), checkpoint (int, 0 = t0),
    # sample_idx, correct, n_gen (already capped at the cut's budget). halves alternate in sample order
    s = samples.sort_values(["problem_id", "trace_id", "checkpoint", "sample_idx"]).copy()
    s["half"] = s.groupby(["trace_id", "checkpoint", "problem_id"]).cumcount() % 2
    g = s.groupby(["problem_id", "trace_id", "checkpoint"])
    out = g.agg(k=("correct", "size"), r=("correct", "mean"), L=("n_gen", "mean")).reset_index()
    h = s.groupby(["problem_id", "trace_id", "checkpoint", "half"]).agg(r=("correct", "mean"), L=("n_gen", "mean")).unstack("half")
    for col in ("r", "L"):
        for k in (0, 1):
            out[f"{col}_h{k}"] = h[(col, k)].reindex(pd.MultiIndex.from_frame(out[["problem_id", "trace_id", "checkpoint"]])).values
    return out


def load_monitors(mon_dir):
    # every frozen monitor, each checked against the manifest's sha256 before it is used
    man = json.load(open(os.path.join(mon_dir, "manifest.json")))
    got = man.get("monitor_spec_sha256")
    if got != spec.monitor_sha():
        raise SystemExit(f"monitors were fit under monitor spec {str(got)[:16]}, this is {spec.monitor_sha()[:16]}; refit")
    out = {}
    for key, info in man["monitors"].items():
        path = os.path.join(mon_dir, info["file"])
        if spec.file_sha(path) != info["sha256"]:
            raise SystemExit(f"{path} does not match its manifest hash")
        out[(info["monitor"], info["checkpoint"])] = (joblib.load(path), info)
    return out, man


def score_rows(monitors, c, trace_ids, X=None, texts=None, tokens=None):
    # long table of every frozen monitor at checkpoint c over the given rows
    rows = []
    for (name, ck), (mon, info) in monitors.items():
        if ck != c:
            continue
        layer = spec.MONITORS[name][2]
        s = mon.score(X if layer is not None else None, texts, tokens)
        rows.append(pd.DataFrame({"trace_id": list(trace_ids), "checkpoint": c, "monitor": name, "score": s,
                                  "threshold": info["threshold"]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["trace_id", "checkpoint", "monitor", "score", "threshold"])


def write(out, kind, provenance, sources, cuts, bank, votes, scores):
    os.makedirs(out, exist_ok=True)
    for name, df in (("sources", sources), ("cuts", cuts), ("bank", bank), ("t0_votes", votes), ("scores", scores)):
        df.reset_index(drop=True).to_parquet(os.path.join(out, f"{name}.parquet"))
    meta = {"kind": kind, "spec_sha256": spec.spec_sha(), "checkpoints": sorted(int(c) for c in cuts.checkpoint.unique()),
            "bank_checkpoints": sorted(int(c) for c in bank.checkpoint.unique()),
            "problems": int(sources.problem_id.nunique()), "attempts": int(len(sources)), "provenance": provenance}
    meta["files"] = {f: spec.file_sha(os.path.join(out, f)) for f in sorted(os.listdir(out)) if f.endswith(".parquet")}
    json.dump(meta, open(os.path.join(out, "bundle.json"), "w"), indent=1, default=str)
    return meta
