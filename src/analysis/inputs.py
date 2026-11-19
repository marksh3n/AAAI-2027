# data layout and loaders shared by the per-model analysis scripts. E0_DATA (default: data/ in the repo):
#   hf/traces_B.jsonl, hf/meta_B.parquet, hf/acts_B.npy   DeepSeek source traces, prefix rows, layer states
#   recovery_ceiling                                       DeepSeek re-runs, 8 per cut at 512 / 2,048 / 4,096
#   rollouts_limit.jsonl                                   DeepSeek re-runs of the limit-hit sources at 4,096
#   qwen3/, ministral/                                     scripts/replication_modal.py outputs:
#                                                          sources.jsonl, bank.jsonl, feats/
#   problems_l45.jsonl                                     Ministral's problems (MATH levels 4-5)
#   confidence/{ds,qwen3,ministral}/                       src/replication/confidence.py outputs
import json, os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "src", "probe_cmp")]
import cheap_rollouts as CR  # noqa: E402
import determination as DT  # noqa: E402
import stats as S  # noqa: E402

DATA = os.environ.get("E0_DATA", os.path.join(ROOT, "data"))
DS_TRACES = f"{DATA}/hf/traces_B.jsonl"
DS_META = f"{DATA}/hf/meta_B.parquet"
DS_ACTS = f"{DATA}/hf/acts_B.npy"
DS_RERUNS = f"{DATA}/recovery_ceiling"
DS_LIMIT_RERUNS = f"{DATA}/rollouts_limit.jsonl"
MINISTRAL_PROBLEMS = f"{DATA}/problems_l45.jsonl"
LIMIT = 32768
CUT = 4096
LABELS = {"deepseek": "DeepSeek-R1-Distill-Qwen-1.5B", "qwen3": "Qwen3-1.7B", "ministral": "Ministral-3-3B-Reasoning"}
LAYER = {"qwen3": 20, "ministral": 19}
RESTART = re.compile(r"\b(wait|hold on|let me re|actually)\b", re.I)  # taxonomy.RESTART


def sources(path):
    # replication sources.jsonl -> one row per source trace (round 0 only)
    rows = []
    for line in open(path):
        r = json.loads(line)
        if r.get("round", 0) != 0:
            continue
        for s in r["samples"]:
            rows.append({"trace_id": f"{r['problem_id']}_{s['sample_idx']}", "problem_id": str(r["problem_id"]),
                         "is_correct": bool(s["correct"]), "n_gen": s["n_gen"], "truncated": bool(s["truncated"])})
    return pd.DataFrame(rows).set_index("trace_id")


def ds_sources():
    tr = pd.DataFrame(json.loads(x) for x in open(DS_TRACES)).set_index("trace_id")
    tr["problem_id"] = tr.problem_id.astype(str)
    tr["truncated"] = tr.n_gen >= LIMIT
    return tr


def levels(model):
    path = MINISTRAL_PROBLEMS if model == "ministral" else DS_TRACES
    return {str(r["problem_id"]): str(r["level"]) for r in map(json.loads, open(path))}


def bank(model, limit_hits="drop", cuts=(CUT,)):
    # re-runs, one row per cut. limit_hits: "drop" keeps finished sources only, "wrong" also keeps sources that
    # hit the token limit, counted as wrong. problems keep >= 2 cuts at each cut either way
    tr = ds_sources() if model == "deepseek" else sources(f"{DATA}/{model}/sources.jsonl")
    if model == "deepseek":
        paths = [DS_RERUNS] + ([DS_LIMIT_RERUNS] if limit_hits == "wrong" else [])
        t = pd.concat([CR.load(p) for p in paths], ignore_index=True)
    else:
        t = CR.load(f"{DATA}/{model}/bank.jsonl")
    t["problem_id"] = t.problem_id.astype(str)
    t = t[t.cut.isin(cuts)]
    if limit_hits == "drop":
        t = t[~tr.loc[t.trace_id, "truncated"].values]
    elif not tr.loc[t.trace_id, "truncated"].any():
        raise SystemExit(f"{model}: no re-runs of limit-hit sources in its bank; limit_hits=wrong needs them")
    t = t[t.groupby(["problem_id", "cut"]).trace_id.transform("size") >= 2].reset_index(drop=True)
    return tr, t


def states(model):
    # prefix rows (meta) and their layer states, aligned by meta.row
    if model == "deepseek":
        meta = pd.read_parquet(DS_META, columns=["trace_id", "problem_id", "pos", "prefix", "restarts", "row"])
        return meta, np.load(DS_ACTS, mmap_mode="r")
    f = pd.read_parquet(f"{DATA}/{model}/feats/feats.parquet")
    if f.prefix_text.str.contains(r"\[/?THINK\]").any():
        raise SystemExit(f"{model}: prefixes still hold [THINK] tags; gpu_prep.canon writes <think> / </think>")
    meta = pd.DataFrame({"trace_id": f.trace_id, "problem_id": f.problem_id.astype(str), "pos": f.checkpoint,
                         "prefix": f.prefix_text, "row": f.row, "pre_answer": f.pre_answer_think_boxed,
                         "restarts": [len(RESTART.findall(x)) / c * 1000 for x, c in zip(f.prefix_text, f.checkpoint)]})
    DT.LAYER_SLOT = 0
    return meta, np.load(f"{DATA}/{model}/feats/acts_L{LAYER[model]}.npy")[:, None, :]


def both_outcomes(t, tr):
    # the cohort rule: problems whose cohort traces include a correct and a wrong source
    n = t.assign(c=tr.loc[t.trace_id, "is_correct"].values).groupby("problem_id").c.nunique()
    return t[t.problem_id.isin(set(n.index[n == 2]))].reset_index(drop=True)


def d_ci(t, n_boot=2000, seed=0):
    # outcome determination D with a problem-bootstrap interval, no readouts
    t = t[t.groupby("problem_id").trace_id.transform("size") >= 2].reset_index(drop=True)
    if t.problem_id.nunique() < 2:
        return {"D": np.nan, "lo": np.nan, "hi": np.nan, "problems": t.problem_id.nunique(), "cuts": len(t)}
    A, B, _ = DT.halves(t)
    s_fix, s_tot = DT.sums(t.problem_id.values, A, B), DT.total_sums(t)
    pids = s_fix.index.values
    fix, tot = s_fix.loc[pids].values, s_tot.loc[pids].values
    draws = S.problem_draws(len(pids), n_boot, seed)
    lo, hi = S.ci((draws @ fix) / (draws @ tot))
    return {"D": fix.sum() / tot.sum(), "lo": lo, "hi": hi, "problems": len(pids), "cuts": len(t)}
