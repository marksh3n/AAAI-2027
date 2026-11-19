# does the trace's commitment to an answer, as written in the prefix, explain the prefix-fixed outcome
# variance D? capture and WP AUROC as in determination.py, on the same cohort and halves.
# gold-free features (monitors), from explicitly stated candidates ("answer is", "final answer", "answer
# should/would/must be", \boxed{}) in the prefix only:
#   stated       any candidate stated
#   n_distinct   distinct candidate values
#   flips        changes between consecutive candidates
#   since_last   share of the prefix written after the last candidate (1 if none)
#   free_all     ridge on the four, fit on other problems' r_hat (grouped CV), like probe_r
# diagnostics that use the gold answer or the future (never monitors):
#   gold_written   the gold value already appears as a standalone value in the prefix
#   final_written  the trace's own final answer already appears in the prefix (future)
#   last_correct   the last stated candidate equals gold
#   python src/analysis/commitment.py --rollouts data/recovery_ceiling --traces <traces_B.jsonl> \
#       --meta <meta_B.parquet> --split results/split.json --half explore --out <dir> [--audit 30]
import argparse, json, os, re, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "probe_cmp"))
import stats as S  # noqa: E402
from ceiling import answer_flags, norm  # noqa: E402
from cheap_rollouts import load  # noqa: E402
from determination import halves, inner_pick, sums, total_sums  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

CUE = re.compile(r"(?i)(?:final answer|answer (?:is|should be|would be|must be)|answer:)\s*(?:is|=|:)?\s*|\\boxed\{")
FREE = ("stated", "n_distinct", "flips", "since_last", "free_all")
ORACLE = ("gold_written", "final_written", "last_correct")


def value_at(text, i, boxed):
    # the candidate value starting at i: brace content for \boxed, else $...$, \(...\), or a token run
    if boxed:
        d, j = 1, i
        while j < len(text) and d:
            d += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        return text[i:j - 1] if d == 0 else None
    s = text[i:i + 60]
    for o, c in (("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
        if s.startswith(o):
            k = s.find(c, len(o))
            return s[len(o):k] if k > 0 else None
    m = re.match(r"[^\s,;*]+", s)  # "**Final Answer**" headings are markdown, not values
    v = m.group(0).rstrip(".?!:") if m else None
    # a value has a digit, a LaTeX command, or is a short symbol; words like "the" are not answers
    return v if v and (re.search(r"[\d\\]", v) or len(v) <= 2) else None


def candidates(text):
    out = []
    for m in CUE.finditer(text):
        v = value_at(text, m.end(), m.group(0).startswith("\\boxed"))
        if v and norm(v):
            out.append((m.start(), norm(v), text[max(0, m.start() - 40): m.end() + 40]))
    return out


def written(prefix, value):
    # the value appears as a standalone value in the prefix; markers removed so "committed" can't set it
    return float(bool(value) and answer_flags(prefix.replace("</think>", "").replace("\\boxed", ""), value)[1])


def feats(prefix, gold, final):
    c = candidates(prefix)
    vals = [v for _, v, _ in c]
    return {"stated": float(bool(c)), "n_distinct": float(len(set(vals))),
            "flips": float(sum(a != b for a, b in zip(vals, vals[1:]))),
            "since_last": (len(prefix) - c[-1][0]) / max(len(prefix), 1) if c else 1.0,
            "gold_written": written(prefix, gold), "final_written": written(prefix, final),
            "last_correct": float(bool(c) and norm(gold) == vals[-1])}


def free_all(F, r, g):
    # grouped ridge on the gold-free features, fit to within-problem-centered r_hat of other problems
    X = F - pd.DataFrame(F).groupby(g).transform("mean").values
    rc = r - pd.Series(r).groupby(g).transform("mean").values
    out = np.zeros(len(r))
    mse = lambda m, X, y: -np.mean((m.predict(X) - y) ** 2)
    for tr, te in GroupKFold(5).split(X, rc, g):
        sc = StandardScaler().fit(X[tr])
        a = inner_pick(sc.transform(X[tr]), rc[tr], g[tr], lambda a: Ridge(alpha=a), (1e-2, 1e-1, 1, 1e1, 1e2), mse)
        out[te] = Ridge(alpha=a).fit(sc.transform(X[tr]), rc[tr]).predict(sc.transform(X[te]))
    return out


def run(t, tr, meta, n_boot=2000, seed=0, log=print):
    rows, m = [], meta.set_index(["trace_id", "pos"])
    for cut in sorted(t.cut.unique()):
        tc = t[t.cut == cut].reset_index(drop=True)
        tc = tc[tc.groupby("problem_id").trace_id.transform("size") >= 2].reset_index(drop=True)
        g = tc.problem_id.values
        A, B, r = halves(tc)
        fix = sums(g, A, B)
        pids = fix.index.values
        fix = fix.loc[pids].values
        D = fix.sum() / total_sums(tc).loc[pids].values.sum()
        draws = S.problem_draws(len(pids), n_boot, seed)
        f = pd.DataFrame([feats(m.at[(i, cut), "prefix"], tr.at[i, "gold"], tr.at[i, "answer"]) for i in tc.trace_id])
        f["free_all"] = free_all(f[["stated", "n_distinct", "flips", "since_last"]].values, r, g)
        pairs = S.Pairs((~tr.loc[tc.trace_id, "is_correct"].astype(bool)).astype(int).values, g)
        log(f"{cut}: D {D:.3f}, stated {f.stated.mean():.3f}, gold_written {f.gold_written.mean():.3f}, "
            f"final_written {f.final_written.mean():.3f}")
        for name in FREE + ORACLE:
            x = f[name].values
            sa, sb, sf = (sums(g, x, h).loc[pids].values for h in (A, B, x))
            ok = sf.sum() > 0
            cap = sa.sum() * sb.sum() / (sf.sum() * fix.sum()) if ok else np.nan
            capb = (draws @ sa) * (draws @ sb) / ((draws @ sf) * (draws @ fix)) if ok else np.full(n_boot, np.nan)
            au = pairs.auroc(x)["weighted"]
            rows.append({"cut": int(cut), "feature": name, "kind": "monitor" if name in FREE else "oracle",
                         "prevalence": float((x != 0).mean()), "capture": cap, "lo": S.ci(capb)[0], "hi": S.ci(capb)[1],
                         "wp_auroc_y": au, "D": D, "problems": len(pids), "traces": len(tc)})
            log(f"  {name:14s} capture {cap:.3f} [{S.ci(capb)[0]:.3f}, {S.ci(capb)[1]:.3f}], WP AUROC {au:.3f}")
    return pd.DataFrame(rows)


def audit(t, meta, k, seed=1):
    # k random stated candidates with context, for a hand check of the parser
    m = meta.set_index(["trace_id", "pos"])
    found = []
    for i, cut in t[["trace_id", "cut"]].itertuples(index=False):
        for pos, v, ctx in candidates(m.at[(i, cut), "prefix"]):
            found.append((i, cut, v, ctx.replace("\n", " ")))
    rng = np.random.default_rng(seed)
    for j in rng.choice(len(found), min(k, len(found)), replace=False):
        print(f"[{found[j][0]} @{found[j][1]}] value={found[j][2]!r} | {found[j][3]}")


def main():
    ap = argparse.ArgumentParser()
    for k in ("rollouts", "traces", "meta", "split", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--half", choices=["explore", "holdout"], required=True)
    ap.add_argument("--audit", type=int, default=0)
    a = ap.parse_args()
    t = load(a.rollouts)
    t = t[t.problem_id.isin(set(json.load(open(a.split))[a.half]))]
    meta = pd.read_parquet(a.meta, columns=["trace_id", "pos", "prefix"])
    if a.audit:
        audit(t, meta, a.audit)
        return
    tr = pd.DataFrame(json.loads(line) for line in open(a.traces)).set_index("trace_id")
    res = run(t, tr, meta)
    res["half"] = a.half
    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, f"x3_{a.half}.csv"), index=False)


if __name__ == "__main__":
    main()
