# registry-driven loading: which checkpoints exist comes from registry.json and the files on
# disk, never from constants here. every pair is checked against traces_B.jsonl on load,
# because meta and acts are joined by row index alone (the cache/offset bug was silent).
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "registry.json")
MAX_NEW = 32768  # tag-B generation cap; n_gen at the cap means truncated
END = "end"


class BadData(RuntimeError):
    """A pair fails an alignment check; nothing downstream may use it."""


def load_registry(path=REGISTRY):
    reg = json.load(open(path))
    names = [s["name"] for s in reg["sources"]]
    if len(set(names)) != len(names):
        raise BadData(f"duplicate source names in {path}: {names}")
    return reg


def load_traces(data_dir, reg):
    rows = [json.loads(line) for line in open(os.path.join(data_dir, reg["traces"]))]
    df = pd.DataFrame(rows).set_index("trace_id", drop=False).rename_axis(None)
    if df.index.duplicated().any():
        raise BadData("duplicate trace_id in traces file")
    df["eligible"] = ~df.trunc.astype(bool) & (df.n_gen < MAX_NEW)
    df["y"] = (~df.is_correct.astype(bool)).astype(int)  # 1 = wrong: every score is P(wrong)
    return df


def paragraphs(text):
    # \n\n paragraphs inside <think>, plus one for the answer once </think> has appeared
    cut = text.find("</think>")
    think = text if cut < 0 else text[:cut]
    return sum(1 for c in think.split("\n\n") if c.strip()) + (cut >= 0)


def checkpoint_frame(src, meta, traces):
    # one row per (trace, checkpoint) with the columns everything downstream uses
    if src["kind"] == "position":
        pos = meta[src["pos_col"]].astype(int).values
        if src.get("name_col"):
            # relative labels (e.g. fractions): place each on the token axis by its median pos
            label = meta[src["name_col"]].astype(str)
            order = label.map(pd.Series(pos, index=label.index).groupby(label).median()).values
        else:
            label = meta[src["pos_col"]].astype(str)
            order = pos.astype(float)
        ckpt = label.values
    elif src["kind"] == "end":
        pos = traces.loc[meta.trace_id, "n_gen"].astype(int).values
        order = np.full(len(meta), np.inf)
        ckpt = np.full(len(meta), END, dtype=object)
    else:
        raise BadData(f"{src['name']}: unknown kind {src['kind']!r}")
    return pd.DataFrame({"trace_id": meta.trace_id.values, "ckpt": ckpt, "order": order,
                         "tokens": pos, "src": src["name"], "row": np.arange(len(meta))})


def check_pair(src, meta, acts, traces, hidden_dim):
    # (name, ok, detail) per check; the loader raises on any failure
    out = []

    def add(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    add("acts is (rows, layers, dim)", acts.ndim == 3 and acts.shape[1:] == (len(src["layers"]), hidden_dim),
        f"{acts.shape} vs (*, {len(src['layers'])}, {hidden_dim})")
    add("acts rows == meta rows", acts.shape[0] == len(meta), f"{acts.shape[0]} vs {len(meta)}")
    if "row" in meta:
        add("meta.row == 0..n-1", (meta.row.values == np.arange(len(meta))).all())
    known = meta.trace_id.isin(traces.index)
    add("every trace_id in traces", known.all(), f"{int((~known).sum())} unknown")
    if not known.all():
        return out
    t = traces.loc[meta.trace_id]
    add("problem_id matches traces", (meta.problem_id.values == t.problem_id.values).all())
    add("is_correct matches traces",
        (meta.is_correct.astype(bool).values == t.is_correct.astype(bool).values).all(),
        f"{int((meta.is_correct.astype(bool).values != t.is_correct.astype(bool).values).sum())} differ")
    if "n_gen" in meta:
        add("n_gen matches traces", (meta.n_gen.values == t.n_gen.values).all())
    if "trunc" in meta:
        add("trunc matches traces", (meta.trunc.astype(bool).values == t.trunc.astype(bool).values).all())
    else:
        add("trunc taken from traces (not in meta)", True)

    if src["kind"] == "end":
        add("one row per trace", not meta.trace_id.duplicated().any())
        if {"step_idx", "n_steps"} <= set(meta):
            add("row is the last step", (meta.step_idx == meta.n_steps - 1).all())
    else:
        pc, nc = src["pos_col"], src.get("name_col")
        add("no duplicate (trace, checkpoint)", not meta.duplicated(["trace_id", nc or pc]).any())
        over = meta[pc].values > t.n_gen.values
        add("pos <= n_gen", not over.any(), f"{int(over.sum())} rows past the trace end")
        if nc is None:
            # absolute positions: the extractor takes every position a trace reaches, so a
            # trace's positions are a prefix of the sorted list; a gap means dropped rows
            grid = np.sort(meta[pc].unique())
            n_per = meta.groupby("trace_id")[pc].agg(["max", "size"])
            gaps = n_per["size"].values != np.searchsorted(grid, n_per["max"].values, side="right")
            add("positions per trace have no gaps", not gaps.any(), f"{int(gaps.sum())} traces")
        else:
            # relative labels: pos must rise with the label inside each trace
            lab = pd.to_numeric(meta[nc], errors="coerce")
            ok = meta.assign(_l=lab).sort_values(["trace_id", "_l"]).groupby("trace_id")[pc].apply(
                lambda v: v.is_monotonic_increasing)
            add("pos increases with the checkpoint label", lab.notna().all() and ok.all(),
                f"{int((~ok).sum())} traces")
            # relative labels exist for every trace, so a trace missing one lost rows
            n_lab = meta.groupby("trace_id")[nc].nunique()
            add("every trace has every label", (n_lab == meta[nc].nunique()).all(),
                f"{int((n_lab != meta[nc].nunique()).sum())} traces")

    kind, _, col = src["prefix"].partition(":")
    if kind == "meta":
        if col not in meta:
            add(f"prefix column {col!r} present", False)
        else:
            # a cut inside a multi-byte char decodes to U+FFFD
            lit = [txt.startswith(p) or txt.startswith(p.rstrip("�"))
                   for p, txt in zip(meta[col].values, t.text.values)]
            add("stored prefix is a prefix of the trace text", all(lit), f"{len(lit) - sum(lit)} rows differ")
            if src["kind"] == "position":
                # a prefix column holding more (or less) than pos tokens of text would leak
                # the future into len/tfidf; tag-B prefixes run 1.5-5 chars per token
                cpt = meta[col].str.len().values / np.maximum(meta[src["pos_col"]].values, 1)
                bad = (cpt < 1) | (cpt > 8)
                add("prefix length fits pos tokens (1-8 chars/token)", not bad.any(), f"{int(bad.sum())} rows")
    elif kind == "trace":
        add("prefix 'trace' only used for the end", src["kind"] == "end")

    rng = np.random.default_rng(0)
    probe = np.sort(rng.choice(len(meta), min(len(meta), 512), replace=False))
    add("acts finite (512-row sample)", np.isfinite(np.asarray(acts[probe], np.float32)).all())
    return out


def layer_norms(acts, n=300, seed=0):
    # median |h| per layer slot: a slot mapped to a different hidden_states index shows up
    # as a jump (the last slot is post-RMSNorm, so its norm drops sharply)
    rows = np.sort(np.random.default_rng(seed).choice(len(acts), min(n, len(acts)), replace=False))
    return np.median(np.linalg.norm(np.asarray(acts[rows], np.float32), axis=-1), axis=0)


class Store:
    """Every registered pair that is present, checked, and merged into one checkpoint table."""

    def __init__(self, data_dir, reg=None, names=None, log=print):
        self.reg = reg or load_registry()
        self.data_dir = data_dir
        self.traces = load_traces(data_dir, self.reg)
        self.sources, self.acts, frames = {}, {}, []
        conventions = set()
        for src in self.reg["sources"]:
            if names and src["name"] not in names:
                continue
            a, m = (os.path.join(data_dir, src[k]) for k in ("acts", "meta"))
            if not (os.path.exists(a) and os.path.exists(m)):
                log(f"  source {src['name']}: {src['acts']} / {src['meta']} not present, skipped")
                continue
            meta = pd.read_parquet(m)
            acts = np.load(a, mmap_mode="r")
            failed = [(n, d) for n, ok, d in check_pair(src, meta, acts, self.traces, self.reg["hidden_dim"])
                      if not ok]
            if failed:
                raise BadData(f"{src['name']} failed alignment: " + "; ".join(f"{n} ({d})" for n, d in failed))
            conventions.add(src["layer_convention"])
            self.sources[src["name"]], self.acts[src["name"]] = (src, meta), acts
            frames.append(checkpoint_frame(src, meta, self.traces))
            log(f"  source {src['name']}: {acts.shape} {acts.dtype}, layers {src['layers']}, "
                f"{meta.trace_id.nunique()} traces, alignment ok")
        if not frames:
            raise BadData(f"no registered source present in {data_dir}")
        if len(conventions) > 1:
            raise BadData(f"sources disagree on layer convention: {conventions}")
        self.convention = conventions.pop()
        tab = pd.concat(frames, ignore_index=True)
        # a checkpoint present in two sources: the earlier registry entry wins
        dup = tab.duplicated(["trace_id", "ckpt"])
        if dup.any():
            log(f"  {int(dup.sum())} (trace, checkpoint) rows appear in two sources; keeping the first")
        tab = tab[~dup]
        t = self.traces.loc[tab.trace_id]
        tab = tab.assign(problem_id=t.problem_id.values, y=t.y.values, eligible=t.eligible.values)
        self.table = tab.reset_index(drop=True)
        self.ckpts = (self.table.groupby("ckpt").order.first().sort_values().index.tolist())

    def layers(self):
        # layers present in every source
        sets = [set(src["layers"]) for src, _ in self.sources.values()]
        return sorted(set.intersection(*sets))

    def rows(self, ckpt, trace_ids):
        sub = self.table[self.table.ckpt == ckpt].set_index("trace_id").loc[list(trace_ids)]
        return sub

    def hidden(self, ckpt, trace_ids, layer):
        sub = self.rows(ckpt, trace_ids)
        out = np.empty((len(sub), self.reg["hidden_dim"]), np.float32)
        for name, g in sub.reset_index().groupby("src"):
            src, _ = self.sources[name]
            li = src["layers"].index(layer)
            order = np.argsort(g.row.values)  # mmap reads in file order
            out[g.index.values[order]] = self.acts[name][g.row.values[order], li, :]
        return out

    def text(self, ckpt, trace_ids):
        sub = self.rows(ckpt, trace_ids).reset_index()
        out = []
        tok = None
        for r in sub.itertuples():
            src, meta = self.sources[r.src]
            kind, _, col = src["prefix"].partition(":")
            if kind == "meta":
                out.append(meta[col].iat[r.row])
            elif kind == "trace":
                out.append(self.traces.at[r.trace_id, "text"])
            elif kind == "tokenize":
                if tok is None:
                    from transformers import AutoTokenizer
                    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
                ids = tok(self.traces.at[r.trace_id, "text"], add_special_tokens=False)["input_ids"]
                out.append(tok.decode(ids[: r.tokens]))
            else:
                raise BadData(f"{src['name']}: unknown prefix rule {src['prefix']!r}")
        return out

    def len_features(self, ckpt, trace_ids, texts=None):
        # tokens and paragraphs observed up to the checkpoint, never the final length
        sub = self.rows(ckpt, trace_ids)
        texts = self.text(ckpt, trace_ids) if texts is None else texts
        para = np.array([paragraphs(x) for x in texts], float)
        return np.log1p(np.column_stack([sub.tokens.values.astype(float), para]))


def mixed(df):
    # problems with at least one correct and one wrong trace in df
    g = df.groupby("problem_id").y.agg(["min", "max"])
    return set(g.index[g["min"] != g["max"]])


def population(store, ckpts):
    # fixed set: eligible traces present at every checkpoint in ckpts, then mixed problems only
    tab = store.table[store.table.eligible & store.table.ckpt.isin(ckpts)]
    n = tab.groupby("trace_id").ckpt.nunique()
    keep = n.index[n == len(ckpts)]
    per = store.traces.loc[list(keep), ["trace_id", "problem_id", "y"]].reset_index(drop=True)
    m = mixed(per)
    per = per[per.problem_id.isin(m)].sort_values(["problem_id", "trace_id"])
    return per.reset_index(drop=True)


def varying_counts(store):
    # secondary: every eligible trace a checkpoint reaches, mixed filter per checkpoint
    rows = []
    for c in store.ckpts:
        per = store.traces.loc[store.table[(store.table.ckpt == c) & store.table.eligible].trace_id,
                               ["problem_id", "y"]]
        m = mixed(per)
        per = per[per.problem_id.isin(m)]
        rows.append({"checkpoint": c, "traces": len(per), "mixed_problems": len(m),
                     "wrong": int(per.y.sum()), "correct": int((per.y == 0).sum())})
    return pd.DataFrame(rows)


def candidate_sets(store):
    # nested checkpoint sets: all positions up to p, with and without the end. the fixed
    # population is set by the deepest checkpoint, so this is the whole menu of choices
    pos = [c for c in store.ckpts if c != END]
    sets = []
    for i in range(len(pos)):
        base = pos[: i + 1]
        sets.append(base)
        if END in store.ckpts:
            sets.append(base + [END])
    if END in store.ckpts:
        sets.append([END])
    out = []
    for s in sets:
        p = population(store, s)
        out.append({"checkpoints": s, "traces": len(p), "mixed_problems": p.problem_id.nunique(),
                    "pairs": pair_total(p)})
    return out


def pair_total(per):
    g = per.groupby("problem_id").y.agg(["sum", "size"])
    return int((g["sum"] * (g["size"] - g["sum"])).sum())
