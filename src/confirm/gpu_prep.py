# confirmation data pipeline around src/recovery/run_rollouts.py. docs/RUN_CONFIRM_GPU.md has the
# order and the exact commands; scripts/confirm_modal.py runs them on Modal.
#   t0       CPU   problems file -> prompt records (one BOS, ids pinned) for the source traces
#                  (run_rollouts round 0, n = samples per problem) and the t0 bank (round 1, K_T0)
#   extract  GPU   sources -> layer-20 states at c - 1 from the exact generated ids, every c the
#                  trace runs past; feats.parquet holds the decoded prefix and its flags
#   cuts     CPU   sources -> one record per LIVE (trace, c): prefix ids, max_tokens = 32768 - c
#   check    CPU   the smoke assertions on whatever the steps above wrote
# every file carries the spec hash; run_rollouts pins the revision with --revision.
import argparse, hashlib, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "recovery"))
import spec  # noqa: E402
import bundle_lib as B  # noqa: E402


def tokenizer():
    if spec.VLLM_ARGS.get("tokenizer_mode") == "mistral":
        return MistralTok()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(spec.MODEL, revision=spec.TOKENIZER_REVISION)


class MistralTok:
    # Mistral profiles: mistral-common's own tokenizer, the one vLLM generates with in mistral mode.
    # transformers 5 hands these repos to MistralCommonBackend, whose chat template drops the default system
    # prompt and whose text round trip is unsafe, so prompt ids come straight from encode_chat_completion,
    # with SYSTEM_PROMPT.txt as the system message split into text + thinking + text (as the model card sends it)
    def __init__(self):
        from huggingface_hub import hf_hub_download
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        get = lambda f: hf_hub_download(spec.MODEL, f, revision=spec.TOKENIZER_REVISION)
        self.mt = MistralTokenizer.from_file(get("tekken.json"))
        self.raw = self.mt.instruct_tokenizer.tokenizer
        self.bos_token_id = self.raw.bos_id
        self.system = open(get("SYSTEM_PROMPT.txt")).read()

    def prompt_ids(self, problem):
        from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
        from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        sp = self.system
        i, j = sp.find("[THINK]"), sp.find("[/THINK]")
        if not 0 <= i < j:
            raise SystemExit("SYSTEM_PROMPT.txt has no [THINK]...[/THINK] block")
        system = SystemMessage(content=[TextChunk(text=sp[:i]), ThinkChunk(thinking=sp[i + 7:j], closed=True),
                                        TextChunk(text=sp[j + 8:])])
        req = ChatCompletionRequest(messages=[system, UserMessage(content=problem + spec.INSTRUCTION)])
        return [int(x) for x in self.mt.encode_chat_completion(req).tokens]

    def decode(self, ids, skip_special_tokens=False):
        # skip_special_tokens=False: every control token kept ([THINK] stays visible for canon).
        # skip_special_tokens=True: vLLM 0.32's own rule for sample text (MistralTokenizer.convert_ids_to_tokens):
        # control tokens dropped except [THINK], [/THINK], [TOOL_CALLS] and [ARGS]
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
        ids = [int(x) for x in ids]
        if skip_special_tokens:
            keep = {self.raw.get_special_token(t) for t in ("[THINK]", "[/THINK]", "[TOOL_CALLS]", "[ARGS]")
                    if self.raw.is_special(t)}
            ids = [i for i in ids if i in keep or not self.raw.is_special(i)]
        return self.raw.decode(ids, special_token_policy=SpecialTokenPolicy.KEEP)


def render(tok, problem):
    # (prompt text, prompt ids). generate.build_prompt, inlined so this file does not import torch / datasets
    # on the CPU steps; Mistral profiles encode with mistral-common and the text is the decoded ids
    if isinstance(tok, MistralTok):
        ids = tok.prompt_ids(problem)
        text = tok.decode(ids)
    else:
        text = tok.apply_chat_template([{"role": "user", "content": problem + spec.INSTRUCTION}],
                                       tokenize=False, add_generation_prompt=True)
        if spec.FORCE_THINK and isinstance(text, str) and not text.rstrip().endswith("<think>"):
            text += "<think>\n"
        ids = None
    if not isinstance(text, str) or (spec.PROMPT_HAS and spec.PROMPT_HAS not in text):
        raise SystemExit(f"prompt rendered as {str(text)[:120]!r}, which lacks {spec.PROMPT_HAS!r}")
    if ids is None:
        ids = tok(text, add_special_tokens=False)["input_ids"]
    if not bos_ok(tok, ids):
        raise SystemExit(f"prompt breaks the BOS rule {spec.BOS!r}: {ids[:3]}")
    return text, ids


def build_prompt(tok, problem):
    return render(tok, problem)[0]


def prompt_ids(tok, problem):
    return render(tok, problem)[1]


def canon(text):
    # the model's own think delimiters written as <think> / </think>; a no-op for the <think> models
    a, b = spec.THINK_TAGS
    return text.replace(b, "</think>").replace(a, "<think>")


def bos_ok(tok, ids):
    # "one": exactly one leading BOS (the confirmation model). "template" (replication): whatever the chat
    # template writes, but never two leading BOS
    bos = tok.bos_token_id
    if spec.BOS == "one":
        return ids[0] == bos and ids[1] != bos
    return bos is None or ids[:2] != [bos, bos]


def read_rows(path):
    # run_rollouts rows; a torn last line is skipped
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_problems(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    need = {"problem_id", "problem", "gold"}
    bad = [r.get("problem_id") for r in rows if not need <= set(r)]
    if bad:
        raise SystemExit(f"problems file rows lack {need}: {bad[:5]}")
    return rows


def cmd_t0(a):
    sha = spec.file_sha(a.problems)
    want = spec.SLOTS["problems_file_sha256"]
    free = a.smoke or spec.REPLICATION  # a replication run is not the confirmation: no slot checks
    if spec.SLOTS.get("status") != "frozen" and not free:
        raise SystemExit(f"team slots are {spec.SLOTS.get('status')}, not frozen")
    if want is None and not free:
        raise SystemExit("spec.SLOTS['problems_file_sha256'] is empty: the confirmation population is not frozen yet")
    if want is not None and sha != want and not free:
        raise SystemExit(f"{a.problems} has sha256 {sha}, the frozen list is {want}")
    probs = load_problems(a.problems)[: (2 if a.smoke else None)]
    tok = tokenizer()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for k, p in enumerate(probs):
            text, ids = render(tok, p["problem"])
            if k == 0:
                print(f"first prompt, {len(ids)} ids {ids[:4]}...: {text[:90]!r} ... {text[-90:]!r}", flush=True)
            f.write(json.dumps({
                "prefix_id": f"t0/{p['problem_id']}", "source": "confirmation", "problem_id": p["problem_id"],
                "trace_id": None, "checkpoint": "t0", "cut_token_idx": 0, "gold_answer": p["gold"],
                "prompt_text": text, "prompt_token_ids": ids, "prefix_gen_token_ids": [],
                "max_tokens": spec.MAX_NEW, "est_cont_tokens": a.est_tokens, "bos": spec.BOS,
                "model": spec.MODEL, "spec_sha256": spec.spec_sha(), "problems_sha256": sha}) + "\n")
    print(f"wrote {len(probs)} t0 records to {a.out} (problems sha256 {sha[:16]}, spec {spec.spec_sha()[:16]})")


def sources_table(path, min_samples=None):
    # one row per source trace from a run_rollouts output of the t0 records, round 0 only
    out = []
    for r in read_rows(path):
        if r["round"] != 0:
            continue
        for s in r["samples"]:
            if "token_ids" not in s:
                raise SystemExit(f"{path}: sources need --save-token-ids full")
            if len(s["token_ids"]) != s["n_gen"]:
                raise SystemExit(f"{r['prefix_id']}#{s['sample_idx']}: {len(s['token_ids'])} ids but n_gen {s['n_gen']}")
            out.append({"trace_id": f"{r['problem_id']}_{s['sample_idx']}", "problem_id": r["problem_id"],
                        "prefix_id": r["prefix_id"], "token_ids": s["token_ids"], "n": s["n_gen"],
                        "finish_reason": s["finish_reason"], "truncated": bool(s["truncated"]),
                        "answer": s["answer"], "correct": int(bool(s["correct"])), "text": s["text"],
                        "seed": s["seed"], "prefix_sha": r["prefix_sha"], "config_hash": r["config_hash"]})
    df = pd.DataFrame(out)
    if df.trace_id.duplicated().any():
        raise SystemExit("duplicate source traces (the file holds two runs of one prefix)")
    return df


def checkpoints(a):
    if a.checkpoints:
        if not (a.smoke or spec.REPLICATION):
            raise SystemExit("--checkpoints is for the smoke test only; the frozen ones are spec.CHECKPOINTS")
        return [int(c) for c in a.checkpoints.split(",")]
    return [spec.Q1_CUT] if spec.REPLICATION else list(spec.CHECKPOINTS)


def replication_pool(src, cuts):
    # replication bank, as the ceiling cohort: every untruncated source past the cut (live or not), in
    # problems whose sources are mixed (some correct, some wrong), with at least 2 such cuts at that c
    ok = src.set_index("trace_id")
    acc = src.groupby("problem_id").correct.agg(["min", "max"])
    mixed = set(acc.index[(acc["min"] == 0) & (acc["max"] == 1)])
    pool = cuts[~ok.loc[cuts.trace_id, "truncated"].values & cuts.problem_id.isin(mixed)]
    n = pool.groupby(["problem_id", "checkpoint"]).trace_id.transform("size")
    return pool[n >= 2]


def cut_table(src, tok, ckpts):
    # every (trace, c) the trace runs past: decoded prefix, live and pre-answer flags
    rows = []
    for t in src.itertuples():
        for c in ckpts:
            if t.n <= c:
                continue
            text = canon(tok.decode(t.token_ids[:c]))
            rows.append({"trace_id": t.trace_id, "problem_id": t.problem_id, "checkpoint": c,
                         **B.cut_rules(text, t.answer, True), "prefix_text": text,
                         "prefix_sha": hashlib.sha256(json.dumps(list(t.token_ids[:c])).encode()).hexdigest()})
    return pd.DataFrame(rows)


def cmd_cuts(a):
    src = sources_table(a.sources)
    t0 = {r["problem_id"]: r for r in map(json.loads, open(a.t0))}
    tok = tokenizer()
    cuts = cut_table(src, tok, checkpoints(a))
    # planning only (run_rollouts' budget gate): mean remaining length of the problem's sources past c
    est = {(p, c): int(np.mean([min(n, spec.MAX_NEW) - c for n in g.n if n > c]) or 1)
           for p, g in src.groupby("problem_id") for c in checkpoints(a) if (g.n > c).any()}
    pool = replication_pool(src, cuts) if spec.REPLICATION and not a.smoke else cuts[cuts.live]
    n = 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for c in pool.itertuples():
            t = src.set_index("trace_id").loc[c.trace_id]
            p = t0[c.problem_id]
            f.write(json.dumps({
                "prefix_id": f"cut/{c.trace_id}/{c.checkpoint}", "source": "confirmation", "problem_id": c.problem_id,
                "trace_id": c.trace_id, "checkpoint": f"p{c.checkpoint}", "cut_token_idx": c.checkpoint,
                "gold_answer": p["gold_answer"], "prompt_text": p["prompt_text"], "prompt_token_ids": p["prompt_token_ids"],
                "prefix_gen_token_ids": list(t.token_ids[: c.checkpoint]), "max_tokens": spec.MAX_NEW - c.checkpoint,
                "est_cont_tokens": est[(c.problem_id, c.checkpoint)], "bos": spec.BOS, "spec_sha256": spec.spec_sha()}) + "\n")
            n += 1
    print(f"{len(src)} sources ({src.correct.mean():.3f} correct), {len(cuts)} cuts past a checkpoint, "
          f"{n} to the bank ({'mixed-problem cohort' if spec.REPLICATION and not a.smoke else 'live'}) -> {a.out}")
    print(cuts.groupby("checkpoint").agg(cuts=("live", "size"), live=("live", "sum"), pre_answer=("pre_answer", "sum")).to_string())
    print(bank_plan(src, pool, est, a.sources))


def bank_plan(src, pool, est, sources_path):
    # the bank's planned tokens (K_PREFIX continuations per cut, each the problem's mean remaining source length
    # past the cut) priced at what the sources cost per token, from their ledger
    tok = sum(spec.K_PREFIX * est[(c.problem_id, c.checkpoint)] for c in pool.itertuples())
    line = (f"bank plan: {len(pool)} cuts in {pool.problem_id.nunique()} problems x {spec.K_PREFIX} = "
            f"~{tok / 1e6:.1f}M tokens")
    ledger = f"{sources_path}.ledger.json"
    if os.path.exists(ledger):
        usd = sum(r["usd"] for runs in json.load(open(ledger)).values() for r in runs)
        per_m = usd / (src.n.sum() / 1e6)
        line += (f"; sources cost ${usd:.2f} for {src.n.sum() / 1e6:.1f}M tokens (${per_m:.2f}/M), "
                 f"so the bank is ~${tok / 1e6 * per_m:.2f} at that rate")
    return line


def decoder():
    # the text decoder without its lm head, on the GPU: model.model of a causal LM, or the language model
    # inside an image-text checkpoint (Ministral 3)
    import torch
    import transformers as T
    cls = T.AutoModelForImageTextToText if spec.MULTIMODAL else T.AutoModelForCausalLM
    model = cls.from_pretrained(spec.MODEL, revision=spec.MODEL_REVISION, dtype=getattr(torch, spec.DTYPE))
    dec = model.model
    if spec.MULTIMODAL:
        dec = getattr(dec, "language_model", None)
        if dec is None:
            raise SystemExit(f"no language_model in {type(model.model).__name__}: "
                             f"{[n for n, _ in model.model.named_children()]}")
    n = getattr(dec.config, "num_hidden_layers", None)
    if n is not None and not spec.LAYER < n:
        raise SystemExit(f"layer {spec.LAYER} but the decoder has {n} layers")
    return dec.to("cuda").eval()


def cmd_extract(a):
    import types
    import torch
    sys.path.insert(0, SRC)
    import extract as E  # chunked forward with a KV cache; E.LAYERS is read by activations()
    E.LAYERS = [spec.LAYER]
    src = sources_table(a.sources)
    t0 = {r["problem_id"]: r for r in map(json.loads, open(a.t0))}
    tok = tokenizer()
    model = types.SimpleNamespace(model=decoder())  # E.activations calls model.model
    ckpts = checkpoints(a)
    cuts = cut_table(src, tok, ckpts)
    os.makedirs(a.out, exist_ok=True)
    acts = np.zeros((len(cuts), spec.HIDDEN), np.float16)
    by_trace = cuts.groupby("trace_id").indices
    for k, (tid, rows) in enumerate(by_trace.items()):
        t = src.set_index("trace_id").loc[tid]
        cs = cuts.checkpoint.values[rows]
        feed = list(t.token_ids[: max(cs)])
        h = E.activations(model, "cuda", t0[t.problem_id]["prompt_token_ids"], feed, [c - 1 for c in cs])
        acts[rows] = h[:, 0, :]
        if k % 50 == 0:
            print(f"{k + 1}/{len(by_trace)} traces", flush=True)
    if not np.isfinite(acts.astype(np.float32)).all():
        raise SystemExit("non-finite states after the cast to float16")
    np.save(os.path.join(a.out, f"acts_L{spec.LAYER}.npy"), acts)
    cuts["row"] = np.arange(len(cuts))
    cuts.to_parquet(os.path.join(a.out, "feats.parquet"))
    json.dump({"spec_sha256": spec.spec_sha(), "model": spec.MODEL, "revision": spec.MODEL_REVISION, "layer": spec.LAYER, "token_index": "c - 1",
               "layer_convention": "hidden_states[l+1]", "dtype": spec.DTYPE, "checkpoints": ckpts,
               "rows": len(cuts), "sources_sha256": spec.file_sha(a.sources),
               "acts_sha256": spec.file_sha(os.path.join(a.out, f"acts_L{spec.LAYER}.npy")),
               "torch": torch.__version__, "transformers": __import__("transformers").__version__},
              open(os.path.join(a.out, "feats.json"), "w"), indent=1)
    print(f"wrote {len(cuts)} rows to {a.out}")


def cmd_check(a):
    # the smoke assertions: one BOS, ids == n_gen, decode round trip, a recomputed state, a bank row per live cut
    ok = []
    src = sources_table(a.sources)
    t0 = {r["problem_id"]: r for r in map(json.loads, open(a.t0))}
    tok = tokenizer()
    ok.append((f"every prompt keeps the BOS rule {spec.BOS!r}", all(bos_ok(tok, r["prompt_token_ids"]) for r in t0.values())))
    ok.append(("token ids length == n_gen for every source", bool((src.token_ids.map(len) == src.n).all())))
    ok.append(("sources differ within a problem", bool(src.groupby("problem_id").text.nunique().min() > 1)))
    dec = [tok.decode(i, skip_special_tokens=True) for i in src.token_ids]
    same = [d == x for d, x in zip(dec, src.text)]
    ok.append((f"source text == decode(ids) ({sum(same)}/{len(same)})", all(same)))
    for d, x in [(d, x) for d, x in zip(dec, src.text) if d != x][:1]:
        k = next((i for i, (u, v) in enumerate(zip(d, x)) if u != v), min(len(d), len(x)))
        print(f"     first mismatch at char {k} (lengths {len(d)} / {len(x)}): decode {d[max(k - 30, 0):k + 30]!r} "
              f"vs vLLM {x[max(k - 30, 0):k + 30]!r}")
    if a.feats:
        f = pd.read_parquet(os.path.join(a.feats, "feats.parquet"))
        acts = np.load(os.path.join(a.feats, f"acts_L{spec.LAYER}.npy"))
        ok.append(("one state per feats row, finite", acts.shape[0] == len(f) and np.isfinite(acts.astype(np.float32)).all()))
        if a.recompute:
            import torch
            dec = decoder()
            r = f.iloc[0]
            t = src.set_index("trace_id").loc[r.trace_id]
            ids = t0[t.problem_id]["prompt_token_ids"] + list(t.token_ids[: r.checkpoint])
            with torch.no_grad():
                h = dec(torch.tensor([ids], device="cuda"), output_hidden_states=True).hidden_states[spec.LAYER + 1][0, -1]
            a1, a2 = h.float().cpu().numpy(), acts[int(r.row)].astype(np.float32)
            cos = float(a1 @ a2 / np.linalg.norm(a1) / np.linalg.norm(a2))
            ok.append((f"stored state == one full forward at c - 1 (cos {cos:.5f})", cos > 0.999))
    if a.bank:
        rows = read_rows(a.bank)
        have = {(r["trace_id"], r["checkpoint"]) for r in rows}
        live = [r for r in map(json.loads, open(a.cuts))]
        ok.append(("a bank row for every live cut", all((r["trace_id"], r["checkpoint"]) in have for r in live)))
        ok.append(("a t0 bank row (round 1) for every problem",
                   {r["problem_id"] for r in rows if r["checkpoint"] == "t0" and r["round"] == 1} == set(t0)))
        ok.append(("bank continuations carry token-id hashes", all("token_ids_sha256" in s for r in rows for s in r["samples"])))
        ok.append(("bank max_tokens == 32768 - cut", all(r["max_tokens"] == spec.MAX_NEW - r["cut_token_idx"] or a.smoke for r in rows)))
    for name, good in ok:
        print(f"{'ok  ' if good else 'FAIL'} {name}")
    if not all(g for _, g in ok):
        raise SystemExit("smoke check failed")
    print("smoke check passed")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("t0")
    p.add_argument("--problems", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--est-tokens", type=int, default=7000,
                   help="planning length of a fresh sample for the budget gate (dev mean is about this); never a cap")
    for name in ("cuts", "extract", "check"):
        p = sub.add_parser(name)
        p.add_argument("--sources", required=True)
        p.add_argument("--t0", required=True)
        p.add_argument("--checkpoints", default="")
        p.add_argument("--smoke", action="store_true")
        if name in ("cuts", "extract"):
            p.add_argument("--out", required=True)
        if name == "check":
            p.add_argument("--feats", default=None)
            p.add_argument("--bank", default=None)
            p.add_argument("--cuts", default=None)
            p.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    {"t0": cmd_t0, "cuts": cmd_cuts, "extract": cmd_extract, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
