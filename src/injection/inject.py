# injection pilot, gate 2: matched benign vs consequential edits of correct arithmetic in tag-B
# traces, scored teacher-forced so the text after the edit is identical in both arms (no
# regenerated continuation, so no text confound). a site is a correct claim "a op b = d" inside
# <think>, from slip_scan's claims_<tag>.jsonl, whose written d is an integer of >= 2 digits.
# the edit changes one non-leading digit of d (slip_scan's single_digit, the commonest natural
# slip): one digit token swapped for another, every other token unchanged, checked per site.
#   consequential: d is written again later and nowhere before (slip_scan's load_bearing),
#                  with the first reuse within --horizon tokens of the edit
#   benign:        d is written nowhere else in the trace, answer included
# a pair is one site of each kind from the same trace, same digit count, nearest in tokens.
# the clean run and both edited runs of a pair share one sequence length, so an identity edit
# must score 0; score.py runs that check on the first --identity pairs.
# what this measures is detection, not propagation: the text after the edit stays the original,
# so later lines still carry the correct value. the reuse metrics (p of the edited digit where the
# value is written again) are a one-token lean toward the edit, not evidence the model would carry
# it forward, repair it, or answer wrongly; that needs rollouts from the edited prefix.
#   python src/injection/inject.py select --tag B   # CPU + tokenizer: sites_<tag>.jsonl
#   python src/injection/inject.py score --tag B    # GPU (a T4 will do), resumable: scores_<tag>.jsonl
#   python src/injection/inject.py report --tag B   # CPU: inject_sites_<tag>.csv, inject_summary_<tag>.json
import argparse, bisect, csv, hashlib, json, math, os, random, re, sys, time
from collections import Counter
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slip_scan import MAX_NEW, MODEL, THINK_END, spans  # noqa: E402

KINDS = ("consequential", "benign")
LAYERS = (12, 16, 20, 24)  # decoder blocks; the probes use 12/16/20


def read_jsonl(path):
    # duplicated from recovery/common.py, which pulls in torch. a copy taken while score was
    # mid-line ends in a torn line; only that one is skipped
    if not os.path.exists(path):
        return []
    lines = [line for line in open(path) if line.strip()]
    out = [json.loads(line) for line in lines[:-1]]
    if lines:
        try:
            out.append(json.loads(lines[-1]))
        except json.JSONDecodeError:
            if lines[-1].endswith("\n"):
                raise
    return out


def seed_of(*parts):
    # python hash() is salted, so sha256
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def make_edit(literal, seed, text):
    # change one non-leading digit (no leading zero, digit count kept); the new value must
    # appear nowhere in the trace, so a later occurrence of it cannot be mistaken for propagation
    rng = random.Random(seed)
    for _ in range(20):
        i = rng.randrange(1, len(literal))
        new = literal[:i] + rng.choice([d for d in "0123456789" if d != literal[i]]) + literal[i + 1:]
        if not spans(text, Fraction(new), 0, len(text)):
            return i, new
    return None


def tok_at(ends, char):
    # index of the token holding char; ends[i] is token i's end offset
    return min(bisect.bisect_right(ends, char), len(ends) - 1)


def digit_token(offsets, char):
    # the token that is exactly the one digit at char, else None
    i = tok_at([b for _, b in offsets], char)
    return i if tuple(offsets[i]) == (char, char + 1) else None


def candidates(trace, claims, ids, offsets, encode, a):
    # every correct claim of the trace that qualifies as a site, with its edit and token positions
    text = trace["text"]
    out, why = [], Counter()
    for c in claims:
        if c["status"] != "ok" or c["context"] != "main":
            continue
        lit = c["true_value"]
        if not re.fullmatch(r"\d+", lit) or len(lit) < a.min_digits:
            why["not a plain integer of enough digits"] += 1
            continue
        ve = c["value_end_char"] + 1
        vs = ve - len(lit)
        if text[vs:ve] != lit:
            why["written differently from its value"] += 1  # 132.0, 0132
            continue
        v = Fraction(lit)
        if spans(text, v, c["char_offset"], vs):
            why["value also on the left side"] += 1
            continue
        before, after = spans(text, v, 0, c["char_offset"]), spans(text, v, ve, len(text))
        kind = "consequential" if after and not before else "benign" if not (after or before) else None
        if kind is None:
            why["written before the claim"] += 1
            continue
        edit = make_edit(lit, seed_of(a.seed, trace["trace_id"], vs), text)
        if edit is None:
            why["no unused edited value"] += 1
            continue
        i, new = edit
        edit_tok = digit_token(offsets, vs + i)
        if edit_tok is None or edit_tok < a.min_tok or edit_tok > a.max_tok:
            why["edit token missing or outside [min_tok, max_tok]"] += 1
            continue
        e_ids, _ = encode(text[:vs] + new + text[ve:])
        if len(e_ids) != len(ids) or [k for k, (x, y) in enumerate(zip(ids, e_ids)) if x != y] != [edit_tok]:
            why["edit changes more than one token"] += 1
            continue
        reuse = [r for r in (digit_token(offsets, s + i) for s, _ in after) if r is not None]
        if kind == "consequential":
            reuse = [r for r in reuse if r > edit_tok]
            if not reuse or reuse[0] - edit_tok > a.horizon:
                why["first reuse beyond the horizon"] += 1
                continue
        out.append({"kind": kind, "char_offset": c["char_offset"], "statement": c["statement"], "value": lit,
                    "edited_value": new, "digit_index": i, "value_char": vs, "edit_tok": edit_tok,
                    "new_token": e_ids[edit_tok], "reuse_toks": reuse, "n_after": len(after)})
    return out, why


def pair_up(cands, max_gap, cap):
    # greedy nearest pairs, same digit count, each site used once, at most cap pairs per trace
    cons = [c for c in cands if c["kind"] == "consequential"]
    ben = [c for c in cands if c["kind"] == "benign"]
    edges = sorted((abs(x["edit_tok"] - y["edit_tok"]), i, j) for i, x in enumerate(cons)
                   for j, y in enumerate(ben)
                   if len(x["value"]) == len(y["value"]) and abs(x["edit_tok"] - y["edit_tok"]) <= max_gap)
    used_c, used_b, pairs = set(), set(), []
    for _, i, j in edges:
        if i in used_c or j in used_b:
            continue
        used_c.add(i)
        used_b.add(j)
        pairs.append({"consequential": cons[i], "benign": ben[j]})
        if len(pairs) == cap:
            break
    return sorted(pairs, key=lambda p: p["consequential"]["edit_tok"])


def select(traces, claims_by_trace, encode, a):
    pairs, why, n_cand, excluded = [], Counter(), Counter(), Counter()
    for line, t in enumerate(traces):
        cl = claims_by_trace.get(t["trace_id"])
        if not cl:
            continue
        if t["trunc"] or t["n_gen"] >= MAX_NEW or THINK_END not in t["text"]:
            excluded["truncated or no </think>"] += 1
            continue
        ids, offsets = encode(t["text"])
        if t["n_gen"] != len(ids) + 1:
            excluded["retokenization drift"] += 1  # vLLM counts the EOS
            continue
        cands, w = candidates(t, cl, ids, offsets, encode, a)
        why.update(w)
        n_cand.update(c["kind"] for c in cands)
        for k, p in enumerate(pair_up(cands, a.max_gap, a.max_per_trace)):
            pairs.append({"pair_id": f"{t['trace_id']}:{k}", "trace_id": t["trace_id"], "line": line,
                          "problem_id": t["problem_id"], "trace_label": "correct" if t["is_correct"] else "wrong",
                          "n_tokens": len(ids), "gap": abs(p["consequential"]["edit_tok"] - p["benign"]["edit_tok"]),
                          "sites": p})
    random.Random(a.seed).shuffle(pairs)
    return pairs[: a.n], {"excluded": dict(excluded), "candidates": dict(n_cand), "pairs_available": len(pairs),
                          "skipped": dict(why.most_common())}


def run_select(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    def encode(text):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    traces = [json.loads(line) for line in open(a.traces or f"data/traces_{a.tag}.jsonl")]
    by = {}
    for line in open(a.claims or os.path.join(a.out_dir, f"claims_{a.tag}.jsonl")):
        c = json.loads(line)
        by.setdefault(c["trace_id"], []).append(c)
    pairs, stats = select(traces, by, encode, a)
    path = os.path.join(a.out_dir, f"sites_{a.tag}.jsonl")
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    json.dump({**stats, "written": len(pairs), "args": vars(a)},
              open(os.path.join(a.out_dir, f"sites_summary_{a.tag}.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"wrote {len(pairs)} pairs ({Counter(p['trace_label'] for p in pairs)}) from "
          f"{len({p['trace_id'] for p in pairs})} traces: {path}")
    for p in pairs[:3]:
        text = next(t["text"] for t in traces if t["trace_id"] == p["trace_id"])
        for k in KINDS:
            s = p["sites"][k]
            ctx = text[max(0, s["value_char"] - 100):s["value_char"]] + " >>>" + s["value"] + " -> " + \
                s["edited_value"] + "<<< " + text[s["value_char"] + len(s["value"]):][:100]
            print(f"[{p['pair_id']} {k} token {s['edit_tok']}, reuse {s['reuse_toks'][:3]}] "
                  f"{ctx.replace(chr(10), ' ⏎ ')}\n")


# ---- score (GPU) ----

def site_range(s, n_ids, a):
    # trace-token indices [lo, hi) scored for a site: --pre before the edit, --horizon after
    return max(1, s["edit_tok"] - a.pre), min(s["edit_tok"] + 1 + a.horizon, n_ids)


def run_score(a):
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recovery"))
    from common import FIRST_BATCH_LINES, build_prompt, encode_trace, load_tokenizer
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer()
    bos = tok.bos_token_id
    traces = [json.loads(line) for line in open(a.traces or f"data/traces_{a.tag}.jsonl")]
    pairs = read_jsonl(os.path.join(a.out_dir, f"sites_{a.tag}.jsonl"))
    out_path = os.path.join(a.out_dir, f"scores_{a.tag}.jsonl")
    done = {r["pair_id"] for r in read_jsonl(out_path)}
    todo = [p for p in pairs if p["pair_id"] not in done][: a.limit or None]
    print(f"{len(pairs)} pairs, {len(done)} scored, {len(todo)} to go", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
    head = model.get_output_embeddings()
    grab = {}
    for l in a.layers:
        model.model.layers[l].register_forward_hook(
            lambda m, i, o, l=l: grab.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0]))

    @torch.no_grad()
    def run(seq):
        grab.clear()
        h = model.model(input_ids=torch.tensor([seq], device="cuda"), use_cache=False).last_hidden_state[0]
        return h, dict(grab)

    def prompt(t, line):
        # the source prompt carried one BOS on lines 0-479 and two after (bos_check on Colab)
        one = tok(build_prompt(tok, t["problem"], t.get("structured", False)), add_special_tokens=False)["input_ids"]
        assert one[0] == bos and one[1] != bos, one[:3]
        return one if line < FIRST_BATCH_LINES else [bos] + one

    @torch.no_grad()
    def compare(clean, edit, n_p, ids, e_ids, lo, hi):
        # per trace token j in [lo, hi): surprisal of j in each run, KL(clean || edited) of the
        # distribution that predicts j, and per-layer cosine distance of the state after reading j
        q = torch.arange(n_p + lo - 1, n_p + hi - 1, device="cuda")
        lc = torch.log_softmax(head(clean[0][q]).float(), -1)
        le = torch.log_softmax(head(edit[0][q]).float(), -1)
        tc = torch.tensor(ids[lo:hi], device="cuda")
        te = torch.tensor(e_ids[lo:hi], device="cuda")
        r = {"lo": lo,
             "nll_clean": (-lc.gather(1, tc[:, None])[:, 0]).tolist(),
             "nll_edit": (-le.gather(1, te[:, None])[:, 0]).tolist(),
             "kl": (lc.exp() * (lc - le)).sum(-1).clamp_min(0).tolist(),
             "cos": {str(l): (1 - torch.cosine_similarity(clean[1][l][n_p + lo:n_p + hi].float(),
                                                          edit[1][l][n_p + lo:n_p + hi].float(), dim=-1)).tolist()
                     for l in a.layers}}
        return r, lc, le

    rnd = lambda xs: [round(x, 5) for x in xs]
    with open(out_path, "a") as f:
        for n, p in enumerate(todo):
            t0 = time.time()
            line = p["line"]
            t = traces[line]
            assert t["trace_id"] == p["trace_id"], (line, t["trace_id"], p["trace_id"])
            ids, _ = encode_trace(tok, t["text"])
            assert len(ids) == p["n_tokens"], (p["pair_id"], len(ids), p["n_tokens"])
            p_ids = prompt(t, line)
            n_p = len(p_ids)
            end = max(site_range(s, len(ids), a)[1] for s in p["sites"].values())
            clean = run(p_ids + ids[:end])
            row = {"pair_id": p["pair_id"], "trace_id": p["trace_id"], "trace_label": p["trace_label"],
                   "n_prompt": n_p, "seq_len": n_p + end, "sites": {}}
            for k in KINDS + (("identity",) if len(done) + n < a.identity else ()):
                s = p["sites"]["consequential" if k == "identity" else k]
                e_ids = list(ids)
                if k != "identity":
                    assert ids[s["edit_tok"]] != s["new_token"]
                    e_ids[s["edit_tok"]] = s["new_token"]
                edit = run(p_ids + e_ids[:end])
                lo, hi = site_range(s, len(ids), a)
                r, lc, le = compare(clean, edit, n_p, ids, e_ids, lo, hi)
                if k == "identity":
                    row["identity"] = {"kl_max": max(r["kl"]),
                                       "nll_absmax": max(abs(x - y) for x, y in zip(r["nll_clean"], r["nll_edit"])),
                                       "cos_max": max(max(v) for v in r["cos"].values())}
                    continue
                # at each reuse of the value: probability of the edited digit, clean vs edited context
                reuse = [j for j in s["reuse_toks"] if lo <= j < hi]
                pn = lambda lp: lp[[j - lo for j in reuse], s["new_token"]].exp().tolist() if reuse else []
                r.update({"nll_clean": rnd(r["nll_clean"]), "nll_edit": rnd(r["nll_edit"]),
                          "kl": rnd(r["kl"]), "cos": {l: rnd(v) for l, v in r["cos"].items()},
                          "edit_tok": s["edit_tok"], "reuse": reuse,
                          "p_new_clean": rnd(pn(lc)), "p_new_edit": rnd(pn(le))})
                row["sites"][k] = r
                del edit
            row["seconds"] = round(time.time() - t0, 2)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{len(done) + n + 1}/{len(pairs)} {p['pair_id']} len {row['seq_len']} {row['seconds']}s"
                  + (f" identity {row['identity']}" if "identity" in row else ""), flush=True)


# ---- report (CPU) ----

def site_metrics(r, partner=None, pre=64):
    # scalars for one scored site. split: the first reuse for a consequential site; for a
    # benign one, the same offset from its edit as its partner's first reuse
    e, lo = r["edit_tok"], r["lo"]
    at = lambda xs, j: xs[j - lo]
    n = len(r["kl"])
    after = range(e + 1, lo + n)
    if r["reuse"]:
        split = r["reuse"][0]
    elif partner and partner["reuse"]:
        split = e + partner["reuse"][0] - partner["edit_tok"]
    else:
        split = None
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    seg = lambda xs, a_, b_: [at(xs, j) for j in range(max(a_, lo), min(b_, lo + n))]
    win = range(max(lo, e - pre), min(lo + n, e + pre + 1))
    surprise_e = [at(r["nll_edit"], j) for j in win]
    surprise_c = [at(r["nll_clean"], j) for j in win]
    rank = lambda xs, v: 1 + sum(x > v for x in xs)
    m = {"write_nll_clean": at(r["nll_clean"], e), "write_nll_edit": at(r["nll_edit"], e),
         "write_rank_edit": rank(surprise_e, at(r["nll_edit"], e)),
         "write_rank_clean": rank(surprise_c, at(r["nll_clean"], e)), "window": len(win),
         "kl_after": mean([at(r["kl"], j) for j in after]),
         "dnll_after": mean([at(r["nll_edit"], j) - at(r["nll_clean"], j) for j in after]),
         "kl_before_split": mean(seg(r["kl"], e + 1, split)) if split else float("nan"),
         "kl_from_split": mean(seg(r["kl"], split, lo + n)) if split else float("nan")}
    m["write_hit"] = m["write_rank_edit"] == 1
    for l, v in r["cos"].items():
        m[f"cos{l}_after"] = mean(seg(v, e, lo + n))
        m[f"cos{l}_before_split"] = mean(seg(v, e, split)) if split else float("nan")
        m[f"cos{l}_from_split"] = mean(seg(v, split, lo + n)) if split else float("nan")
    if r["reuse"]:
        j = r["reuse"][0]
        m["reuse_dnll"] = at(r["nll_edit"], j) - at(r["nll_clean"], j)
        m["reuse_p_new_clean"], m["reuse_p_new_edit"] = r["p_new_clean"][0], r["p_new_edit"][0]
        # copied: the edited digit is now more likely than the original one at the reuse
        m["reuse_copied"] = r["p_new_edit"][0] > math.exp(-at(r["nll_edit"], j))
    return m


def paired(x, y, rng, n_boot=2000):
    # consequential - benign within pairs: mean, bootstrap 95% CI, sign-flip permutation p
    d = [a - b for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if not d:
        return {"n": 0}
    k = len(d)
    mean = sum(d) / k
    boots = sorted(sum(rng.choice(d) for _ in range(k)) / k for _ in range(n_boot))
    flips = sum(abs(sum(v if rng.random() < .5 else -v for v in d)) / k >= abs(mean) - 1e-12 for _ in range(n_boot))
    return {"n": k, "mean_diff": mean, "ci95": [boots[int(.025 * n_boot)], boots[int(.975 * n_boot) - 1]],
            "p_signflip": (flips + 1) / (n_boot + 1), "frac_positive": sum(v > 0 for v in d) / k}


def describe(xs):
    xs = sorted(v for v in xs if not (isinstance(v, float) and math.isnan(v)))
    if not xs:
        return {"n": 0}
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return {"n": len(xs), "min": xs[0], "p10": q(.1), "median": q(.5), "p90": q(.9), "max": xs[-1],
            "mean": sum(xs) / len(xs)}


def report(rows, pre=64, seed=0):
    rng = random.Random(seed)
    per = []
    for r in rows:
        c, b = r["sites"]["consequential"], r["sites"]["benign"]
        per.append({"pair_id": r["pair_id"], "trace_label": r["trace_label"], "seconds": r["seconds"],
                    "consequential": site_metrics(c, None, pre), "benign": site_metrics(b, c, pre)})
    keys = [k for k in per[0]["benign"] if k not in ("window",)] if per else []
    cols = {f"{kind}.{k}": describe([float(p[kind][k]) for p in per if k in p[kind]])
            for kind in KINDS for k in per[0][kind] if k != "window"} if per else {}
    tests = {k: paired([p["consequential"][k] for p in per], [p["benign"][k] for p in per], rng)
             for k in keys if not k.startswith(("write_hit", "write_rank"))}
    hit = {kind: sum(p[kind]["write_hit"] for p in per) / len(per) for kind in KINDS} if per else {}
    chance = sum(1 / p[kind]["window"] for p in per for kind in KINDS) / (2 * len(per)) if per else None
    clean_hit = sum(p[k]["write_rank_clean"] == 1 for p in per for k in KINDS) / (2 * len(per)) if per else None
    ident = [r["identity"] for r in rows if "identity" in r]
    return per, {"pairs": len(per), "by_label": dict(Counter(p["trace_label"] for p in per)),
                 "seconds_per_pair": describe([p["seconds"] for p in per]),
                 "identity_checks": len(ident),
                 "identity_max": {k: max(i[k] for i in ident) for k in ident[0]} if ident else None,
                 "write_point_localization": {"hit_at_1": hit, "clean_run_hit_at_1": clean_hit,
                                              "chance": chance},
                 "copied_at_reuse": sum(bool(p["consequential"].get("reuse_copied")) for p in per) / len(per)
                 if per else None,
                 "paired_consequential_minus_benign": tests, "columns": cols}


def run_report(a):
    rows = read_jsonl(os.path.join(a.out_dir, f"scores_{a.tag}.jsonl"))
    per, s = report(rows, a.pre, a.seed)
    csv_path = os.path.join(a.out_dir, f"inject_sites_{a.tag}.csv")
    with open(csv_path, "w", newline="") as f:
        names = sorted({k for p in per for kind in KINDS for k in p[kind]})
        w = csv.writer(f)
        w.writerow(["pair_id", "trace_label", "kind"] + names)
        for p in per:
            for kind in KINDS:
                w.writerow([p["pair_id"], p["trace_label"], kind] + [p[kind].get(k, "") for k in names])
    json.dump(s, open(os.path.join(a.out_dir, f"inject_summary_{a.tag}.json"), "w"), indent=2)
    print(f"{s['pairs']} pairs {s['by_label']}; seconds per pair {s['seconds_per_pair']}")
    if s["identity_max"] is None:
        print("identity check: none run (score with --identity > 0)")
    else:
        ok = s["identity_max"]["kl_max"] <= 1e-6 and s["identity_max"]["cos_max"] <= 1e-4
        print(f"identity check over {s['identity_checks']} pairs: {s['identity_max']} -> {'PASS' if ok else 'FAIL'}")
    w = s["write_point_localization"]
    print(f"edited digit is the most surprising token within +-{a.pre}: {w['hit_at_1']}; the original digit "
          f"in the clean run: {w['clean_run_hit_at_1']}; chance {w['chance']}")
    print(f"consequential: edited digit preferred over the original at the first reuse in "
          f"{s['copied_at_reuse']} of pairs")
    print("\nconsequential - benign, paired (mean, 95% CI, sign-flip p, share of pairs > 0):")
    for k, v in s["paired_consequential_minus_benign"].items():
        if v.get("n"):
            print(f"  {k:<22} n {v['n']:>4}  {v['mean_diff']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  "
                  f"p {v['p_signflip']:.4f}  {v['frac_positive']:.2f}")
    print("\nper column (min / median / max):")
    for k, v in s["columns"].items():
        if v.get("n"):
            print(f"  {k:<36} {v['min']:.4g} / {v['median']:.4g} / {v['max']:.4g}  (n {v['n']})")
    print(f"wrote {csv_path} and inject_summary_{a.tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "score", "report"])
    ap.add_argument("--tag", default="B")
    ap.add_argument("--traces", default=None, help="default data/traces_<tag>.jsonl")
    ap.add_argument("--claims", default=None, help="default <out-dir>/claims_<tag>.jsonl (slip_scan.py)")
    ap.add_argument("--out-dir", default="data/injection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=120, help="pairs written by select")
    ap.add_argument("--min-digits", type=int, default=2)
    ap.add_argument("--min-tok", type=int, default=128, help="earliest edit token in the trace")
    ap.add_argument("--max-tok", type=int, default=16384, help="latest edit token (bounds sequence length)")
    ap.add_argument("--max-gap", type=int, default=2048, help="max token distance between a pair's edits")
    ap.add_argument("--max-per-trace", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=256, help="tokens scored after the edit")
    ap.add_argument("--pre", type=int, default=64, help="tokens scored before the edit")
    ap.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=list(LAYERS))
    ap.add_argument("--identity", type=int, default=5, help="pairs that also score an identity edit")
    ap.add_argument("--limit", type=int, default=0, help="score at most this many pairs this run")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    {"select": run_select, "score": run_score, "report": run_report}[a.stage](a)


if __name__ == "__main__":
    main()
