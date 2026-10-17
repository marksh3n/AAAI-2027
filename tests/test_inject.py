# inject: matched edit selection (kinds, single-token edits, reuse positions, pairing) and the
# report's metrics and paired tests, on synthetic traces and scores. CPU, no torch, no tokenizer:
# a stand-in tokenizer splits digits one per token, as the model's pre-tokenizer does.
import json, math, os, re, sys, types
from fractions import Fraction

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "injection"))
import inject as I  # noqa: E402
import slip_scan as S  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(extra)) if extra else ''}")
    if not cond:
        fails.append(name)


VOCAB = {}


def encode(text):
    # digits one per token, words, runs of spaces, single punctuation: offsets are exact
    ids, offs = [], []
    for m in re.finditer(r"\d|[A-Za-z]+|\s+|[^\sA-Za-z\d]", text):
        ids.append(VOCAB.setdefault(m.group(0), len(VOCAB)))
        offs.append((m.start(), m.end()))
    return ids, offs


def claims_of(trace):
    # what slip_scan writes to claims_<tag>.jsonl for the evaluated claims of one trace
    text = trace["text"]
    cl = S.find_claims(text[:text.find(S.THINK_END)])
    S.tag_context(cl)
    return [{"trace_id": trace["trace_id"], "status": c["status"], "char_offset": c["start"], "eq_char": c["eq"],
             "value_end_char": c["end"] - 1, "statement": c["statement"], "true_value": S.fmt(c["true"]),
             "written_value": S.fmt(c["written"]), "context": c["context"]}
            for c in cl if not c["status"].startswith("skip:")]


def trace(tid, text, correct=True):
    ids, _ = encode(text)
    return {"trace_id": tid, "problem_id": tid.split("_")[0], "text": text, "trunc": False,
            "n_gen": len(ids) + 1, "is_correct": correct, "problem": "p"}


ARGS = types.SimpleNamespace(seed=0, min_digits=2, min_tok=0, max_tok=10 ** 6, horizon=60, max_gap=10 ** 6,
                             max_per_trace=2, n=100)

print("1. spans: positions of a value, whole numbers only")
text = "132 and 1320 and 132, then 2132 and 132."
check("spans finds the three whole 132s", S.spans(text, Fraction(132), 0, len(text)) == [(0, 3), (17, 20), (36, 39)],
      S.spans(text, Fraction(132), 0, len(text)))
check("occurrences is the span count", S.occurrences(text, Fraction(132), 0, len(text)) == 3)
check("spans honours lo/hi", S.spans(text, Fraction(132), 10, len(text)) == [(17, 20), (36, 39)])

print("\n2. make_edit: one non-leading digit, value absent from the trace, deterministic")
for lit in ["12", "132", "90817"]:
    for seed in range(30):
        i, new = I.make_edit(lit, seed, "nothing here")
        ok = (len(new) == len(lit) and i >= 1 and new[0] == lit[0] and new[i] != lit[i]
              and new[:i] + new[i + 1:] == lit[:i] + lit[i + 1:])
        if not ok:
            break
    check(f"edit of {lit}", ok, (i, new))
check("same seed, same edit", I.make_edit("4567", 7, "") == I.make_edit("4567", 7, ""))
check("no edit when every candidate value is already in the text",
      all(I.make_edit("13", s, "10 11 12 14 15 16 17 18 19") is None for s in range(5)))

print("\n3. select: kinds, edit token, reuse positions, pairing")
T1 = ("<think>\nFirst, 12 * 11 = 132. Next 100 + 47 = 147 is a side note. Then 132 + 5 = 137 so "
      "the total is 137. Also 60 * 7 = 420.\n</think>\nThe answer is 137.")
T2 = "<think>\nWe know 25 already. So 20 + 5 = 25 and 25 * 2 = 50 here.\n</think>\nAnswer 50."
T3 = "<think>\nSo 30 + 3 = 33 and 33 + 1 = 34.\n</think>\nAnswer 34."
traces = [trace("1_0", T1), trace("2_0", T2, correct=False), trace("3_0", T3)]
traces[2]["n_gen"] += 5  # retokenization drift
by = {t["trace_id"]: claims_of(t) for t in traces}
pairs, stats = I.select(traces, by, encode, ARGS)
check("drifted trace excluded", stats["excluded"].get("retokenization drift") == 1, stats["excluded"])
check("two pairs, both from trace 1_0", len(pairs) == 2 and {q["trace_id"] for q in pairs} == {"1_0"},
      [q["pair_id"] for q in pairs])
p = next(q for q in pairs if q["sites"]["consequential"]["value"] == "132")
con, ben = p["sites"]["consequential"], p["sites"]["benign"]
check("consequential sites are 132 and 137 (both reused later)",
      sorted(q["sites"]["consequential"]["value"] for q in pairs) == ["132", "137"])
check("benign sites are 147 and 420 (never written again)",
      sorted(q["sites"]["benign"]["value"] for q in pairs) == ["147", "420"])
check("132 paired with its nearest benign, 147", ben["value"] == "147", ben["value"])
ids, offs = encode(T1)
d = con["value_char"] + con["digit_index"]
check("edit token is the changed digit of the claim's value", tuple(offs[con["edit_tok"]]) == (d, d + 1))
edited = T1[:con["value_char"]] + con["edited_value"] + T1[con["value_char"] + 3:]
e_ids, _ = encode(edited)
check("edited trace differs at exactly the edit token", [k for k, (x, y) in enumerate(zip(ids, e_ids)) if x != y]
      == [con["edit_tok"]] and e_ids[con["edit_tok"]] == con["new_token"])
reuse_char = T1.index("132 + 5") + con["digit_index"]
check("reuse token is the same digit of the later 132", con["reuse_toks"] == [I.digit_token(offs, reuse_char)],
      (con["reuse_toks"], I.digit_token(offs, reuse_char)))
check("value written before its claim is never a site (25 in trace 2_0)",
      stats["skipped"].get("written before the claim", 0) >= 1, stats["skipped"])
check("p['line'] indexes the traces file", traces[p["line"]]["trace_id"] == p["trace_id"])

near = I.select(traces, by, encode, types.SimpleNamespace(**{**vars(ARGS), "horizon": 3}))[0]
check("first reuse beyond the horizon: no consequential site, no pair", near == [], near)

print("\n4. pair_up: greedy nearest, same digit count, each site once, capped")
mk = lambda kind, tok, v="12": {"kind": kind, "edit_tok": tok, "value": v}
cands = [mk("consequential", 100), mk("consequential", 500), mk("benign", 120), mk("benign", 480),
         mk("benign", 101, "123")]
pr = I.pair_up(cands, max_gap=1000, cap=5)
check("pairs 100-120 and 500-480; the 3-digit benign is never matched to 2-digit",
      [(x["consequential"]["edit_tok"], x["benign"]["edit_tok"]) for x in pr] == [(100, 120), (500, 480)],
      [(x["consequential"]["edit_tok"], x["benign"]["edit_tok"]) for x in pr])
check("max_gap respected", I.pair_up(cands, max_gap=15, cap=5) == [])
check("cap respected", len(I.pair_up(cands, max_gap=1000, cap=1)) == 1)

print("\n5. report: metrics and paired tests on planted scores")


def scored(e, reuse, kl_after_reuse, write_nll, lo=10, hi=60, cos=0.0, seconds=1.0):
    n = hi - lo
    kl = [0.0] * n
    for j in range(e + 1, hi):
        kl[j - lo] = 0.01 + (kl_after_reuse if reuse and j >= reuse[0] else 0.0)
    nll_c = [1.0] * n
    nll_e = [1.0] * n
    nll_c[e - lo], nll_e[e - lo] = 0.1, write_nll
    for j in reuse:
        nll_e[j - lo] = 4.0
    return {"lo": lo, "nll_clean": nll_c, "nll_edit": nll_e, "kl": kl,
            "cos": {"20": [0.0 if j < e else cos for j in range(lo, hi)]}, "edit_tok": e, "reuse": reuse,
            "p_new_clean": [0.01] * len(reuse), "p_new_edit": [0.9] * len(reuse)}


rows = []
for k in range(30):
    rows.append({"pair_id": f"x:{k}", "trace_label": "wrong" if k % 2 else "correct", "seconds": 2.0 + k,
                 "sites": {"consequential": scored(30, [40], 0.5 + 0.01 * k, 9.0, cos=0.3),
                           "benign": scored(20, [], 0.0, 9.0, cos=0.05)},
                 **({"identity": {"kl_max": 0.0, "nll_absmax": 0.0, "cos_max": 0.0}} if k < 3 else {})})
per, s = I.report(rows, pre=64)
c0, b0 = per[0]["consequential"], per[0]["benign"]
check("consequential split at its reuse: no extra KL before, +0.5 from it",
      abs(c0["kl_before_split"] - 0.01) < 1e-9 and abs(c0["kl_from_split"] - 0.51) < 1e-9,
      (c0["kl_before_split"], c0["kl_from_split"]))
check("benign split at its partner's offset (edit + 10)", abs(b0["kl_from_split"] - 0.01) < 1e-9 and
      not math.isnan(b0["kl_before_split"]), (b0["kl_before_split"], b0["kl_from_split"]))
check("write-point: edited digit is the most surprising token, the clean digit is not",
      c0["write_hit"] and c0["write_rank_clean"] > 1, (c0["write_rank_edit"], c0["write_rank_clean"]))
check("reuse: surprisal up 3 nats, edited digit copied", abs(c0["reuse_dnll"] - 3.0) < 1e-9 and c0["reuse_copied"])
t = s["paired_consequential_minus_benign"]
check("kl_from_split: consequential above benign, small p", t["kl_from_split"]["mean_diff"] > 0.4
      and t["kl_from_split"]["p_signflip"] < 0.01 and t["kl_from_split"]["frac_positive"] == 1, t["kl_from_split"])
check("write surprisal matched across arms (diff 0)", abs(t["write_nll_edit"]["mean_diff"]) < 1e-12)
check("bootstrap CI brackets the mean", t["kl_after"]["ci95"][0] <= t["kl_after"]["mean_diff"] <= t["kl_after"]["ci95"][1])
check("identity checks summarised", s["identity_checks"] == 3 and s["identity_max"]["kl_max"] == 0.0)
check("per-column ranges reported, timing too", s["columns"]["consequential.kl_from_split"]["min"] > 0.5
      and s["seconds_per_pair"]["min"] == 2.0, s["seconds_per_pair"])
check("summary is JSON-serialisable", bool(json.dumps(s)))

print("\n6. site_range stays inside the trace and never starts at token 0")
a = types.SimpleNamespace(pre=64, horizon=256)
check("near the start", I.site_range({"edit_tok": 10}, 1000, a) == (1, 267))
check("near the end", I.site_range({"edit_tok": 900}, 1000, a) == (836, 1000))

print(f"\n{len(fails)} failed" if fails else "\nall passed")
sys.exit(1 if fails else 0)
