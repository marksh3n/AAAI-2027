# recovery v2 pure logic: snapping, checkpoints, scoring, resume, commitment, temp MLE.
# heavy imports stubbed; the temperature check runs only when torch is installed.
import os, sys, tempfile, types

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, os.path.join(ROOT, "recovery"))
sys.path.insert(0, ROOT)


def stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    stub("torch", cuda=types.SimpleNamespace(is_available=lambda: False),
         float16=None, float32=None, Tensor=type("Tensor", (), {}), __version__="0.0.0")
stub("datasets", load_dataset=lambda *a, **k: {"test": []})
# normalised string equality is enough here
stub("math_verify", parse=lambda s: s.strip("$ ").strip(), verify=lambda a, b: a == b)
stub("transformers", AutoTokenizer=object, AutoModelForCausalLM=object)

import pandas as pd

import analyze
import build_prefixes as bp
import run_rollouts as rr
from cache import StaleCache
from common import prefix_seed, read_jsonl, wilson

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  pass  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


def offsets_of(toks):
    out, pos = [], 0
    for t in toks:
        out.append((pos, pos + len(t)))
        pos += len(t)
    return "".join(toks), out


TOKS = ["Step", " one", ".\n\n", "x", " = 1", "2", ".\n\n", "\n", "</think>", "\n\n", "So",
        " \\boxed{12}"]
TEXT, OFFS = offsets_of(TOKS)
POINTS = bp.cut_points(TEXT, OFFS)

print("\n1. step-boundary snapping")
check("boundaries after '.\\n\\n' tokens and at </think>", POINTS == [3, 7, 8, 10], POINTS)
check("mid-number request snaps back to the previous step", bp.snap(POINTS, 5) == 3)
check("request before any boundary snaps to 0", bp.snap(POINTS, 2) == 0)
check("request exactly on a boundary stays", bp.snap(POINTS, 7) == 7)
check("request past </think> snaps to the </think> cut", bp.snap(POINTS, 9) == 8)
t2, o2 = offsets_of(["a", "\n\nb", "c"])
check("'\\n\\n' merged into the next word is not a boundary", bp.cut_points(t2, o2) == [])

print("\n2. first mention and pre_final")
fm = bp.first_mention(TEXT, "12", lambda ans, para: ans in para)
check("first paragraph mentioning the answer", fm == TEXT.index("x = 12"), fm)
check("mapped to the cut right before it", bp.token_at(OFFS, fm) == 3, bp.token_at(OFFS, fm))
check("absent answer gives None", bp.first_mention(TEXT, "99", lambda a, p: a in p) is None)
pf = bp.pre_final(TEXT)
check("</think> before the boxed paragraph wins", pf == TEXT.index("</think>"), pf)
check("pre_final token is the </think> cut", bp.token_at(OFFS, pf) == 8)
inner = "a\n\nwe get \\boxed{3}\n\nmore\n</think>\n\nso 3"
check("boxed paragraph inside thinking", bp.pre_final(inner) == 3, bp.pre_final(inner))
check("no boxed gives None", bp.pre_final("no answer here") is None)

print("\n3. Wilson interval")
lo, hi = wilson(5, 10)
check("k=5 n=10", abs(lo - 0.2366) < 1e-4 and abs(hi - 0.7634) < 1e-4, (lo, hi))
lo, hi = wilson(0, 8)
check("k=0 n=8", lo == 0 and abs(hi - 0.3244) < 1e-4, (lo, hi))
lo, hi = wilson(8, 8)
check("k=n=8", abs(lo - 0.6756) < 1e-4 and hi == 1, (lo, hi))

print("\n4. continuation scoring")
s = rr.score("", "12", "so \\boxed{12} and on", 256, "length")
check("length-capped: no answer", s["answer"] is None and s["correct"] is False and s["truncated"])
s = rr.score("x \\boxed{12}\n\n", "12", " done.", 3, "stop")
check("answer found in the prefix", s["answer"] == "12" and s["answer_from_prefix"], s)
s = rr.score("x \\boxed{11}\n\n", "12", " no wait \\boxed{12}", 9, "stop")
check("continuation's later boxed wins", s["answer"] == "12" and not s["answer_from_prefix"], s)
check("and is correct", s["correct"] is True)
s = rr.score("", "12", "\\boxed{13}", 4, "stop")
check("wrong answer scored wrong", s["correct"] is False and not s["truncated"])

print("\n5. resumability")


class FakeLLM:
    def get_tokenizer(self):
        return types.SimpleNamespace(decode=lambda ids: "".join(map(str, ids)))

    def generate(self, prompts, params):
        out = []
        for p, sp in zip(prompts, params):
            outs = [types.SimpleNamespace(text=f"s{sp.seed}-{j} \\boxed{{12}}", token_ids=[1, 2],
                                          finish_reason="stop") for j in range(sp.n)]
            out.append(types.SimpleNamespace(outputs=outs, prompt_token_ids=p["prompt_token_ids"]))
        return out


def prefix(i):
    return {"prefix_id": f"trunc/{i}_0/f50", "source": "truncation", "problem_id": i,
            "trace_id": f"{i}_0", "checkpoint": "f50", "cut_token_idx": 2, "gold_answer": "12",
            "prompt_token_ids": [9, 9], "prefix_gen_token_ids": [i, i], "max_tokens": 100}


params = lambda **kw: types.SimpleNamespace(**kw)
args = types.SimpleNamespace(chunk=2, temp=0.7, top_p=0.95, max_tokens_cap=0, seed=0, round=0)
prefixes = [prefix(i) for i in range(3)]
with tempfile.TemporaryDirectory() as d:
    out = f"{d}/rollouts.jsonl"
    rr.run(FakeLLM(), params, rr.todo(prefixes, rr.done_rows(out, prefixes), 0, 2, limit=2), args,
           out, "h")
    check("partial run writes 2 records", len(read_jsonl(out)) == 2)
    work = rr.todo(prefixes, rr.done_rows(out, prefixes), 0, 2)
    check("rerun only has the missing prefix", [r["prefix_id"] for r, _ in work] == [prefixes[2]["prefix_id"]])
    rr.run(FakeLLM(), params, work, args, out, "h")
    rows = read_jsonl(out)
    keys = [(r["prefix_id"], r["round"]) for r in rows]
    check("no duplicate (prefix, round)", len(keys) == len(set(keys)) == 3, keys)
    check("third rerun has nothing to do", rr.todo(prefixes, rr.done_rows(out, prefixes), 0, 2) == [])

    top = types.SimpleNamespace(**{**vars(args), "round": 1})
    work = rr.todo(prefixes, rr.done_rows(out, prefixes), 1, 2, {prefixes[0]["prefix_id"]: 3})
    check("top-up picks only overridden prefixes, with their n",
          [(r["prefix_id"], n) for r, n in work] == [(prefixes[0]["prefix_id"], 3)])
    rr.run(FakeLLM(), params, work, top, out, "h")
    rows = read_jsonl(out)
    r0 = [r for r in rows if r["prefix_id"] == prefixes[0]["prefix_id"]]
    check("top-up appended as round 1", sorted(r["round"] for r in r0) == [0, 1])
    check("round 1 draws a different seed", r0[0]["seed"] != r0[1]["seed"])
    check("seed is deterministic", r0[1]["seed"] == prefix_seed(0, prefixes[0]["prefix_id"], 1))
    pooled = analyze.pool_rounds(rows)
    check("analysis pools rounds: 2 + 3 continuations", len(pooled[prefixes[0]["prefix_id"]]) == 5)

    changed = [dict(prefixes[0], prefix_gen_token_ids=[7, 7])] + prefixes[1:]
    try:
        rr.done_rows(out, changed)
        check("changed prefix tokens refused", False)
    except StaleCache:
        check("changed prefix tokens refused", True)

print("\n6. commitment, ordered by cut")
C = analyze.commitment
check("monotone", C([(0, 0, "t0", .2), (100, 100, "f25", .5), (200, 200, "f50", .8),
                     (300, 300, "f75", .9)], .75)[2] == "f50")
check("dip then recover", C([(0, 0, "t0", .8), (100, 100, "f25", .9), (200, 200, "f50", .5),
                             (300, 300, "f75", .8), (400, 400, "pre_final", .9)], .75)[2] == "f75")
check("never", C([(0, 0, "t0", .9), (100, 100, "f25", .9), (200, 200, "f50", .5)], .75) is None)
check("committed from t0", C([(0, 0, "t0", .8), (100, 100, "f25", .9)], .75)[2] == "t0")
# first_mention falls before f25 on this trace; order must follow the cut, not the name
pts = [(200, 210, "f50", .9), (0, 0, "t0", .1), (100, 120, "f25", .6), (50, 60, "first_mention", .9)]
check("uses cut order, not name order", C(pts, .75)[2] == "f50", C(pts, .75))
check("NaN stability breaks the run", C([(0, 0, "t0", .9), (1, 1, "f25", float("nan"))], .75) is None)

print("\n7. adjacent-drop top-up follows cut order")
df = pd.DataFrame({"trace_id": ["a"] * 3, "prefix_id": ["t0", "fm", "f25"], "r": [1.0, 0.5, 1.0],
                   "cut_token_idx": [0, 50, 100]})
up = analyze.topup(df, {"t0": 1.0, "fm": 0.9, "f25": 1.0}, 0.2, 0.8, 0.25, 24)
check("drop flags both sides", {u["prefix_id"] for u in up} == {"t0", "fm"}, up)
check("top-up n", all(u["n"] == 24 for u in up))

print("\n8. temperature MLE")
if not HAVE_TORCH:
    print("  skip  torch not installed")
else:
    import numpy as np
    import estimate_temp as et
    g = torch.Generator().manual_seed(0)
    P, V, T0 = 6000, 64, 0.7
    logits = torch.randn(P, V, generator=g) * 3
    z, order = (logits / T0).sort(-1, descending=True)
    p = z.softmax(-1)
    p = p * ((p.cumsum(-1) - p) < 0.95)
    pick = torch.multinomial(p / p.sum(-1, keepdim=True), 1, generator=g).squeeze(1)
    targets = order.gather(1, pick[:, None]).squeeze(1)
    temps = np.round(np.arange(0.4, 1.01, 0.01), 4)
    ll, inside = et.nucleus_loglik(logits, targets, temps, 0.95)
    t_hat = temps[ll.sum(1).argmax()]
    check("recovers T=0.7 from top-p samples", abs(t_hat - 0.7) <= 0.03, t_hat)
    i7 = int(np.abs(temps - 0.7).argmin())
    check("every sampled token inside the nucleus at the true T", bool(inside[i7].all()))

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
