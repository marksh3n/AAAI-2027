# recovery v2 pure logic: snapping, checkpoints, scoring, resume, commitment, temp MLE.
# heavy imports stubbed; the temperature check runs only when torch is installed.
import json, os, sys, tempfile, types
from collections import Counter

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

print("\n2b. ceiling population, estimates, tiers")


def tr(tid, pid, ok, n_gen, trunc=False):
    return {"trace_id": tid, "problem_id": pid, "is_correct": ok, "n_gen": n_gen, "trunc": trunc}


T = [tr("1_0", 1, True, 5000), tr("1_1", 1, False, 6000), tr("1_2", 1, False, 32768, True),
     tr("2_0", 2, True, 5000), tr("2_1", 2, True, 7000),  # all correct: not mixed
     tr("3_0", 3, True, 5000), tr("3_1", 3, False, 3000)]  # wrong one too short
lens = {t["trace_id"]: t["n_gen"] - 1 for t in T}
pop = bp.ceiling_population(T, lens, 4096)
check("mixed problems among long untruncated traces", list(pop) == [1], list(pop))
check("keeps correct and wrong traces", [t["trace_id"] for t in pop[1]] == ["1_0", "1_1"])
check("t0 estimate: mean length, truncated at the cap", bp.est_cont_tokens(T[:3], 0, 32768)
      == int((5000 + 6000 + 32768) / 3))
check("position estimate: only traces past the cut", bp.est_cont_tokens(T[:3], 5500, 32768)
      == int((500 + 32768 - 5500) / 2))
units = {p: (rate, 4) for p, rate in zip(range(12), [0.1, 0.2, 0.5, 0.5, 0.9, 0.9] * 2)}
order = bp.tier_order(units, 10, seed=0)
check("every problem once", sorted(p for p, _ in order) == list(range(12)))
check("tier 1 is the first problems reaching the target", [t for _, t in order] == [1] * 3 + [2] * 9,
      order)
check("tier 1 spans the pass bins", {bp.pass_bin(units[p][0]) for p, t in order if t == 1}
      == {"low", "mid", "high"})
check("seeded", order == bp.tier_order(units, 10, seed=0) and order != bp.tier_order(units, 10, seed=1))

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


class FakeEngine:
    # DELTA semantics with n>1: every step, each unfinished child of each request emits one
    # token as its own output (tagged with its index); child i finishes after 2 + i tokens,
    # and a request is finished once all its children are. the newest request steps first
    def __init__(self):
        self.live, self.max_seen, self.added, self.aborted = [], 0, [], []
        self.overlap = False  # a new request arrived while an older one still ran

    def add_request(self, rid, prompt, sp):
        if self.live:
            self.overlap = True
        self.live.append([rid, sp, [0] * sp.n])
        self.added.append((rid, sp))
        self.max_seen = max(self.max_seen, self.unfinished())

    def unfinished(self):
        return sum(k < 2 + i for _, _, ks in self.live for i, k in enumerate(ks))

    def abort_request(self, rids):
        self.aborted += rids

    def step(self):
        outs = []
        for item in reversed(list(self.live)):
            rid, sp, ks = item
            for i in range(sp.n):
                if ks[i] >= 2 + i:
                    continue
                ks[i] += 1
                done = ks[i] == 2 + i
                all_done = all(k >= 2 + j for j, k in enumerate(ks))
                text = f"s{sp.seed + i} " if ks[i] == 1 else ("\\boxed{12}" if done else ".")
                outs.append(types.SimpleNamespace(request_id=rid, finished=all_done, outputs=[
                    types.SimpleNamespace(index=i, text=text, token_ids=[1],
                                          finish_reason="stop" if done else None)]))
            if all(k >= 2 + j for j, k in enumerate(ks)):
                self.live.remove(item)
        return outs


class FakeLLM:
    def __init__(self):
        self.llm_engine = FakeEngine()

    def get_tokenizer(self):
        return types.SimpleNamespace(decode=lambda ids: "".join(map(str, ids)))


def prefix(i, problem=None, tier=None, est=10):
    return {"prefix_id": f"trunc/{i}_0/f50", "source": "truncation",
            "problem_id": i if problem is None else problem, "tier": tier,
            "trace_id": f"{i}_0", "checkpoint": "f50", "cut_token_idx": 2, "gold_answer": "12",
            "prompt_token_ids": [9, 9], "prefix_gen_token_ids": [i, i], "max_tokens": 100,
            "est_cont_tokens": est}


params = lambda **kw: types.SimpleNamespace(**kw)
args = types.SimpleNamespace(inflight=6, temp=0.7, top_p=0.95, max_tokens_cap=0, seed=0, round=0,
                             n=2, max_minutes=0, max_num_seqs=4)
prefixes = [prefix(i) for i in range(3)]
with tempfile.TemporaryDirectory() as d:
    out = f"{d}/rollouts.jsonl"
    rr.run(FakeLLM(), params, rr.todo(prefixes, rr.done_rows(out, prefixes), 0, 2, limit=2), args,
           out, "h")
    rows = read_jsonl(out)
    check("one row per prefix: 2 prefixes", len(rows) == 2, len(rows))
    check("each row holds its n samples, in index order",
          all([s["sample_idx"] for s in r["samples"]] == [0, 1] for r in rows))
    check("token count per sample sums its deltas",
          all(s["n_gen"] == 2 + s["sample_idx"] for r in rows for s in r["samples"]))
    check("deltas stitched into each sample's text",
          all(s["text"].startswith(f"s{s['seed']} ") and s["text"].endswith("\\boxed{12}")
              for r in rows for s in r["samples"]), rows[0]["samples"][0]["text"])
    check("scored per sample", all(s["correct"] for r in rows for s in r["samples"]))
    pid0 = prefixes[0]["prefix_id"]
    r0 = next(r for r in rows if r["prefix_id"] == pid0)
    check("sample seeds follow vLLM's seed + index",
          [s["seed"] for s in r0["samples"]] == [prefix_seed(0, pid0, 0) + i for i in range(2)]
          and r0["seed"] == prefix_seed(0, pid0, 0))
    check("rerun with the same limit is a no-op",
          rr.todo(prefixes, rr.done_rows(out, prefixes), 0, 2, limit=2) == [])

    # a stop mid-write leaves a torn last line: repaired, and that prefix alone reruns
    with open(out) as f:
        lines = f.readlines()
    with open(out, "w") as f:
        f.writelines(lines[:-1])
        f.write(lines[-1][:40])
    check("torn line tolerated on read", len(read_jsonl(out)) == 1)
    check("torn line cut before appending", rr.repair_tail(out) == 40 and rr.repair_tail(out) == 0)
    torn_pid = json.loads(lines[-1])["prefix_id"]
    work = rr.todo(prefixes, rr.done_rows(out, prefixes), 0, 2)
    check("resume requests only prefixes without a row",
          sorted(r["prefix_id"] for r, _ in work) == sorted([torn_pid, prefixes[2]["prefix_id"]]),
          [r["prefix_id"] for r, _ in work])
    llm = FakeLLM()
    rr.run(llm, params, work, args, out, "h")
    rows = read_jsonl(out)
    keys = [(r["prefix_id"], r["round"]) for r in rows]
    check("no duplicate (prefix, round)", len(keys) == len(set(keys)) == 3, keys)
    check("third rerun has nothing to do", rr.todo(prefixes, rr.done_rows(out, prefixes), 0, 2) == [])

    llm = FakeLLM()
    with tempfile.TemporaryDirectory() as d2:
        small = types.SimpleNamespace(**{**vars(args), "inflight": 4})
        rr.run(llm, params, rr.todo(prefixes, set(), 0, 2), small, f"{d2}/r.jsonl", "h")
        check("never more than --inflight + n - 1 unfinished samples on the engine",
              llm.llm_engine.max_seen <= 4 + 2 - 1, llm.llm_engine.max_seen)
    llm = FakeLLM()
    with tempfile.TemporaryDirectory() as d2:
        # inflight == n: the next prefix can only start on a slot freed by a finished sample
        tight = types.SimpleNamespace(**{**vars(args), "inflight": 2})
        rr.run(llm, params, rr.todo(prefixes, set(), 0, 2), tight, f"{d2}/r.jsonl", "h")
        check("a finished sample frees its slot before its prefix is done", llm.llm_engine.overlap)
        check("a prefix is admitted whole: at most inflight + n - 1 unfinished",
              llm.llm_engine.max_seen <= 2 + 2 - 1, llm.llm_engine.max_seen)
        check("all prefixes written", len(read_jsonl(f"{d2}/r.jsonl")) == 3)
        check("one request per prefix with n samples",
              [sp.n for _, sp in llm.llm_engine.added] == [2, 2, 2])

    top = types.SimpleNamespace(**{**vars(args), "round": 1})
    work = rr.todo(prefixes, rr.done_rows(out, prefixes), 1, 2, {pid0: 3})
    check("top-up picks only overridden prefixes, with their n", [(r["prefix_id"], n) for r, n in work]
          == [(pid0, 3)])
    rr.run(FakeLLM(), params, work, top, out, "h")
    rows = read_jsonl(out)
    r0 = [r for r in rows if r["prefix_id"] == pid0]
    check("top-up appended as round 1", sorted(r["round"] for r in r0) == [0, 1])
    seeds0 = {s["seed"] for r in r0 if r["round"] == 0 for s in r["samples"]}
    seeds1 = {s["seed"] for r in r0 if r["round"] == 1 for s in r["samples"]}
    check("round 1 draws a different seed stream", not seeds0 & seeds1)

    changed = [dict(prefixes[0], prefix_gen_token_ids=[7, 7])] + prefixes[1:]
    try:
        rr.done_rows(out, changed)
        check("changed prefix tokens refused", False)
    except StaleCache:
        check("changed prefix tokens refused", True)

print("\n5b. problem units, budget and time stops")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class SlowEngine(FakeEngine):
    # every step costs the clock `dt` seconds
    def __init__(self, clock, dt):
        super().__init__()
        self.clock, self.dt = clock, dt

    def step(self):
        self.clock.t += self.dt
        return super().step()


def slow_llm(clock, dt):
    llm = FakeLLM()
    llm.llm_engine = SlowEngine(clock, dt)
    return llm


# 4 problems x 3 prefixes; tier 1 = problems 0-1
units = [prefix(10 * p + j, problem=p, tier=1 if p < 2 else 2) for p in range(4) for j in range(3)]
check("units group consecutive (tier, problem)", [len(u) for u in rr.units_of([(r, 2) for r in units])]
      == [3, 3, 3, 3])
with tempfile.TemporaryDirectory() as d:
    clock = Clock()
    # $36/h = 1 cent a second; the startup allowance alone is 240 s = $2.40
    bargs = types.SimpleNamespace(**{**vars(args), "inflight": 2, "rate_usd_hr": 36.0,
                                     "budget_usd": 5.0, "ledger_key": "k1"})
    budget = rr.Budget(bargs, f"{d}/r.jsonl.ledger.json", clock=clock)
    stop, _ = rr.run(slow_llm(clock, 20.0), params, rr.todo(units, set(), 0, 2), bargs,
                     f"{d}/r.jsonl", "h", budget, clock=clock)
    rows = read_jsonl(f"{d}/r.jsonl")
    per_problem = Counter(r["problem_id"] for r in rows)
    check("budget stop is reported", stop is not None and stop.startswith("budget"), stop)
    check("each problem complete or not started", all(v == 3 for v in per_problem.values()),
          dict(per_problem))
    check("stops before the budget, with problems left", 0 < len(per_problem) < 4, dict(per_problem))
    check("spend stays under the budget", budget.spent() < bargs.budget_usd, budget.spent())
    check("started in priority order", sorted(per_problem) == list(range(len(per_problem))))
    budget.save()
    again = rr.Budget(bargs, f"{d}/r.jsonl.ledger.json", clock=clock)
    check("a retry with the same ledger key inherits the spend",
          abs(again.prior - budget.spent()) < 1e-9, (again.prior, budget.spent()))
    other = rr.Budget(types.SimpleNamespace(**{**vars(bargs), "ledger_key": "k2"}),
                      f"{d}/r.jsonl.ledger.json", clock=clock)
    check("a new launch starts its own spend", other.prior == 0)

    clock = Clock()
    big = [dict(r, est_cont_tokens=10**6) for r in units]
    bargs2 = types.SimpleNamespace(**{**vars(bargs), "budget_usd": 10.0, "ledger_key": None})
    stop, _ = rr.run(slow_llm(clock, 20.0), params, rr.todo(big, set(), 0, 2), bargs2,
                     f"{d}/big.jsonl", "h", rr.Budget(bargs2, None, clock=clock), clock=clock)
    check("a problem whose planned cost would cross the budget is never started",
          stop is not None and read_jsonl(f"{d}/big.jsonl") == [], stop)

    clock = Clock()
    targs = types.SimpleNamespace(**{**vars(args), "inflight": 2, "max_minutes": 1.0})
    stop, _ = rr.run(slow_llm(clock, 25.0), params, rr.todo(units, set(), 0, 2), targs,
                     f"{d}/t.jsonl", "h", None, clock=clock)
    rows = read_jsonl(f"{d}/t.jsonl")
    check("--max-minutes stops the run", stop is not None and "max-minutes" in stop, stop)
    check("and leaves work for a resume", 0 < len(rows) < len(units), len(rows))

print("\n6. temperature MLE")
if not HAVE_TORCH:
    print("  skip  torch not installed")
else:
    import numpy as np
    import check_temp as et
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
