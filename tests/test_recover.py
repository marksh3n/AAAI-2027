# recover.py pure logic. torch, vLLM, datasets, math_verify stubbed.
import json, os, subprocess, sys, tempfile, types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# scipy probes torch.Tensor at import
stub("torch", cuda=types.SimpleNamespace(is_available=lambda: False),
     float16=None, float32=None, Tensor=type("Tensor", (), {}), __version__="0.0.0",
     no_grad=lambda *a, **k: None, inference_mode=lambda *a, **k: None)
stub("datasets", load_dataset=lambda *a, **k: {"test": []})
# normalised string equality is enough here
stub("math_verify", parse=lambda s: s.strip("$ ").strip(), verify=lambda a, b: a == b)
stub("transformers", AutoTokenizer=object, AutoModelForCausalLM=object)
stub("vllm", LLM=object, SamplingParams=object)

import numpy as np
import pandas as pd

import recover
from cache import StaleCache

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  pass  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


TRACE = {"trace_id": "7_2", "problem_id": 7, "level": 4, "gold": "42", "is_correct": False}

print("\n1. correctness judged on prefix + continuation")
r = recover.record(TRACE, 512, 0, prefix="...so x = 6.\n</think>\nThe answer is \\boxed{42}",
                   text=" and that is final.", n_gen=5, budget=8192)
check("finds an answer that lives in the prefix", r["answer"] == "42", r["answer"])
check("scores it correct", r["is_correct"] is True, r["is_correct"])
check("does not call it truncated", r["trunc"] is False, r["trunc"])

r2 = recover.record(TRACE, 128, 0, prefix="let me think.",
                    text=" </think> so \\boxed{41}", n_gen=9, budget=8192)
check("finds an answer in the continuation", r2["answer"] == "41", r2["answer"])
check("scores a wrong answer wrong", r2["is_correct"] is False)

print("\n2. capped and wrong recorded separately")
r3 = recover.record(TRACE, 128, 0, prefix="a", text="b", n_gen=8192, budget=8192)
check("capped flagged", r3["capped"] is True)
check("capped has no answer", r3["answer"] is None and r3["is_correct"] is False)
check("uncapped not flagged", recover.record(TRACE, 128, 0, "a", "b", 10, 8192)["capped"] is False)
check("budget recorded", r3["budget"] == 8192)
check("gold recorded", r3["gold"] == "42")

print("\n3. resume seed: fresh per top-up, stable across processes")
a = recover.row_seed(0, "7_2", 512, 0)
check("deterministic", a == recover.row_seed(0, "7_2", 512, 0))
check("differs by have", a != recover.row_seed(0, "7_2", 512, 6))
check("differs by pos", a != recover.row_seed(0, "7_2", 128, 0))
check("differs by trace", a != recover.row_seed(0, "7_3", 512, 0))
other = subprocess.run(
    [sys.executable, "-c",
     "import hashlib;print(int(hashlib.sha256(b'0:7_2:512:0').hexdigest()[:8],16))"],
    capture_output=True, text=True).stdout.strip()
check("not salted per process", str(a) == other, f"{a} vs {other}")

print("\n4. rows survive json.dumps")
rn = recover.record(TRACE, np.int64(512), np.int64(3), "a", "b", np.int64(10), np.int64(8192))
try:
    json.dumps(rn)
    check("numpy scalars refused by json", False, "expected TypeError")
except TypeError:
    check("numpy scalars refused by json, so main() casts", True)
check("plain ints serialise", json.loads(json.dumps(
    recover.record(TRACE, 512, 3, "a", "b", 10, 8192)))["sample_idx"] == 3)


print("\n5. select() tops up by shortfall")
df = pd.DataFrame([
    {"trace_id": "1_0", "pos": 128, "prefix": "p", "is_correct": False},
    {"trace_id": "1_0", "pos": 512, "prefix": "p", "is_correct": False},
    {"trace_id": "2_0", "pos": 128, "prefix": "p", "is_correct": True},
])
counts = {("1_0", 128): 12, ("1_0", 512): 5}
args = types.SimpleNamespace(samples=12, n=0, seed=0)
sel = recover.select(df, args, counts)
check("drops a prefix already at budget", ("1_0", 128) not in set(zip(sel.trace_id, sel.pos)))
check("keeps a partial prefix", ("1_0", 512) in set(zip(sel.trace_id, sel.pos)))
check("records what it already has", int(sel[sel.pos == 512].have.iloc[0]) == 5)
check("keeps an untouched prefix", int(sel[sel.trace_id == "2_0"].have.iloc[0]) == 0)
check("empty frame does not crash", len(recover.select(df.iloc[:0], args, counts)) == 0)

print("\n6. within-problem AUROC of r")
g = pd.DataFrame([
    {"problem_id": 1, "recovery_rate": 0.9, "y": 1}, {"problem_id": 1, "recovery_rate": 0.1, "y": 0},
    {"problem_id": 2, "recovery_rate": 0.8, "y": 1}, {"problem_id": 2, "recovery_rate": 0.2, "y": 0},
])
auroc, pairs, probs = recover.within_problem_auroc(g, "recovery_rate", "y")
check("perfect separation is 1.0", auroc == 1.0 and pairs == 2 and probs == 2, (auroc, pairs, probs))
g.loc[g.y == 1, "recovery_rate"] = 0.05
check("inverted is 0.0", recover.within_problem_auroc(g, "recovery_rate", "y")[0] == 0.0)
g["recovery_rate"] = 0.5
check("ties are 0.5", recover.within_problem_auroc(g, "recovery_rate", "y")[0] == 0.5)
solo = pd.DataFrame([{"problem_id": 3, "recovery_rate": 0.4, "y": 1}])
check("single-outcome problems ignored",
      recover.within_problem_auroc(solo, "recovery_rate", "y")[1] == 0)

d = tempfile.mkdtemp()
os.makedirs(f"{d}/data")
os.chdir(d)

print("\n7. the config gate")
args = types.SimpleNamespace(temp=0.7, top_p=0.95, max_continuation=8192)
cfg, h = recover.run_config(args)
recover.check_config("T", cfg, h)
check("writes a config sidecar", os.path.exists("data/recovery_T.config.json"))
open("data/recovery_T.jsonl", "w").write("{}\n")
recover.check_config("T", cfg, h)
check("same config passes", True)
cfg2, h2 = recover.run_config(types.SimpleNamespace(temp=0.6, top_p=0.95, max_continuation=8192))
try:
    recover.check_config("T", cfg2, h2)
    check("changed temperature refused", False)
except StaleCache as e:
    check("changed temperature refused", "temp" in str(e), str(e))
recover.check_config("T", cfg2, h2, rebuild=True)
check("--rebuild sets old rows aside", not os.path.exists("data/recovery_T.jsonl"))
open("data/recovery_U.jsonl", "w").write("{}\n")
try:
    recover.check_config("U", cfg, h)
    check("unrecorded jsonl refused", False)
except StaleCache as e:
    check("unrecorded jsonl refused", "--rebuild" in str(e))

print("\n8. summarise dedupes, reports both estimators")
rows = []
for k in range(4):  # first run, sample_idx 0..3
    rows.append({"trace_id": "1_0", "problem_id": 1, "level": 4, "pos": 128, "sample_idx": k,
                 "is_correct": k < 2, "capped": False, "trunc": False})
for k in range(2, 6):  # overlapping top-up rewrites 2..3 and adds 4..5
    rows.append({"trace_id": "1_0", "problem_id": 1, "level": 4, "pos": 128, "sample_idx": k,
                 "is_correct": False, "capped": k >= 4, "trunc": k >= 4})
rows.append({"trace_id": "1_1", "problem_id": 1, "level": 4, "pos": 128, "sample_idx": 0,
             "is_correct": True, "capped": False, "trunc": False})
with open("data/recovery_S.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
traces = {"1_0": {"is_correct": False}, "1_1": {"is_correct": True}}
recover.summarise("S", traces)
agg = pd.read_csv("data/recovery_summary_S.csv")
one = agg[agg.trace_id == "1_0"].iloc[0]
check("kept 6 distinct samples", one.n_samples == 6, one.n_samples)
check("r counts capped as failure", abs(one.recovery_rate - 2 / 6) < 1e-9, one.recovery_rate)
check("r_terminated excludes capped", abs(one.recovery_rate_terminated - 2 / 4) < 1e-9,
      one.recovery_rate_terminated)
check("cap rate reported", abs(one.cap_rate - 2 / 6) < 1e-9, one.cap_rate)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
