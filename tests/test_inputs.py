# analysis/inputs.py on synthetic files: limit-hit handling, the >= 2 cuts rule and the cohort rule.
# plain script, CPU only, exits non-zero on failure.
import json, os, sys, tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
tmp = tempfile.mkdtemp()
os.environ["E0_DATA"] = tmp
sys.path.insert(0, os.path.join(ROOT, "src", "analysis"))
import inputs as I  # noqa: E402

fails = []


def check(name, cond, extra=None):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(extra)) if extra is not None else ''}")
    if not cond:
        fails.append(name)


# problem 1: sources 0, 1 finished (correct, wrong), 2 hit the limit
# problem 2: source 0 finished correct, 1 hit the limit (mixed only when limit hits count as wrong)
# problem 3: sources 0, 1 finished, both correct
SRC = {1: [(True, False), (False, False), (False, True)], 2: [(True, False), (False, True)],
       3: [(True, False), (True, False)]}
os.makedirs(f"{tmp}/ministral")
with open(f"{tmp}/ministral/sources.jsonl", "w") as f:
    for p, ss in SRC.items():
        f.write(json.dumps({"problem_id": p, "round": 0, "samples": [
            {"sample_idx": i, "correct": c, "n_gen": 32768 if t else 9000, "truncated": t}
            for i, (c, t) in enumerate(ss)]}) + "\n")
with open(f"{tmp}/ministral/bank.jsonl", "w") as f:
    for p, ss in SRC.items():
        for i in range(len(ss)):
            f.write(json.dumps({"prefix_id": f"cut/{p}_{i}/4096", "trace_id": f"{p}_{i}", "problem_id": p,
                                "checkpoint": "p4096", "round": 0, "samples": [
                                    {"sample_idx": k, "correct": k % 2 == 0, "n_gen": 100, "answer": "1",
                                     "finish_reason": "stop", "truncated": False} for k in range(8)]}) + "\n")

tr, t = I.bank("ministral", "drop")
check("drop: limit-hit cuts left out, a problem left with one cut dropped",
      sorted(t.trace_id) == ["1_0", "1_1", "3_0", "3_1"], sorted(t.trace_id))
tr, w = I.bank("ministral", "wrong")
check("wrong: every cut kept", len(w) == 7, len(w))
check("wrong: limit hits count as wrong", not tr.loc["2_1", "is_correct"])
check("cohort rule keeps problems with both outcomes",
      sorted(I.both_outcomes(w, tr).problem_id.unique()) == ["1", "2"])
check("cohort rule under drop", sorted(I.both_outcomes(t, tr).problem_id.unique()) == ["1"])
d = I.d_ci(t)
check("d_ci counts problems and cuts", (d["problems"], d["cuts"]) == (2, 4), d)

print(f"\n{'all passed' if not fails else f'{len(fails)} FAILED: {fails}'}")
sys.exit(1 if fails else 0)
