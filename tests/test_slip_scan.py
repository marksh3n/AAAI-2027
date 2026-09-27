# slip_scan: claim detection, false-positive skips, exact evaluation, load-bearing and
# self-correction flags, and an end-to-end run on a tiny traces file. CPU, no tokenizer.
import json, os, subprocess, sys, tempfile
from fractions import Fraction

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "injection"))
import slip_scan as S  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(extra)) if extra else ''}")
    if not cond:
        fails.append(name)


def statuses(text):
    return [c["status"] for c in S.find_claims(text)]


print("1. slips found, with the right true value")
for text, true, written in [
    ("So 3 + 4 = 8.", 7, 8),
    (r"\(12 \times 11 = 142\)", 132, 142),
    (r"Then \frac{3}{4} + \frac{1}{4} = 2 here", 1, 2),
    ("We get 2^{10} = 1000 ways", 1024, 1000),
    (r"and (3+4) \times 2 = 15.", 14, 15),
    ("Sum: 3 + 4 + 5 = 13.", 12, 13),
    ("we have 3 - 5 = -3 now", -2, -3),
    ("**3 + 4 = 8**", 7, 8),
]:
    c = S.find_claims(text)
    ok = len(c) == 1 and c[0]["status"] == "slip" and c[0]["true"] == true and c[0]["written"] == written
    check(f"slip: {text}", ok, [(x["status"], x.get("true"), x.get("written")) for x in c])

print("\n2. correct claims, rounding and truncation are not slips")
for text in [r"Check: 7 \cdot 8 = 56.", "so 1/3 = 0.333 and", "and 2/3 = 0.67 so", r"\(6 \div 4 = 1.5\)",
             r"then \dfrac{1}{2} \cdot 4 = 2."]:
    check(f"ok: {text}", statuses(text) == ["ok"], statuses(text))

print("\n3. skipped, never counted as slips")
for text in ["then x + 3 + 4 = 7", "roughly 22/7 = 3.15", r"gives 5 \times 3 = 16 cm",
             "so 100 / 7 = 14 remainder 2", "so 100/7 = 14 each", "and x^2 + 3 = 12", "is 2 + 2 = 5?",
             "total 1,000 + 5 = 1006", "if 3 + 4 = 8 then", "gives 3 + 4 = 7x", r"3 \times 4 = 12 \times 1",
             "so -3^2 = 9", "so 10^{100} = 5", "so 5/0 = 1", "about 1/3 = 0.3", r"so 2 + 3 \approx 5.1 = 5",
             "a_{1} + 2 = 9",
             # holes found in review, each seen in traces_B
             r"so 3 \times 4 = 1,200", "and 125*187=23,375 total", r"so 2^{10} = 1{,}024", r"so 10^3 = 1\,000",
             "list:\n- 3 * 4 = 12", "1875 +1465=3340/18=1670/9", "2^3 = 1 (mod 7)",
             "the units digit cycles: 3^1 = 3, 3^2 = 9, 3^3 = 7, 3^4 = 1", "6 + 8 = 2(3 + 4)", "3 \\times 3 = 3²",
             "from 1:45 + 30 = 2:15", "so 2**3 + 1 = 10", r"so \frac{1}{10000000}^{-64} = 1"]:
    st = statuses(text)
    check(f"skip: {text}", all(s.startswith("skip:") for s in st), st)
for text in ["then 2x + 3 = 7", "f(2) + 1 = 9", "3 + 4\n= 8"]:
    check(f"not a claim: {text!r}", S.find_claims(text) == [], statuses(text))

print("\n4. evaluator")
check("exact rationals", S.evaluate(S.normalize(r"\frac{1}{3} \times 3")) == 1)
check("precedence and right-assoc ^", S.evaluate("2+3*2^2^1") == 14)
check("decimals exact", S.evaluate("0.1+0.2") == Fraction(3, 10))
for bad in ["2+", "(2+3", "2+a", "2**3"]:
    try:
        S.evaluate(bad)
        check(f"rejects {bad!r}", False)
    except S.Skip:
        check(f"rejects {bad!r}", True)

print("\n5. load-bearing and self-correction")
t = r"First 3 + 4 = 8. Then 8 \times 2 = 16. Wait, 3 + 4 = 7."
c = [x for x in S.find_claims(t) if x["status"] == "slip"]
check("one slip, the other two claims fine", len(c) == 1 and statuses(t) == ["slip", "ok", "ok"], statuses(t))
a = S.annotate(c[0], t, len(t), "16")
check("written value reused later, not seen before -> load-bearing", a["load_bearing"], a)
check("true value restated in the window -> self-corrected, with a cue", a["self_corrected"] and a["correction_cue"], a)
a2 = S.annotate(c[0], t, c[0]["end"] + 5, "16")
check("window that stops before the restatement -> not self-corrected", not a2["self_corrected"], a2)
t = r"So 12 \times 11 = 142, so the total is 142 + 8 = 150."
c = S.find_claims(t)
a = S.annotate(c[0], t, len(t), "150")
check("uncorrected and load-bearing", a["load_bearing"] and not a["self_corrected"] and not a["written_is_final_answer"], a)
t = "We have 8 apples and 3 + 4 = 8. Then 8 more."
c = S.find_claims(t)
check("value already present before the claim -> not load-bearing", not S.annotate(c[0], t, len(t), None)["load_bearing"])
t = "Thus 2 \\times 7 = 15.\n\\boxed{15}"
c = S.find_claims(t)
check("written value is the final answer", S.annotate(c[0], t, len(t), "15")["written_is_final_answer"])
check("fraction forms searched", {"3/4", "\\frac{3}{4}", "0.75"} <= S.forms(Fraction(3, 4)), S.forms(Fraction(3, 4)))
check("23 not found inside 23,375 or 1{,}023", S.occurrences("got 23,375 and 1{,}023", Fraction(23), 0, 30) == 0)
check("23 found on its own", S.occurrences("so 23, then 23.", Fraction(23), 0, 20) == 2)
t = "So 3 + 4 = 8. Again 3 + 4 = 8."
c = S.find_claims(t)
a = S.annotate(c[0], t, len(t), None, [(c[1]["start"], c[1]["end"])])
check("rewriting the same wrong claim is not reuse", a["written_reused_after"] == 0 and not a["load_bearing"], a)

print("\n6. end to end, --no-tokens")
with tempfile.TemporaryDirectory() as d:
    rows = [
        {"trace_id": "1_0", "problem_id": 1, "level": 5, "trunc": False, "n_gen": 900, "is_correct": False,
         "answer": "150", "text": r"So 12 \times 11 = 142, so the total is 142 + 8 = 150. Again 12 \times 11 = 142."
                                  "</think> \\boxed{150}"},
        {"trace_id": "1_1", "problem_id": 1, "level": 5, "trunc": False, "n_gen": 500, "is_correct": True,
         "answer": "56", "text": r"Here 7 \cdot 8 = 56.</think> \boxed{56}"},
        {"trace_id": "1_2", "problem_id": 1, "level": 5, "trunc": True, "n_gen": 32768, "is_correct": False,
         "answer": None, "text": "3 + 4 = 8 " * 5},
        {"trace_id": "1_3", "problem_id": 1, "level": 5, "trunc": False, "n_gen": 50, "is_correct": False,
         "answer": None, "text": "3 + 4 = 8 and no think end"},
    ]
    p = os.path.join(d, "traces_T.jsonl")
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cmd = [sys.executable, os.path.join(ROOT, "src", "injection", "slip_scan.py"), "--tag", "T", "--traces", p,
           "--out-dir", d, "--no-tokens", "--go", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    check("exits cleanly", r.returncode == 0, r.stderr[-600:])
    if r.returncode == 0:
        s = json.load(open(os.path.join(d, "slip_summary_T.json")))
        w, c = s["by_label"]["wrong"], s["by_label"]["correct"]
        check("truncated and no-</think> traces excluded", s["excluded"] == {"truncated": 1, "no </think>": 1}, s["excluded"])
        check("wrong: 3 claims, 1 slip counted once, 1 repeat", (w["claims"], w.get("slips"), w.get("repeats")) == (3, 1, 1), w)
        check("wrong: that slip is load-bearing and uncorrected", w.get("load_bearing_uncorrected") == 1, w)
        check("correct: 1 claim, no slip", c["claims"] == 1 and c.get("slips", 0) == 0, c)
        out = [json.loads(l) for l in open(os.path.join(d, "slips_T.jsonl"))]
        check("both occurrences written, first flagged", [o["first_occurrence"] for o in out] == [True, False], out)
        check("char offset points at the statement", rows[0]["text"][out[0]["char_offset"]:].startswith("12 \\times 11"))
        check("go/no-go line printed", "GO, pending the precision check" in r.stdout, r.stdout[-400:])

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all passed")
