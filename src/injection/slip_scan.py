# injection pilot, gate 1 (CPU only): natural arithmetic slips inside <think>. a slip is an
# explicit claim "a op b [op c ...] = d" whose written d is not the value of the left side.
# precision over recall: anything that might be a variable, an approximation, a unit, a
# remainder or a set is skipped, never counted. the pilot is worth building only if wrong
# traces hold enough load-bearing, uncorrected slips.
#   python src/injection/slip_scan.py                 # token offsets need transformers (Colab)
#   python src/injection/slip_scan.py --no-tokens     # char offsets only, window ~4 chars/token
import argparse, bisect, json, math, os, random, re
from collections import Counter
from fractions import Fraction

# duplicated, not imported: generate.py pulls in torch
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_NEW = 32768
THINK_END = "</think>"

NUM = r"\d+(?:\.\d+)?"
FRAC = rf"\\[dt]?frac\s*\{{\s*{NUM}\s*\}}\s*\{{\s*{NUM}\s*\}}"
BRACED = rf"\{{\s*-?\s*{NUM}\s*\}}"
ATOM = rf"(?:{FRAC}|{BRACED}|{NUM})"
OP = r"(?:\\times|\\cdot|\\div|[-+*/^×·÷−])"
SP = r"[ \t]*"  # never across a line break
PAREN = rf"(?:\\left{SP})?\({SP}{ATOM}(?:{SP}{OP}{SP}{ATOM})+{SP}(?:\\right{SP})?\)"
TERM = rf"(?:{PAREN}|{ATOM})"
# a sign glued to its number only: "- 343*1" is a markdown bullet, not a minus
LHS = rf"(?:[-−])?{TERM}(?:{SP}{OP}{SP}{TERM}){{1,3}}"
RHS = rf"[-−]?{SP}(?:{FRAC}|{NUM}(?:{SP}/{SP}{NUM})?)"
CLAIM = re.compile(rf"(?P<lhs>{LHS}){SP}=(?![=>]){SP}(?P<rhs>{RHS})")

# a word or command right after the written value that makes it something other than a plain number
UNITS = {"cm", "mm", "m", "km", "kg", "g", "mg", "lb", "lbs", "oz", "s", "sec", "secs", "second",
         "seconds", "min", "mins", "minute", "minutes", "h", "hr", "hrs", "hour", "hours", "day", "days",
         "week", "weeks", "year", "years", "ft", "feet", "foot", "inch", "inches", "yd", "yard", "yards",
         "mile", "miles", "mph", "meter", "meters", "metre", "metres", "liter", "liters", "litre", "ml",
         "l", "degree", "degrees", "dollar", "dollars", "cent", "cents", "percent", "unit", "units", "sq",
         "square", "cubic", "remainder", "r", "rem", "mod", "modulo", "times"}
RIGHT_CMDS = {")", "]", "quad", "qquad", "implies", "Rightarrow", "Longrightarrow", "rightarrow",
              "to", "therefore", "\\", ",", ";"}
LEFT_CMDS = {"\\quad", "\\qquad", "\\implies", "\\Rightarrow", "\\Longrightarrow", "\\rightarrow",
             "\\therefore", "\\to"}
HYPOTHETICAL = {"if", "suppose", "assume", "assuming", "whether", "does"}
APPROX = ("≈", "\\approx", "\\sim", "~", "≃", "\\simeq", "approx", "roughly", "around", "nearly",
          "almost", "estimat", "round", "floor", "ceil", "\\lfloor", "\\lceil")
# modular, base-n and last-digit work: "3^4 = 1" is right mod 10. its header sits further back
MODULAR = re.compile(r"\bmod\b|modulo|\\equiv|≡|\\pmod|\\bmod|remainder|\bdigits?\b|\bcycle|\bbase\b", re.I)
SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
CUE = re.compile(r"\b(wait|no,|mistake|actually|miscalculat\w*|recalculat\w*|recompute|correction|"
                 r"oops|incorrect|that's wrong|not right)", re.I)
MAX_EXP, MAX_DIGITS = 64, 60


class Skip(Exception):
    pass


def normalize(s):
    s = s.replace("−", "-").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\[dt]?frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", r"((\1)/(\2))", s)
    for a, b in (("\\times", "*"), ("\\cdot", "*"), ("×", "*"), ("·", "*"), ("\\div", "/"),
                 ("÷", "/"), ("{", "("), ("}", ")")):
        s = s.replace(a, b)
    return re.sub(r"\s+", "", s)


def evaluate(expr):
    # exact rational arithmetic over + - * / ^ and parentheses; never eval()
    toks = re.findall(r"\d+(?:\.\d+)?|[-+*/^()]", expr)
    if "".join(toks) != expr:
        raise Skip("unparsed")
    if re.search(r"(?:^|[-+*/^(])-[\d.]+\^", expr):
        raise Skip("unary minus before ^")  # -3^2: writers disagree on the convention
    i = 0

    def peek():
        return toks[i] if i < len(toks) else None

    def take():
        nonlocal i
        i += 1
        return toks[i - 1]

    def expr_():
        v = term()
        while peek() in ("+", "-"):
            v = v + term() if take() == "+" else v - term()
        return v

    def term():
        v = unary()
        while peek() in ("*", "/"):
            if take() == "*":
                v = v * unary()
            else:
                w = unary()
                if w == 0:
                    raise Skip("division by zero")
                v = v / w
        return v

    def unary():
        if peek() == "-":
            take()
            return -unary()
        if peek() == "+":
            take()
            return unary()
        return power()

    def power():
        base = atom()
        if peek() != "^":
            return base
        take()
        exp = unary()
        if exp.denominator != 1 or abs(exp) > MAX_EXP or (base == 0 and exp < 0):
            raise Skip("exponent out of range")
        if base != 0 and abs(float(exp)) * abs(math.log10(abs(float(base)))) > MAX_DIGITS:
            raise Skip("exponent out of range")  # huge or tiny either way
        return base ** int(exp)

    def atom():
        t = take() if peek() is not None else None
        if t == "(":
            v = expr_()
            if take() != ")":
                raise Skip("unbalanced")
            return v
        if t is None or not t[0].isdigit():
            raise Skip("unparsed")
        return Fraction(t)

    try:
        v = expr_()
    except IndexError:
        raise Skip("unbalanced")
    if i != len(toks):
        raise Skip("unparsed")
    return v


def decimals(literal):
    m = re.search(r"\.(\d+)", literal)
    return len(m.group(1)) if m else 0


def left_context(text, start):
    j = start
    while j > 0 and text[j - 1] in " \t":
        j -= 1
    if j == 0:
        return None
    before = text[:j]
    c = before[-1]
    if before.endswith("**") and not before[:-2][-1:].isdigit():
        return None  # markdown bold, not 2**3
    if c == ":" and before[:-1][-1:].isdigit():
        return "clock time or ratio"
    if c == "[" and not before.endswith("\\["):
        return "interval"
    if c == "{" and before[:-1].rstrip().endswith(("^", "_")):
        return "exponent or subscript"
    if c.isdigit() or c in "_^]}|+-−*/×·÷":
        return "expression continues left"
    if c in ".,)!'" and j == start:
        return "expression continues left"  # 1,000 or 2.5 or f(2)+1; "Wait, 3+4" is fine
    if c.isalpha():
        word = re.search(r"(\\?[^\W\d_]+)$", before).group(1)
        if j == start:
            return "variable"
        if word.startswith("\\"):
            return None if word in LEFT_CMDS else "latex command"
        if len(word) == 1 and word not in "aAI":
            return "variable"
        if word.lower() in HYPOTHETICAL:
            return "hypothetical"
    return None


def right_context(text, end):
    if re.match(r"(?:,|\{,\}|\\,|\\ | )\d", text[end:end + 5]):
        return "digit group"  # 23,375 or 1{,}024 or 1\,000 read as 23 / 1
    rest = text[end:end + 40]
    r = rest.lstrip(" \t")
    spaced = len(r) < len(rest)
    if not r:
        return None
    c = r[0]
    if r.startswith("**") and not r[2:3].isdigit():
        return None  # markdown bold closing
    if c in "+-−*/×·÷^_!?%°~≈" or c in SUPERSCRIPTS:
        return "expression continues right"
    if c in "([" and not spaced:
        return "expression continues right"  # = 2(3 + 4)
    if c == ":" and r[1:2].isdigit():
        return "clock time or ratio"
    if c == "." and len(r) > 1 and (r[1].isdigit() or r[1] == "."):
        return "decimal continues"
    if c == "\\":
        cmd = re.match(r"\\([A-Za-z]+|.)", r)
        return None if cmd and cmd.group(1) in RIGHT_CMDS else "latex command after value"
    if c.isalpha():
        if not spaced:
            return "unit or variable"
        if re.match(r"[^\W\d_]+", r).group(0).lower() in UNITS:
            return "unit"
    return None


def approx_context(text, start, end, written_literal):
    w = text[max(0, start - 60):end + 40].lower()
    if any(k in w for k in APPROX):
        return "approximation"
    if decimals(written_literal) and "about" in w:
        return "approximation"
    if MODULAR.search(text[max(0, start - 300):end + 40]):
        return "modular, base or digit context"
    return None


def find_claims(text):
    # every explicit claim in text, with a status: ok, slip, or skip:<reason>
    out = []
    for m in CLAIM.finditer(text):
        start, end = m.start(), m.end()
        lhs, rhs = m.group("lhs"), m.group("rhs")
        rec = {"start": start, "end": end, "statement": m.group(0), "lhs": normalize(lhs),
               "written_literal": rhs.strip()}
        reason = (left_context(text, start) or right_context(text, end)
                  or approx_context(text, start, end, rhs))
        div = re.fullmatch(rf"[-−]?{SP}(\d+){SP}/{SP}(\d+)", rhs.strip())
        if reason is None and div and (re.match(rf"{SP}=", text[end:]) or
                                       math.gcd(int(div.group(1)), int(div.group(2))) > 1):
            reason = "written value is a division"  # 3340/18 = 1670/9: a step, not a value
        if reason is None:
            try:
                true, written = evaluate(normalize(lhs)), evaluate(normalize(rhs))
                d = decimals(rhs)
                if true == written or (d and abs(true - written) <= Fraction(1, 10 ** d)):
                    rec["status"] = "ok"  # exact, or rounded / truncated to the written decimals
                elif (not d and true.denominator != 1 and written.denominator == 1
                      and written in (math.floor(true), math.ceil(true))):
                    raise Skip("integer part of a non-integer")
                else:
                    rec["status"] = "slip"
                rec["true"], rec["written"] = true, written
            except Skip as e:
                reason = str(e)
        if reason is not None:
            rec["status"] = f"skip:{reason}"
        out.append(rec)
    return out


def forms(v):
    # strings a value is plausibly written as
    if v.denominator == 1:
        return {str(v.numerator)}
    p, q = v.numerator, v.denominator
    out = {f"{p}/{q}", f"\\frac{{{p}}}{{{q}}}", f"\\dfrac{{{p}}}{{{q}}}"}
    qq = q
    for f in (2, 5):
        while qq % f == 0:
            qq //= f
    if qq == 1:
        out.add(f"{float(v):.12f}".rstrip("0").rstrip("."))
    else:
        out |= {f"{float(v):.{k}f}" for k in (2, 3, 4)}
    return out


def occurrences(text, v, lo, hi):
    n = 0
    for f in forms(v):
        # whole numbers only: 23 is not inside 23,375 or 1{,}023
        n += len(re.findall(rf"(?<![\d.])(?<!\d,)(?<!\{{,\}}){re.escape(f)}(?!\d|[.,]\d|\{{,\}})",
                            text[lo:hi]))
    return n


def fmt(v):
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def annotate(claim, text, window_end, final, repeats=()):
    # text is the whole trace, - normalised so offsets are unchanged. repeats: spans of the
    # same wrong claim written again, blanked so repetition alone does not count as reuse
    t = text.replace("−", "-")
    for a, b in repeats:
        t = t[:a] + " " * (b - a) + t[b:]
    s, e = claim["start"], claim["end"]
    true, written = claim["true"], claim["written"]
    after = occurrences(t, written, e, len(t))
    before = occurrences(t, written, 0, s)
    corrected = occurrences(t, true, e, window_end) > 0
    try:
        in_final = final is not None and evaluate(normalize(final)) == written
    except Skip:
        in_final = False
    return {
        "statement": claim["statement"], "lhs": claim["lhs"],
        "true_value": fmt(true), "written_value": fmt(written),
        "abs_error": float(abs(true - written)),
        "rel_error": float(abs(true - written) / abs(true)) if true else None,
        "written_reused_after": after, "written_seen_before": before,
        "load_bearing": after > 0 and before == 0,
        "self_corrected": corrected,
        "correction_cue": bool(CUE.search(t[e:window_end])),
        "written_is_final_answer": in_final,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="B")
    ap.add_argument("--traces", default=None, help="default data/traces_<tag>.jsonl")
    ap.add_argument("--out-dir", default="data/injection")
    ap.add_argument("--no-tokens", action="store_true", help="skip the tokenizer; window in chars")
    ap.add_argument("--window", type=int, default=500, help="tokens after a slip checked for a correction")
    ap.add_argument("--examples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--go", type=int, default=30, help="load-bearing, uncorrected slips in wrong traces needed")
    a = ap.parse_args()
    path = a.traces or f"data/traces_{a.tag}.jsonl"
    traces = [json.loads(line) for line in open(path)]
    tok = None
    if not a.no_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)

    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, f"slips_{a.tag}.jsonl")
    stats = {k: Counter() for k in ("wrong", "correct")}
    skips = {k: Counter() for k in ("wrong", "correct")}
    excluded = Counter()
    rows = []
    for t in traces:
        if t["trunc"] or t["n_gen"] >= MAX_NEW:
            excluded["truncated"] += 1
            continue
        text = t["text"]
        cut = text.find(THINK_END)
        if cut < 0:
            excluded["no </think>"] += 1
            continue
        label = "correct" if t["is_correct"] else "wrong"
        st = stats[label]
        st["traces"] += 1
        offsets, seen, had = None, set(), False
        claims = find_claims(text[:cut])
        spans = {}
        for c in claims:
            if c["status"] == "slip":
                spans.setdefault((c["lhs"], fmt(c["written"])), []).append((c["start"], c["end"]))
        for c in claims:
            if c["status"].startswith("skip:"):
                skips[label][c["status"][5:]] += 1
                continue
            st["claims"] += 1
            if c["status"] != "slip":
                continue
            if tok is not None and offsets is None:
                offsets = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
                ends = [b for _, b in offsets]
            if offsets is not None:
                k0, k1 = bisect.bisect_right(ends, c["start"]), bisect.bisect_left(ends, c["end"])
                window_end = ends[min(k1 + a.window, len(ends) - 1)]
            else:
                k0, window_end = None, c["end"] + 4 * a.window
            key = (c["lhs"], fmt(c["written"]))
            first = key not in seen
            seen.add(key)
            row = {"trace_id": t["trace_id"], "problem_id": t["problem_id"], "level": t.get("level"),
                   "trace_label": label, "char_offset": c["start"], "token_offset": k0,
                   "window": f"{a.window} tokens" if offsets is not None else f"{4 * a.window} chars (~{a.window} tokens)",
                   "first_occurrence": first,
                   **annotate(c, text, window_end, t.get("answer"),
                              [s for s in spans[key] if s != (c["start"], c["end"])])}
            rows.append(row)
            if first:
                had = True
                st["slips"] += 1
                st["load_bearing"] += row["load_bearing"]
                st["self_corrected"] += row["self_corrected"]
                st["uncorrected"] += not row["self_corrected"]
                st["load_bearing_uncorrected"] += row["load_bearing"] and not row["self_corrected"]
                st["written_is_final_answer"] += row["written_is_final_answer"]
            else:
                st["repeats"] += 1
        st["traces_with_slip"] += had

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {"traces_file": path, "excluded": dict(excluded), "window_tokens": a.window,
               "token_offsets": tok is not None,
               "by_label": {k: {**dict(v), "per_1k_claims": round(1000 * v["slips"] / v["claims"], 2)
                                if v["claims"] else None, "skipped": dict(skips[k].most_common())}
                            for k, v in stats.items()}}
    json.dump(summary, open(os.path.join(a.out_dir, f"slip_summary_{a.tag}.json"), "w"), indent=2)

    print(f"excluded {dict(excluded)}")
    cols = ["traces", "claims", "slips", "per_1k", "traces_with_slip", "load_bearing", "self_corrected",
            "uncorrected", "load_bearing_uncorrected", "written_is_final_answer", "repeats"]
    print(f"\n{'':>8}" + "".join(f"{c:>{max(len(c), 6) + 2}}" for c in cols))
    for k, v in stats.items():
        vals = [v[c] if c != "per_1k" else summary["by_label"][k]["per_1k_claims"] for c in cols]
        print(f"{k:>8}" + "".join(f"{str(x):>{max(len(c), 6) + 2}}" for c, x in zip(cols, vals)))
    print("\nslips counted once per (trace, left side, written value); repeats listed separately.")
    print("load_bearing = written value reappears later (rewriting the same wrong claim does not count) "
          "and did not appear anywhere before the claim, which is stricter than 'reused'.")
    print(f"self_corrected = true value restated within {a.window} tokens after the claim"
          + (" (approximated as chars; run with the tokenizer for the real window)." if tok is None else "."))
    for k in ("wrong", "correct"):
        print(f"skipped as not a clean claim ({k}): {dict(skips[k].most_common(8))}")
    n = stats["wrong"]["load_bearing_uncorrected"]
    print(f"\ngo/no-go (before your manual precision check): {n} load-bearing, uncorrected slips in wrong "
          f"traces vs {a.go} needed -> {'GO, pending the precision check' if n >= a.go else 'NO-GO'}")

    pool = [r for r in rows if r["trace_label"] == "wrong" and r["first_occurrence"]]
    by_id = {t["trace_id"]: t["text"] for t in traces}
    print(f"\n{min(a.examples, len(pool))} random wrong-trace slips (seed {a.seed}); all in {out_path}\n")
    for r in random.Random(a.seed).sample(pool, min(a.examples, len(pool))):
        text, s = by_id[r["trace_id"]], r["char_offset"]
        e = s + len(r["statement"])
        ctx = (text[max(0, s - 120):s] + " >>>" + text[s:e] + "<<< " + text[e:e + 120]).replace("\n", " ⏎ ")
        print(f"[{r['trace_id']} @ char {s}, token {r['token_offset']}] true {r['true_value']}, written "
              f"{r['written_value']} | load-bearing {r['load_bearing']}, self-corrected {r['self_corrected']}, "
              f"cue {r['correction_cue']}\n    {ctx}\n")


if __name__ == "__main__":
    main()
