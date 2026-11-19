# replication profiles (E0_MODEL) and the Ministral port: per-profile engine, prompt render, think tags,
# the level filter, the bank plan, and run_rollouts' request-id mapping and idle guard.
# plain script, CPU only, no torch or vLLM, exits non-zero on failure.
import json, os, subprocess, sys, tempfile, types

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REPL = os.path.join(ROOT, "src", "replication")
fails = []


def check(name, cond, extra=None):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(extra)) if extra is not None else ''}")
    if not cond:
        fails.append(name)


def spec_under(key):
    # spec reads E0_MODEL at import, so each profile is loaded in its own interpreter
    code = ("import json, sys; sys.path.insert(0, %r); import spec as s; print(json.dumps({k: getattr(s, k) for k in "
            "('MODEL', 'LAYER', 'HIDDEN', 'BOS', 'VLLM_VERSION', 'TRANSFORMERS_VERSION', 'VLLM_ARGS', 'THINK_TAGS', "
            "'FORCE_THINK', 'MULTIMODAL', 'PROMPT_HAS', 'THINK_END', 'ENV')} | {'sha': s.spec_sha()}))" % REPL)
    env = {**os.environ, "E0_MODEL": key} if key else {k: v for k, v in os.environ.items() if k != "E0_MODEL"}
    return json.loads(subprocess.check_output([sys.executable, "-c", code], env=env))


print("1. profiles")
base, qwen, mini = spec_under(""), spec_under("qwen3-1.7b"), spec_under("ministral3-3b")
check("confirmation spec hash unchanged by the port", base["sha"]
      == "f653c2e774a37981ecd9fc17cdeafe8e5aa13894ab0cb6ff4196013d2a4e60ca", base["sha"])
check("qwen3-1.7b keeps the old stack, <think> and no engine args",
      (qwen["VLLM_VERSION"], qwen["TRANSFORMERS_VERSION"], qwen["VLLM_ARGS"], qwen["THINK_TAGS"], qwen["FORCE_THINK"])
      == ("0.10.2", "4.56.2", {}, ["<think>", "</think>"], True), qwen)
check("ministral: repo, layer 19 of 26, hidden 3072, one BOS",
      (mini["MODEL"], mini["LAYER"], mini["HIDDEN"], mini["BOS"])
      == ("mistralai/Ministral-3-3B-Reasoning-2512", 19, 3072, "one"), mini)
check("ministral: vLLM 0.32.0 + transformers 5.19.0, Mistral formats",
      (mini["VLLM_VERSION"], mini["TRANSFORMERS_VERSION"]) == ("0.32.0", "5.19.0")
      and mini["VLLM_ARGS"] == {"tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral"}, mini)
check("ministral: [THINK] tags, nothing forced, image-text checkpoint, system prompt required",
      mini["THINK_TAGS"] == ["[THINK]", "[/THINK]"] and not mini["FORCE_THINK"] and mini["MULTIMODAL"]
      and mini["PROMPT_HAS"].startswith("[SYSTEM_PROMPT]") and mini["THINK_END"] == "</think>", mini)
check("ministral: FlashInfer sampler off in its image; other profiles add no env",
      mini["ENV"] == {"VLLM_USE_FLASHINFER_SAMPLER": "0"} and qwen["ENV"] == {} and base["ENV"] == {}, mini["ENV"])
check("profiles hash apart", len({base["sha"], qwen["sha"], mini["sha"]}) == 3)
bad = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {REPL!r}); import spec"],
                     env={**os.environ, "E0_MODEL": "nope"}, capture_output=True, text=True)
check("an unknown E0_MODEL is refused", bad.returncode != 0 and "ministral3-3b" in bad.stderr, bad.stderr[-200:])

# the rest runs in this interpreter under the Ministral profile
os.environ["E0_MODEL"] = "ministral3-3b"
sys.path.insert(0, REPL)
sys.path.insert(0, os.path.join(ROOT, "src", "recovery"))
import spec  # noqa: E402
import gpu_prep as G  # noqa: E402
import bundle_lib as BL  # noqa: E402
for name, attrs in (("torch", {}), ("datasets", {"load_dataset": None}),
                    ("math_verify", {"parse": lambda x: x, "verify": lambda a, b: a == b}),
                    ("transformers", {"AutoTokenizer": object, "AutoModelForCausalLM": object})):
    try:
        __import__(name)
    except ImportError:  # generate.py imports these at module level; nothing here needs them
        sys.modules[name] = types.SimpleNamespace(__name__=name, **attrs)
import run_rollouts as RR  # noqa: E402
import pandas as pd  # noqa: E402

print("\n2. prompt and think tags")


class Tok:
    # an HF-style tokenizer: renders BOS, an optional system prompt and the user turn; ids are a stand-in
    bos_token_id = 1

    def __init__(self, system=True):
        self.system = system

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        sp = "[SYSTEM_PROMPT]# HOW YOU SHOULD THINK AND ANSWER\n\n...[THINK]x[/THINK]y[/SYSTEM_PROMPT]" if self.system else ""
        return f"<s>{sp}[INST]{msgs[0]['content']}[/INST]"

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [1, 17, len(text)]}


p = G.build_prompt(Tok(), "What is 1+1?")
check("HF path: user turn with the instruction, no <think> appended",
      p.endswith("What is 1+1?" + spec.INSTRUCTION + "[/INST]") and "<think>" not in p, p[-60:])
try:
    G.build_prompt(Tok(system=False), "What is 1+1?")
    check("a prompt without the system prompt is refused (what transformers 5 rendered)", False)
except SystemExit as e:
    check("a prompt without the system prompt is refused (what transformers 5 rendered)", "SYSTEM_PROMPT" in str(e))
check("canon: [THINK] / [/THINK] -> <think> / </think>",
      G.canon("[THINK]a\n\nb[/THINK]c \\boxed{2}") == "<think>a\n\nb</think>c \\boxed{2}")
live = BL.cut_rules(G.canon("[THINK]so 2 + 2"), "4", True)
closed = BL.cut_rules(G.canon("[THINK]so 4[/THINK]The answer"), "4", True)
check("a cut inside [THINK] is live", live["live"] and live["pre_answer"], live)
check("a cut after [/THINK] is not live", not closed["live"] and not closed["pre_answer_think_boxed"], closed)
check("BOS rule one: [1, 17, ...] ok, [1, 1, ...] not", G.bos_ok(Tok(), [1, 17, 5]) and not G.bos_ok(Tok(), [1, 1, 17]))

try:
    import mistral_common  # noqa: F401
    have_mc = True
except ImportError:
    have_mc = False
if not have_mc:
    print("skip  MistralTok: mistral_common not installed")
else:
    from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

    class MT:
        # records the request; one id per chunk / message, so the rendered order is checkable
        def encode_chat_completion(self, req):
            self.req = req
            return types.SimpleNamespace(tokens=[1, 17, 34, 35, 18, 3, 4])

    class Raw:
        # Tekken-like: ids below 1000 are control tokens
        bos_id = 1
        specials = {"<s>": 1, "</s>": 2, "[INST]": 3, "[/INST]": 4, "[TOOL_CALLS]": 9, "[SYSTEM_PROMPT]": 17,
                    "[/SYSTEM_PROMPT]": 18, "[ARGS]": 32, "[THINK]": 34, "[/THINK]": 35}

        def is_special(self, t):
            return t in self.specials if isinstance(t, str) else t < 1000

        def get_special_token(self, t):
            return self.specials[t]

        def decode(self, ids, special_token_policy):
            self.policy = special_token_policy
            names = {1: "<s>", 2: "</s>", 17: "[SYSTEM_PROMPT]# HOW YOU SHOULD THINK AND ANSWER", 34: "[THINK]",
                     35: "[/THINK]", 18: "[/SYSTEM_PROMPT]", 3: "[INST]q", 4: "[/INST]", 1001: "x", 1002: " = 2"}
            return "".join(names[i] for i in ids)

    mt = G.MistralTok.__new__(G.MistralTok)
    mt.mt, mt.raw, mt.bos_token_id = MT(), Raw(), 1
    mt.system = "# HOW YOU SHOULD THINK AND ANSWER\n\nFirst draft.[THINK]Your thoughts.[/THINK]Here, respond."
    text, ids = G.render(mt, "What is 1+1?")
    sysm, user = mt.mt.req.messages
    kinds = [c.type for c in sysm.content]
    check("MistralTok: system = text + closed thinking + text, split at the [THINK] block",
          [str(k).split(".")[-1].lower() for k in kinds] == ["text", "thinking", "text"]
          and sysm.content[1].thinking == "Your thoughts." and sysm.content[1].closed
          and sysm.content[0].text.endswith("First draft.") and sysm.content[2].text == "Here, respond.", kinds)
    check("MistralTok: user turn = problem + instruction", user.content == "What is 1+1?" + spec.INSTRUCTION)
    check("MistralTok: ids are mistral-common's, text is their decode with specials kept",
          ids == [1, 17, 34, 35, 18, 3, 4] and text.startswith("<s>[SYSTEM_PROMPT]#")
          and mt.raw.policy == SpecialTokenPolicy.KEEP, text)
    got = mt.decode([34, 1001, 35, 1002, 2], skip_special_tokens=True)
    check("MistralTok: skip_special_tokens keeps [THINK] / [/THINK] and drops </s>, as vLLM 0.32's sample text does",
          got == "[THINK]x[/THINK] = 2", got)
    check("MistralTok: without skip, every control token is kept", mt.decode([34, 1001, 2]) == "[THINK]x</s>")
    mt.system = "no think block"
    try:
        G.render(mt, "q")
        check("MistralTok: a system prompt without [THINK] is refused", False)
    except SystemExit:
        check("MistralTok: a system prompt without [THINK] is refused", True)
real_mt = G.MistralTok
try:
    G.MistralTok = lambda: "mistral-common tokenizer"
    check("Ministral profile tokenizes with mistral-common, not transformers", G.tokenizer() == "mistral-common tokenizer")
finally:
    G.MistralTok = real_mt

print("\n3. level filter")
with tempfile.TemporaryDirectory() as d:
    tr = os.path.join(d, "traces.jsonl")
    with open(tr, "w") as f:
        for pid, lv in ((1, 5), (2, 4), (3, 5), (4, 3), (5, 5)):
            for k in range(2):
                f.write(json.dumps({"problem_id": pid, "problem": f"p{pid}", "gold": "1", "level": lv,
                                    "subject": "Algebra", "uid": f"u{pid}"}) + "\n")
    out = os.path.join(d, "l5.jsonl")
    run = lambda *x: subprocess.run([sys.executable, os.path.join(REPL, "dev_problems.py"), "--traces", tr, *x],
                                    capture_output=True, text=True)
    r = run("--out", out, "--levels", "5")
    rows = [json.loads(x) for x in open(out)]
    check("--levels 5 keeps the level-5 problems, one row each",
          r.returncode == 0 and sorted(int(x["problem_id"]) for x in rows) == [1, 3, 5], r.stdout + r.stderr)
    r = run("--out", os.path.join(d, "all.jsonl"))
    check("no --levels keeps them all", sum(1 for _ in open(os.path.join(d, "all.jsonl"))) == 5)
    check("an empty level is refused", run("--out", os.path.join(d, "x.jsonl"), "--levels", "9").returncode != 0)

print("\n4. bank plan")
with tempfile.TemporaryDirectory() as d:
    src = pd.DataFrame({"problem_id": ["1", "1", "2"], "n": [5000, 10000, 2_000_000 - 15000]})
    pool = pd.DataFrame({"problem_id": ["1", "1"], "checkpoint": [4096, 4096]})
    est = {("1", 4096): 2500}
    s = G.bank_plan(src, pool, est, os.path.join(d, "sources.jsonl"))
    check("plan without a ledger: tokens only", "2 cuts in 1 problems x 8 = ~0.0M" in s and "$" not in s, s)
    json.dump({"a": [{"usd": 1.0}], "b": [{"usd": 0.5}, {"usd": 0.5}]}, open(os.path.join(d, "sources.jsonl.ledger.json"), "w"))
    s = G.bank_plan(src, pool, est, os.path.join(d, "sources.jsonl"))
    # 2 cuts x 8 x 2500 = 40k tokens; $2 over 2M tokens = $1/M -> $0.04
    check("plan with a ledger: $/M from the sources, bank priced at it", "($1.00/M)" in s and "~$0.04" in s, s)

print("\n5. run_rollouts under a newer vLLM")
check("own_id: exact id", RR.own_id({"t0/1#0": 1}, "t0/1#0") == "t0/1#0")
check("own_id: a suffixed id maps back", RR.own_id({"t0/1#0": 1, "t0/12#0": 2}, "t0/12#0-a1b2") == "t0/12#0")
check("own_id: t0/1 does not claim t0/12", RR.own_id({"t0/1#0": 1}, "t0/12#0-x") is None)
check("own_id: unknown id", RR.own_id({"t0/1#0": 1}, "other") is None)


class Engine:
    # finishes each request in one step, reporting ids with a suffix as newer vLLM may
    def __init__(self, suffix="", lose=False):
        self.live, self.suffix, self.lose = [], suffix, lose

    def add_request(self, rid, prompt, sp):
        self.live.append((rid, sp))

    def has_unfinished_requests(self):
        return bool(self.live)

    def step(self):
        outs = [] if self.lose else [types.SimpleNamespace(request_id=rid + self.suffix, finished=True, outputs=[
            types.SimpleNamespace(index=i, text="\\boxed{12}", token_ids=[7, 8], finish_reason="stop")
            for i in range(sp.n)]) for rid, sp in self.live]
        self.live = []
        return outs


class LLM:
    def __init__(self, engine):
        self.llm_engine = engine

    def get_tokenizer(self):
        return types.SimpleNamespace(decode=lambda ids: "")


rec = lambda i: {"prefix_id": f"t0/{i}", "source": "confirmation", "problem_id": str(i), "trace_id": None,
                 "checkpoint": "t0", "cut_token_idx": 0, "gold_answer": "12", "prompt_token_ids": [1, 17],
                 "prefix_gen_token_ids": [], "max_tokens": 50, "est_cont_tokens": 10}
args = types.SimpleNamespace(inflight=8, temp=0.7, top_p=0.95, max_tokens_cap=0, seed=0, round=0, n=2,
                             max_minutes=0, max_num_seqs=4, save_token_ids="full")
params = lambda **kw: types.SimpleNamespace(**kw)
with tempfile.TemporaryDirectory() as d:
    out = os.path.join(d, "s.jsonl")
    stop, _ = RR.run(LLM(Engine(suffix="-9f3c")), params, [(rec(1), 2), (rec(12), 2)], args, out, "h")
    rows = [json.loads(x) for x in open(out)]
    check("suffixed ids: every prefix written once, with its ids",
          sorted(r["prefix_id"] for r in rows) == ["t0/1", "t0/12"]
          and all(s["token_ids"] == [7, 8] and s["correct"] for r in rows for s in r["samples"]), rows)
    try:
        RR.run(LLM(Engine(lose=True)), params, [(rec(2), 2)], args, os.path.join(d, "t.jsonl"), "h")
        check("idle engine with prefixes outstanding raises instead of spinning", False)
    except RuntimeError as e:
        check("idle engine with prefixes outstanding raises instead of spinning", "t0/2#0" in str(e), e)
check("tail estimate: before any engine stats, max_num_seqs share the throughput",
      RR.tail_seconds(28672, 2200, 0, 256) == 28672 / (2200 / 256))
check("tail estimate: once vLLM reports its peak running count, that count is used",
      abs(RR.tail_seconds(28672, 2200, 90, 256) - 28672 / (2200 / 90)) < 1e-9)
ap = RR.add_args(__import__("argparse").ArgumentParser())
a = ap.parse_args(["--engine-args", json.dumps(spec.VLLM_ARGS), "--vllm-version", spec.VLLM_VERSION])
check("runner takes the version and the engine args", a.vllm_version == "0.32.0"
      and json.loads(a.engine_args)["load_format"] == "mistral")
check("runner default version is still the confirmation one", ap.parse_args([]).vllm_version == "0.10.2")

print(f"\n{'all passed' if not fails else f'{len(fails)} FAILED: {fails}'}")
sys.exit(1 if fails else 0)
