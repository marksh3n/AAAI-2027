# frozen design of the monitor / policy comparison for the confirmation run. every confirm/ script
# reads its constants from here, and docs/CONFIRMATION_SPEC.md states the same design in words.
# changing anything below after confirmation data exists makes the run exploratory.
# team slots (population, margin, sample size) stay None until filled; analyze.py runs
# on development data without them but refuses to print a confirmatory verdict.
import hashlib, json, os

# ---- engine and sampling: sources and rollouts both come from this, nothing else ----
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"  # HF main as of 2026-10-25 (unchanged since 2025-02-24)
TOKENIZER_REVISION = MODEL_REVISION  # same repo, same commit
VLLM_VERSION = "0.10.2"  # the Modal image the ceiling rollouts ran on
TRANSFORMERS_VERSION = "4.56.2"  # 5.x drops a BOS and breaks vllm 0.10.2
DTYPE = "float16"  # generation and extraction; the dev activations were computed in fp16
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_NEW = 32768  # whole-trace budget; a rollout from cut c gets MAX_NEW - c
# one BOS: the chat template's own. prompt ids = tokenizer(prompt, add_special_tokens=False) and
# are handed to vLLM as token ids, never as a string (a string prompt gets a second BOS)
BOS = "one"
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."

# ---- checkpoints: tokens of the trace observed, hidden state at index c - 1 ----
CHECKPOINTS = (2048, 4096, 6144)
DECISION = 4096  # the primary decision checkpoint
BACKTRACK_TO = {4096: 2048, 6144: 4096}  # backtrack from c restarts the trace at its earlier cut
LAYER = 20  # hidden_states[LAYER + 1]
# a trace is live at c when it is still generating after c tokens (n_gen > c, so no EOS among the
# first c) and has not closed its thinking (no </think> among the first c tokens). policies act on
# live traces only; everything else keeps its own outcome and cost.
THINK_END = "</think>"
# pre-answer at c (story 2, S7), decided 2026-10-25 after the dev dry run had shown S7 under the strict
# rule (disclosed in the spec doc). primary "stated": live, no \boxed, and the trace's final answer value
# not written within 40 characters after an answer cue on the same line. reported beside it, not tested:
# "strict" (the value anywhere, as a standalone number) and "think_boxed" (only </think> and \boxed)
ANSWER_CUES = ("final answer", "answer is", "answer:")
ANSWER_CUE_WINDOW = 40
PRE_ANSWER_PRIMARY = "stated"
PRE_ANSWER_SENSITIVITY = ("strict", "think_boxed")

# ---- model replication (added 2026-11-08): the env var E0_MODEL runs the same pipeline on another model.
# unset (or "r1-1.5b") is the confirmation model above and leaves every hash unchanged. any other key is a
# replication run, not the confirmation: no slot checks, cut at Q1_CUT only, its own Modal app and Volume.
# key: (HF repo, revision, dtype, layer, hidden size, BOS rule). layer is about 20/28 of the depth;
# "template": no BOS added beyond what the chat template writes, and never two
PROFILES = {
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "bfloat16", 20, 2048, "template"),
    "qwen3-4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", "bfloat16", 26, 2560, "template"),
    "phi4-mini-reasoning": ("microsoft/Phi-4-mini-reasoning", "0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a",
                            "bfloat16", 23, 3072, "template"),
    "r1-llama-8b": ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "6a6f4aa4197940add57724a7707d069478df56b1",
                    "bfloat16", 23, 4096, "one"),
}
MODEL_KEY = os.environ.get("E0_MODEL", "r1-1.5b")
REPLICATION = MODEL_KEY != "r1-1.5b"
HIDDEN = 1536
if REPLICATION:
    if MODEL_KEY not in PROFILES:
        raise SystemExit(f"E0_MODEL={MODEL_KEY!r} is not one of r1-1.5b, {', '.join(PROFILES)}")
    MODEL, MODEL_REVISION, DTYPE, LAYER, HIDDEN, BOS = PROFILES[MODEL_KEY]
    TOKENIZER_REVISION = MODEL_REVISION

# ---- monitors (fit on all development data by fit_monitors.py) ----
# name -> (probe_cmp model, probe_cmp feature, layer or None, checkpoints)
MONITORS = {
    "lr_raw_L20": ("lr", "raw", 20, CHECKPOINTS),  # primary
    "hgb_topk50_L20": ("hgb", "topk50", 20, (6144,)),  # pre-registered secondary (the dev 6144 candidate)
    "len": ("lr", "len", None, CHECKPOINTS),  # observable baseline: log1p(tokens, paragraphs)
    "tfidf": ("lr", "tfidf", None, CHECKPOINTS),  # observable baseline: word 1-2 gram tf-idf
}
PRIMARY_MONITOR = "lr_raw_L20"
TREE_MONITOR = "hgb_topk50_L20"
# operating point, fixed before any policy result was seen: flag the top quarter of live traces.
# the threshold is the (1 - Q_DEV) quantile of the monitor's dev scores on live traces, all problems
Q_DEV = 0.25
FIT_SEED = 0  # inner CV for C, the HGB random_state, the OOF folds used for thresholds

# ---- policies ----
ACTIONS = ("restart", "backtrack")  # restart: fresh sample from the prompt; backtrack: from BACKTRACK_TO[c]
AGG_EXTRA = (2, 4)  # aggregation arm: m extra fresh samples + plurality vote; m=2 (maj@3) is primary
PRIMARY = {"checkpoint": DECISION, "action": "restart", "monitor": PRIMARY_MONITOR}
# secondary family, Holm over these at ALPHA. "vs" is the token-matched comparator.
SECONDARY = [
    {"id": "S1", "checkpoint": 4096, "action": "backtrack", "monitor": PRIMARY_MONITOR, "vs": "random"},
    {"id": "S2", "checkpoint": 2048, "action": "restart", "monitor": PRIMARY_MONITOR, "vs": "random"},
    {"id": "S3", "checkpoint": 6144, "action": "restart", "monitor": PRIMARY_MONITOR, "vs": "random"},
    {"id": "S4", "checkpoint": 6144, "action": "restart", "monitor": TREE_MONITOR, "vs": "random"},
    {"id": "S5", "checkpoint": 6144, "action": "backtrack", "monitor": TREE_MONITOR, "vs": "random"},
    {"id": "S6", "checkpoint": 4096, "action": "restart", "monitor": PRIMARY_MONITOR, "vs": "len"},
    {"id": "S7", "checkpoint": 6144, "kind": "auroc_pre_answer", "rule": PRE_ANSWER_PRIMARY, "monitor": TREE_MONITOR,
     "vs": "len"},
    # added 2026-11-02 from dev exploration X4: a source still running at T tokens is aborted and replaced by fresh
    # t0-bank samples (at most R restarts, the last runs out), every token charged; vs the better, at the same
    # tokens per attempt, of the plurality-vote frontier (own answer + m bank samples, m = 0..4) and the
    # short-1@k frontier (first of k parallel samples to finish, k = 2..5; arXiv 2505.17813), mixtures included
    # (upper concave envelope of all baseline points)
    {"id": "S8", "kind": "timeout", "T": 8192, "R": 5, "vs": "best of vote and short-1@k frontiers"},
]
TIMEOUT_ORDERS = 200  # random draw orders of each problem's t0 bank, seeded BOOT_SEED
# S7 fallback, pre-registered 2026-10-27: if the confirmation data has fewer than this many mixed problems
# among live "stated" pre-answer cuts at 6144 (a label count, no scores), S7 is reported as effect + CI
# only, makes no claim and leaves the Holm family. 500 is roughly what 80% power on the dev-sized effect
# (+0.065) needs, scaled from the dev CI width
S7_MIN_MIXED = 500
ALPHA = 0.05
N_BOOT = 2000
BOOT_SEED = 0

# ---- question 1 family (added 2026-11-02, before any confirmation data): how much of the within-problem
# rollout-outcome variance the prefix fixes (D), and how much of it a readout or a short forward run captures.
# claim only if all three hold (intersection-union); each bound is a 95% percentile bootstrap bound at Q1_CUT
#   T1  D lower bound > D_MIN
#   T2  primary monitor's capture upper bound < CAPTURE_MAX
#   T3  FWD_K forward runs capped at FWD_CAP keep a share of what the same runs capture uncapped: lower bound >
#       RETAIN_MIN. (uncapped runs vs a readout is not a test: their capture is about D / (D + (1 - D) / FWD_K)
#       by construction, so it is reported, not tested)
Q1_CUT = 4096
D_MIN = 0.2
CAPTURE_MAX = 0.25
FWD_K = 2
FWD_CAP = 4096
RETAIN_MIN = 0.5

# ---- rollout bank (story-1 diagnostic and the replacement outcomes) ----
K_PREFIX = 8  # continuations per live (trace, checkpoint)
K_T0 = 8  # fresh samples per problem: restart replacements and the aggregation votes

# ---- team slots: parameters, not decisions made here. filled 2026-10-27 from the planning package
# (accuracy planning zip, q 0.25, protocol status "awaiting_policy_and_freshness_check") and
# PENDING TEAM CHOICE: analyze.py gives no primary verdict and gpu_prep.py refuses to start until
# "status" is "frozen". docs/CONFIRMATION_SPEC.md lists where the planning design and this spec conflict
SLOTS = {
    "status": "pending_team_choice",
    "source": "accuracy planning zip sha256 7aef6367060ef35c6f08745c8826350119d581c8eb89df52d0091e063496aa25, "
              "protocol/confirmation_protocol.json (2026-10-26)",
    "problem_ids_csv_sha256": "6609db7bea9205ab00d3c733edb1d4e6d415b5348001a0112e2f1e40fd050d86",
    # confirmation_after_pilot_1000.csv, original reservation positions 101-1100 (recommended disjoint set)
    "pilot_ids_csv_sha256": "aef5c628b0984a1a98615969af62726fd02bb7bad9edd44d53e9e9b30ae2644e",  # pilot_first_100.csv
    "problems_file_sha256": None,  # the jsonl gpu_prep.py reads (prompt + gold per id); build_problems.py, not yet run
    "n_problems": 1000,
    "samples_per_problem": None,  # planning design: 1 outcome per arm per problem; ours needs several (conflict 3)
    "margin_pp": 5.0,  # a useful gain: ruled out when the one-sided 95% limit is below it; superiority is vs 0
    "endpoint": "paired final-answer accuracy, monitor vs aggregation, one outcome per arm per problem: exact "
                "McNemar superiority (two-sided .05) and one-sided Tango score test ruling out a difference >= 5 pp",
    "expected_mixed_6144_stated": 0,  # one sample per problem has no within-problem pairs: S7 not evaluable
}


def frozen():
    # frozen design (everything above except SLOTS), for manifests and the hash in the doc. the slots
    # are hashed apart (slots_sha), so filling them does not invalidate the fitted monitors
    keys = ["MODEL", "MODEL_REVISION", "TOKENIZER_REVISION", "VLLM_VERSION", "TRANSFORMERS_VERSION", "DTYPE",
            "TEMPERATURE", "TOP_P", "MAX_NEW", "BOS", "INSTRUCTION", "CHECKPOINTS", "DECISION", "BACKTRACK_TO",
            "LAYER", "THINK_END", "ANSWER_CUES", "ANSWER_CUE_WINDOW", "PRE_ANSWER_PRIMARY", "PRE_ANSWER_SENSITIVITY",
            "MONITORS", "PRIMARY_MONITOR", "TREE_MONITOR", "Q_DEV", "FIT_SEED", "ACTIONS", "AGG_EXTRA",
            "PRIMARY", "SECONDARY", "S7_MIN_MIXED", "ALPHA", "N_BOOT", "BOOT_SEED", "K_PREFIX", "K_T0",
            "Q1_CUT", "D_MIN", "CAPTURE_MAX", "FWD_K", "FWD_CAP", "RETAIN_MIN", "TIMEOUT_ORDERS"]
    g = globals()
    return {k: g[k] for k in keys}


def spec_sha():
    return hashlib.sha256(json.dumps(frozen(), sort_keys=True, default=str).encode()).hexdigest()


MONITOR_KEYS = ["MODEL", "MODEL_REVISION", "TOKENIZER_REVISION", "DTYPE", "CHECKPOINTS", "LAYER", "THINK_END",
                "MONITORS", "PRIMARY_MONITOR", "TREE_MONITOR", "Q_DEV", "FIT_SEED"]


def monitor_sha():
    # only what the monitor fit and thresholds depend on; analysis-only rules can change without a refit
    g = globals()
    return hashlib.sha256(json.dumps({k: g[k] for k in MONITOR_KEYS}, sort_keys=True, default=str).encode()).hexdigest()


def slots_sha():
    return hashlib.sha256(json.dumps(SLOTS, sort_keys=True, default=str).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_sha():
    # the confirm/ sources, so a manifest names the code that wrote it
    here = os.path.dirname(os.path.abspath(__file__))
    return {n: file_sha(os.path.join(here, n)) for n in sorted(os.listdir(here)) if n.endswith(".py")}
