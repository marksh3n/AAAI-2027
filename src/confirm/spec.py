# frozen design of the monitor / policy comparison for the confirmation run. every confirm/ script
# reads its constants from here, and docs/CONFIRMATION_SPEC.md states the same design in words.
# changing anything below after confirmation data exists makes the run exploratory.
# Jaden's slots (population, margin, sample size) are None until he fills them; analyze.py runs
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
]
# S7 fallback, pre-registered 2026-10-27: if the confirmation data has fewer than this many mixed problems
# among live "stated" pre-answer cuts at 6144 (a label count, no scores), S7 is reported as effect + CI
# only, makes no claim and leaves the Holm family. 500 is roughly what 80% power on the dev-sized effect
# (+0.065) needs, scaled from the dev CI width
S7_MIN_MIXED = 500
ALPHA = 0.05
N_BOOT = 2000
BOOT_SEED = 0

# ---- rollout bank (story-1 diagnostic and the replacement outcomes) ----
K_PREFIX = 8  # continuations per live (trace, checkpoint)
K_T0 = 8  # fresh samples per problem: restart replacements and the aggregation votes

# ---- Jaden's slots: parameters, not decisions made here. filled 2026-10-27 from his planning package
# (Jaden_Accuracy_Planning_Handoff_q025.zip, protocol status "awaiting_policy_and_freshness_check") and
# PENDING TEAM CHOICE: analyze.py gives no primary verdict and gpu_prep.py refuses to start until
# "status" is "frozen". docs/CONFIRMATION_SPEC.md section 11 lists where his design and this spec conflict
JADEN = {
    "status": "pending_team_choice",
    "source": "Jaden_Accuracy_Planning_Handoff_q025.zip, protocol/confirmation_protocol.json (2026-10-26)",
    "problem_ids_csv_sha256": "6609db7bea9205ab00d3c733edb1d4e6d415b5348001a0112e2f1e40fd050d86",
    # confirmation_after_pilot_1000.csv, original reservation positions 101-1100 (his recommended disjoint set)
    "pilot_ids_csv_sha256": "aef5c628b0984a1a98615969af62726fd02bb7bad9edd44d53e9e9b30ae2644e",  # pilot_first_100.csv
    "problems_file_sha256": None,  # the jsonl gpu_prep.py reads (prompt + gold per id); built from his ids, not yet
    "n_problems": 1000,
    "samples_per_problem": None,  # his design: 1 outcome per arm per problem; ours needs several (conflict 3)
    "margin_pp": 5.0,  # his: a useful gain, powered for and ruled out above; ours: a superiority margin (conflict 7)
    "endpoint": "paired final-answer accuracy, monitor vs aggregation, one outcome per arm per problem: exact "
                "McNemar superiority (two-sided .05) and one-sided Tango score test ruling out a difference >= 5 pp",
    "expected_mixed_6144_stated": 0,  # one sample per problem has no within-problem pairs: S7 not evaluable
}


def frozen():
    # Siddhant's part (everything above except JADEN), for manifests and the hash in the doc. Jaden's
    # slots are hashed apart (jaden_sha), so filling them does not invalidate the fitted monitors
    keys = ["MODEL", "MODEL_REVISION", "TOKENIZER_REVISION", "VLLM_VERSION", "TRANSFORMERS_VERSION", "DTYPE",
            "TEMPERATURE", "TOP_P", "MAX_NEW", "BOS", "INSTRUCTION", "CHECKPOINTS", "DECISION", "BACKTRACK_TO",
            "LAYER", "THINK_END", "ANSWER_CUES", "ANSWER_CUE_WINDOW", "PRE_ANSWER_PRIMARY", "PRE_ANSWER_SENSITIVITY",
            "MONITORS", "PRIMARY_MONITOR", "TREE_MONITOR", "Q_DEV", "FIT_SEED", "ACTIONS", "AGG_EXTRA",
            "PRIMARY", "SECONDARY", "S7_MIN_MIXED", "ALPHA", "N_BOOT", "BOOT_SEED", "K_PREFIX", "K_T0"]
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


def jaden_sha():
    return hashlib.sha256(json.dumps(JADEN, sort_keys=True, default=str).encode()).hexdigest()


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
