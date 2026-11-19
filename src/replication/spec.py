# engine, sampling and model profiles for the replication pipeline (gpu_prep.py, confidence.py,
# scripts/replication_modal.py). E0_MODEL picks the profile; unset is DeepSeek-R1-Distill-Qwen-1.5B.
import hashlib, json, os

# ---- engine and sampling: sources and re-runs both use these ----
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
TOKENIZER_REVISION = MODEL_REVISION
VLLM_VERSION = "0.10.2"
TRANSFORMERS_VERSION = "4.56.2"  # 5.x drops a BOS and breaks vllm 0.10.2
DTYPE = "float16"
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_NEW = 32768  # whole-trace budget; a re-run from cut c gets MAX_NEW - c
# "one": exactly one leading BOS. prompt ids go to vLLM as ids, never as a string (a string gets a second BOS)
BOS = "one"
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."

# ---- cut and re-runs ----
CUT = 4096  # tokens of the trace kept; the hidden state is read at index CUT - 1
K_PREFIX = 8  # re-runs per cut
LAYER = 20  # hidden_states[LAYER + 1]
HIDDEN = 1536
THINK_END = "</think>"
# a prefix "states" its answer when the trace's final answer follows one of these cues within the window
ANSWER_CUES = ("final answer", "answer is", "answer:")
ANSWER_CUE_WINDOW = 40

# ---- profiles: key -> (HF repo, revision, dtype, layer, hidden size, BOS rule) ----
# layer is about 20/28 of the depth. BOS "template": what the chat template writes, never two
PROFILES = {
    "r1-1.5b": (MODEL, MODEL_REVISION, DTYPE, LAYER, HIDDEN, BOS),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "bfloat16", 20, 2048, "template"),
    "ministral3-3b": ("mistralai/Ministral-3-3B-Reasoning-2512", "4a36357c811bf511a7b625d132e12f22408aac91",
                      "bfloat16", 19, 3072, "one"),
}
# per-profile stack and prompt. Ministral 3 needs vLLM >= 0.12 and transformers 5; vLLM 0.32.0 pins
# transformers >= 5.16.1, < 5.20, and loads it the model card's way (Mistral tokenizer, config and weights).
# the checkpoint is image-text, so extraction reads its text decoder. its template writes SYSTEM_PROMPT.txt,
# which opens [THINK]: no think tag is forced, and [THINK] / [/THINK] become <think> / </think> in prefixes
ENGINE = {
    "ministral3-3b": {"vllm": "0.32.0", "transformers": "5.19.0",
                      "vllm_args": {"tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral"},
                      "think_tags": ("[THINK]", "[/THINK]"), "force_think": False, "multimodal": True,
                      # FlashInfer's top-p sampler needs nvcc, which the image lacks; vLLM's PyTorch sampler instead
                      "env": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
                      "prompt_has": "[SYSTEM_PROMPT]# HOW YOU SHOULD THINK AND ANSWER"},
}
VLLM_ARGS = {}  # extra vllm.LLM keyword arguments
THINK_TAGS = ("<think>", "</think>")
FORCE_THINK = True  # append <think>\n unless the template already ends with it
MULTIMODAL = False  # hidden states from model.model.language_model, not model.model
PROMPT_HAS = None  # a string the rendered prompt must contain
ENV = {}  # extra environment variables for the Modal image

MODEL_KEY = os.environ.get("E0_MODEL") or "r1-1.5b"
if MODEL_KEY not in PROFILES:
    raise SystemExit(f"E0_MODEL={MODEL_KEY!r} is not one of {', '.join(PROFILES)}")
MODEL, MODEL_REVISION, DTYPE, LAYER, HIDDEN, BOS = PROFILES[MODEL_KEY]
TOKENIZER_REVISION = MODEL_REVISION
_e = ENGINE.get(MODEL_KEY, {})
VLLM_VERSION = _e.get("vllm", VLLM_VERSION)
TRANSFORMERS_VERSION = _e.get("transformers", TRANSFORMERS_VERSION)
VLLM_ARGS = _e.get("vllm_args", VLLM_ARGS)
THINK_TAGS = _e.get("think_tags", THINK_TAGS)
FORCE_THINK = _e.get("force_think", FORCE_THINK)
MULTIMODAL = _e.get("multimodal", MULTIMODAL)
PROMPT_HAS = _e.get("prompt_has", PROMPT_HAS)
ENV = _e.get("env", ENV)

KEYS = ["MODEL", "MODEL_REVISION", "TOKENIZER_REVISION", "VLLM_VERSION", "TRANSFORMERS_VERSION", "VLLM_ARGS", "DTYPE",
        "TEMPERATURE", "TOP_P", "MAX_NEW", "BOS", "INSTRUCTION", "CUT", "K_PREFIX", "LAYER", "THINK_TAGS",
        "FORCE_THINK", "PROMPT_HAS"]


def spec_sha():
    # written into every output file, so a file names the settings that made it
    g = globals()
    return hashlib.sha256(json.dumps({k: g[k] for k in KEYS}, sort_keys=True, default=str).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()
