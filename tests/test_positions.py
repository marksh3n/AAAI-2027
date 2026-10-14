# --positions scopes the ablation; each run writes its own JSON.
import os, sys, tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import cache
import probe_fixed as pf

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  pass  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


tmp = tempfile.mkdtemp()
os.makedirs(f"{tmp}/data")
os.chdir(tmp)
rng = np.random.default_rng(0)
rows = []
for pid in range(20):
    for k in range(6):
        y = int(rng.random() < 0.5)
        n_gen = int(rng.integers(600, 5000))
        for pos in pf.POSITIONS:
            if pos > n_gen:
                continue
            rows.append({"trace_id": f"{pid}_{k}", "problem_id": pid, "level": 4, "pos": pos,
                         "is_correct": y == 0, "n_gen": n_gen,
                         "prefix": " ".join(rng.choice("a b c wait so then".split(), 25)),
                         "restarts": float(rng.normal(4, 1))})
meta = pd.DataFrame(rows)
meta["row"] = np.arange(len(meta))
acts = rng.normal(size=(len(meta), len(pf.LAYERS), 20)).astype(np.float16)
meta.to_parquet("data/meta_T.parquet")
np.save("data/acts_T.npy", acts)
cfg = {"layers": pf.LAYERS, "positions": pf.POSITIONS, "model": "t"}
cache.write_dataset_manifest("data/meta_T.parquet", "data/acts_T.npy", cfg,
                             cache.sha_obj(cfg), meta.trace_id.tolist(), acts.shape)


def run(argv):
    old = sys.argv
    sys.argv = ["probe_fixed.py"] + argv
    try:
        pf.main()
        return True
    except SystemExit as e:
        print("   SystemExit:", e)
        return False
    finally:
        sys.argv = old


base = ["--tag", "T", "--ablation", "--seeds", "1", "--null-reps", "0", "--svd-hidden", "5"]
print("\n1. subset of positions")
check("runs with --positions 128", run(base + ["--positions", "128"]))
out = "data/ablation_T_L16_svd5_p128.json"
check(f"wrote {out}", os.path.exists(out), sorted(os.listdir("data")))
if os.path.exists(out):
    j = pd.read_json(out)
    check("scored exactly one position", list(j.pos) == [128], list(j.pos))

print("\n2. two positions")
check("runs with --positions 128,512", run(base + ["--positions", "128,512"]))
o2 = "data/ablation_T_L16_svd5_p128-512.json"
check("separate file, both positions", os.path.exists(o2) and list(pd.read_json(o2).pos) == [128, 512])

print("\n3. unextracted position refused")
check("bad position rejected", not run(base + ["--positions", "999"]))

print("\n4. POSITIONS not mutated")
check("module POSITIONS intact", pf.POSITIONS == [128, 256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768], pf.POSITIONS)
check("full run still works after subset runs", run(base))
check("full run gets its own file", os.path.exists("data/ablation_T_L16_svd5.json"))

print("\n5. layers do not overwrite each other")
check("layer 12 runs", run(["--tag", "T", "--ablation", "--seeds", "1", "--null-reps", "0",
                            "--svd-hidden", "5", "--layer", "12", "--positions", "128"]))
check("layer 12 has its own file", os.path.exists("data/ablation_T_L12_svd5_p128.json"))
check("layer 16 file still there", os.path.exists("data/ablation_T_L16_svd5_p128.json"))

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
