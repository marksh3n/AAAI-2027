# exercises every stale-reuse path the mentor named, using fake part files.
# stdlib only, so it runs without torch/numpy.
import os, shutil, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cache import (StaleCache, check_dataset_manifest, dataset_manifest_path,
                   load_manifest, part_ok, save_manifest, sha_file, sha_obj,
                   write_dataset_manifest)

CFG_A = {"model": "deepseek-1.5B", "layers": [0, 16], "positions": [128, 256], "v": 2}
CFG_B = dict(CFG_A, model="deepseek-7B")
HASH_A, HASH_B = sha_obj(CFG_A), sha_obj(CFG_B)

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  pass  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")


def expect(trace_id, text, n_tok, idx):
    return {"trace_id": trace_id, "trace_sha": sha_obj({"text": text}),
            "n_trace_tokens": n_tok, "token_idx": list(idx)}


def write_part(d, n, payload=b"activations"):
    p = f"{d}/{n}.npy"
    open(p, "wb").write(payload)
    return p


def entry_for(d, n, e):
    e = dict(e)
    e["shape"] = [len(e["token_idx"]), 2, 1536]
    e["sha256"] = sha_file(f"{d}/{n}.npy")
    return e


def fresh_dir():
    d = tempfile.mkdtemp()
    shutil.rmtree(d)
    return d


print("\n1. fresh directory, extract, then reuse")
d = fresh_dir()
m = load_manifest(d, CFG_A, HASH_A)
check("empty manifest on a new dir", m["parts"] == {} and m["config_hash"] == HASH_A)
e0 = expect("12_0", "hello", 900, [127, 255])
write_part(d, 0)
m["parts"]["0"] = entry_for(d, 0, e0)
save_manifest(d, m)
m2 = load_manifest(d, CFG_A, HASH_A)
check("reuses a part it wrote", part_ok(m2["parts"].get("0"), f"{d}/0.npy", e0) is None,
      part_ok(m2["parts"].get("0"), f"{d}/0.npy", e0))

print("\n2. config change (model swapped) is refused")
try:
    load_manifest(d, CFG_B, HASH_B)
    check("raises on config change", False)
except StaleCache as ex:
    check("raises on config change", "model" in str(ex), str(ex))
    check("names the changed field", "deepseek-7B" in str(ex))

print("\n3. traces.jsonl reordered: line 0 now holds a different trace")
other = expect("47_1", "a different trace entirely", 900, [127, 255])
why = part_ok(m2["parts"].get("0"), f"{d}/0.npy", other)
check("refuses a part keyed to another trace", why is not None and "trace_id" in why, why)

print("\n4. traces regenerated: same trace_id, resampled text")
same_id = expect("12_0", "resampled at temperature 0.6", 900, [127, 255])
why = part_ok(m2["parts"].get("0"), f"{d}/0.npy", same_id)
check("refuses same id with different text", why is not None and "trace_sha" in why, why)

print("\n5. same trace_id and same shape, but a different token count")
shorter = expect("12_0", "hello", 200, [127])
why = part_ok(m2["parts"].get("0"), f"{d}/0.npy", shorter)
check("refuses a changed token count", why is not None, why)

print("\n6. positions changed: same trace, different extraction offsets")
moved = expect("12_0", "hello", 900, [127, 511])
why = part_ok(m2["parts"].get("0"), f"{d}/0.npy", moved)
check("refuses changed token_idx", why is not None and "token_idx" in why, why)

print("\n7. the .npy itself was overwritten out of band")
write_part(d, 0, b"different bytes entirely")
why = part_ok(m2["parts"].get("0"), f"{d}/0.npy", e0)
check("catches a rewritten file", why is not None and "contents" in why, why)
check("skips the hash when asked", part_ok(m2["parts"].get("0"), f"{d}/0.npy", e0,
                                           check_bytes=False) is None)

print("\n8. a missing part file")
os.remove(f"{d}/0.npy")
check("catches a deleted part", part_ok(m2["parts"].get("0"), f"{d}/0.npy", e0) == "file is missing")

print("\n9. a part with no manifest entry")
check("catches an unrecorded part", part_ok(None, f"{d}/0.npy", e0) == "not in the manifest")

print("\n10. legacy dir: parts on disk, no manifest (run B)")
d2 = fresh_dir()
os.makedirs(d2)
for n in range(3):
    write_part(d2, n)
try:
    load_manifest(d2, CFG_A, HASH_A)
    check("refuses an unmanifested cache", False)
except StaleCache as ex:
    check("refuses an unmanifested cache", "--verify" in str(ex) and "--rebuild" in str(ex), str(ex))

print("\n11. --rebuild sets the old parts aside instead of deleting them")
m3 = load_manifest(d2, CFG_A, HASH_A, rebuild=True)
check("returns an empty manifest", m3["parts"] == {})
check("old dir is empty of parts", not os.path.exists(f"{d2}/0.npy"))
aside = [p for p in os.listdir(os.path.dirname(d2))
         if p.startswith(os.path.basename(d2) + ".stale-")]
check("old parts were preserved", len(aside) == 1 and
      os.path.exists(f"{os.path.dirname(d2)}/{aside[0]}/0.npy"), aside)

print("\n12. dataset manifest binds meta.parquet to acts.npy")
d3 = fresh_dir()
os.makedirs(d3)
meta_p, acts_p = f"{d3}/meta_B.parquet", f"{d3}/acts_B.npy"
open(meta_p, "wb").write(b"parquet")
open(acts_p, "wb").write(b"acts")
tids = ["12_0", "12_0", "47_1", "47_1"]
write_dataset_manifest(meta_p, acts_p, CFG_A, HASH_A, tids, (4, 2, 1536))
check("accepts the pair it wrote",
      check_dataset_manifest(meta_p, acts_p, tids, (4, 2, 1536), strict=True)["config_hash"] == HASH_A)

for name, args, needle in [
    ("row count differs", (tids[:3], (4, 2, 1536)), "rows"),
    ("acts shape differs", (tids, (6, 2, 1536)), "acts is"),
    ("row order differs", (["47_1", "47_1", "12_0", "12_0"], (4, 2, 1536)), "order"),
]:
    try:
        check_dataset_manifest(meta_p, acts_p, *args)
        check(name, False)
    except StaleCache as ex:
        check(name, needle in str(ex), str(ex))

open(acts_p, "wb").write(b"acts rewritten")
try:
    check_dataset_manifest(meta_p, acts_p, tids, (4, 2, 1536), strict=True)
    check("strict catches a rewritten acts.npy", False)
except StaleCache as ex:
    check("strict catches a rewritten acts.npy", "contents changed" in str(ex))
check("non-strict skips the acts hash",
      check_dataset_manifest(meta_p, acts_p, tids, (4, 2, 1536)) is not None)

print("\n13. a missing dataset manifest is refused, not assumed fine")
os.remove(dataset_manifest_path(meta_p))
try:
    check_dataset_manifest(meta_p, acts_p, tids, (4, 2, 1536))
    check("refuses an unmanifested dataset", False)
except StaleCache as ex:
    check("refuses an unmanifested dataset", "extract.py --fixed" in str(ex))

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
