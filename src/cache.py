# manifest-validated cache for activation parts and the acts/meta pair.
# stdlib only, probe scripts import this.
import glob, hashlib, json, os, shutil, time


class StaleCache(RuntimeError):
    """Cache cannot be proven to match the current run."""


def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def short(h):
    return h[:12]


def changed_keys(old, new):
    return [k for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]


def _diff_lines(old, new):
    out = []
    for k in changed_keys(old, new):
        a, b = repr(old.get(k)), repr(new.get(k))
        out.append(f"    {k}: cached {a[:90]} -> current {b[:90]}")
    return "\n".join(out) or "    (no field-level diff; the hash covers the whole config)"


def set_aside(path):
    # move aside, never delete
    dest = f"{path}.stale-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.move(path, dest)
    print(f"moved the previous cache to {dest}")
    return dest


# ---------------------------------------------------------------- parts manifest


def manifest_path(parts_dir):
    return f"{parts_dir}/manifest.json"


def load_manifest(parts_dir, cfg, cfg_hash, rebuild=False):
    """Manifest whose parts are safe to reuse, else StaleCache."""
    path = manifest_path(parts_dir)
    fresh = {"config_hash": cfg_hash, "config": cfg, "parts": {}}
    stray = glob.glob(f"{parts_dir}/*.npy")

    if not os.path.exists(path):
        if not stray:
            os.makedirs(parts_dir, exist_ok=True)
            return fresh
        if rebuild:
            set_aside(parts_dir)
            os.makedirs(parts_dir, exist_ok=True)
            return fresh
        raise StaleCache(
            f"{len(stray)} parts in {parts_dir}, no manifest.json. Provenance unknown.\n"
            "  --verify --adopt   check a sample against a fresh pass, then stamp\n"
            "  --rebuild          set aside and re-extract"
        )

    m = json.load(open(path))
    if m.get("config_hash") != cfg_hash:
        if rebuild:
            set_aside(parts_dir)
            os.makedirs(parts_dir, exist_ok=True)
            return fresh
        raise StaleCache(
            f"{parts_dir} config {short(m.get('config_hash', 'unknown'))} "
            f"!= current {short(cfg_hash)}:\n"
            f"{_diff_lines(m.get('config', {}), cfg)}\n"
            "Use --rebuild to set aside and re-extract."
        )
    m["config"] = cfg  # same hash
    m.setdefault("parts", {})
    return m


def save_manifest(parts_dir, manifest):
    path = manifest_path(parts_dir)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def part_ok(entry, path, expect, check_bytes=True):
    """None if the part matches expect, else the reason."""
    if entry is None:
        return "not in the manifest"
    if not os.path.exists(path):
        return "file is missing"
    for k, v in expect.items():
        if entry.get(k) != v:
            return f"{k} changed: cached {entry.get(k)!r} -> current {v!r}"
    if check_bytes and entry.get("sha256") != sha_file(path):
        return "file contents changed since the manifest was written"
    return None


# ------------------------------------------------------------- dataset manifest


def dataset_manifest_path(meta_path):
    return meta_path.replace(".parquet", "") + ".manifest.json"


def write_dataset_manifest(meta_path, acts_path, cfg, cfg_hash, trace_ids, acts_shape, strict=True):
    """Bind meta.parquet to acts.npy and both to the config."""
    m = {
        "config_hash": cfg_hash,
        "config": cfg,
        "n_rows": len(trace_ids),
        "acts_shape": list(acts_shape),
        "acts_path": os.path.basename(acts_path),
        "row_order_hash": sha_obj(list(trace_ids)),
        "acts_sha256": sha_file(acts_path) if strict else None,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = dataset_manifest_path(meta_path)
    with open(path, "w") as f:
        json.dump(m, f, indent=2, sort_keys=True)
    print(f"wrote {path} (config {short(cfg_hash)})")
    return m


def check_dataset_manifest(meta_path, acts_path, trace_ids, acts_shape, strict=False):
    """Refuse a meta/acts pair from different extractions."""
    path = dataset_manifest_path(meta_path)
    if not os.path.exists(path):
        raise StaleCache(
            f"{path} missing: cannot tie {os.path.basename(meta_path)} to "
            f"{os.path.basename(acts_path)}.\n"
            "Re-run extract.py --fixed; validated parts are reused."
        )
    m = json.load(open(path))
    problems = []
    if m["n_rows"] != len(trace_ids):
        problems.append(f"meta has {len(trace_ids)} rows, manifest recorded {m['n_rows']}")
    if list(m["acts_shape"]) != list(acts_shape):
        problems.append(f"acts is {list(acts_shape)}, manifest recorded {m['acts_shape']}")
    if m["row_order_hash"] != sha_obj(list(trace_ids)):
        problems.append("meta's trace_id order differs from the one acts.npy was concatenated in")
    if strict and m.get("acts_sha256") and m["acts_sha256"] != sha_file(acts_path):
        problems.append("acts.npy contents changed since the manifest was written")
    if problems:
        raise StaleCache(
            "acts and meta are from different runs:\n  "
            + "\n  ".join(problems)
            + "\nRe-run extract.py --fixed."
        )
    return m
