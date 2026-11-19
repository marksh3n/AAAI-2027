# put every registered file into data/ from the HF dataset named by E0_HF_REPO and, with --iter1, the
# between-problem run's meta and acts. registry-driven, so a new source is fetched as soon as its
# entry and its files exist.
#   python src/probe_cmp/fetch.py            # traces + every registered acts/meta pair
#   %run src/probe_cmp/fetch.py --iter1      # also between-problem run meta.parquet and acts.npy (1.6 GB);
#                                            # %run, not !python: Colab auth needs the kernel
import argparse, os, sys

# under %run the kernel's cwd (repo root, which has a data/ dir) shadows this folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402

REPO = os.environ.get("E0_HF_REPO", "")
# between-problem run: 367 problems x 1 trace, steps-mode acts (Google Drive file ids)
ITER1 = {"meta.parquet": "1Wku64a4n3p7DUd6O-IYd62oTvCA8LKOW", "acts.npy": "1uwAR6VlI1mF_3tf-nYHsIIeKHxw72YNE"}


def hf(data_dir, names):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
    for f in names:
        dst = os.path.join(data_dir, f)
        if os.path.lexists(dst) and not os.path.exists(dst):
            os.remove(dst)  # a link into an HF cache that a runtime restart wiped
        if os.path.exists(dst):
            continue
        try:
            # a real file in data/, not a link into the cache
            hf_hub_download(REPO, f, repo_type="dataset", local_dir=data_dir)
            print(f"  {f}")
        except EntryNotFoundError as e:
            if isinstance(e, LocalEntryNotFoundError):
                raise  # offline or the Hub is down: fail loudly, never drop a source silently
            print(f"  {f}: not on {REPO} yet, skipped")


def drive(dst_dir, files):
    # by file id, so it works from whichever account the folder is shared with
    import io
    from google.colab import auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    auth.authenticate_user()
    svc = build("drive", "v3")
    os.makedirs(dst_dir, exist_ok=True)
    for name, fid in files.items():
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst):
            continue
        with io.FileIO(dst + ".part", "wb") as fh:
            dl = MediaIoBaseDownload(fh, svc.files().get_media(fileId=fid, supportsAllDrives=True),
                                     chunksize=64 << 20)
            done = False
            while not done:
                _, done = dl.next_chunk()
        os.rename(dst + ".part", dst)
        print(f"  {dst} ({os.path.getsize(dst) / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--registry", default=D.REGISTRY)
    ap.add_argument("--iter1", action="store_true")
    a = ap.parse_args()
    if not REPO:
        raise SystemExit("set E0_HF_REPO to the HF dataset holding the traces and states")
    # a data/ link into a Drive folder that no longer exists
    if os.path.islink(a.data) and not os.path.exists(a.data):
        os.remove(a.data)
    os.makedirs(a.data, exist_ok=True)
    reg = D.load_registry(a.registry)
    print(f"from {REPO}:")
    hf(a.data, [reg["traces"]] + [s[k] for s in reg["sources"] for k in ("acts", "meta")])
    if a.iter1:
        print("between-problem run files from Drive:")
        drive(os.path.join(a.data, "iter1"), ITER1)


if __name__ == "__main__":
    main()
