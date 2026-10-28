# r(t) ceiling run on Modal

The ceiling design: every trace in the probe_cmp population (untruncated traces that reach
4096 tokens, 87 mixed problems, correct and wrong traces). Each trace is cut at exactly
512, 2048 and 4096 tokens, and each problem gets one t0 restart. K=8 samples per cut at
T=0.7, top_p=0.95, max_tokens 32768. Prefixes run in priority order. Tier 1 is a
pass-rate-stratified set of problems worth about 12.5M tokens. Tier 2 is the other
problems. A problem is either complete or not started.

Code: `src/recovery/build_prefixes.py --design ceiling`, `src/recovery/run_rollouts.py`,
`src/recovery/check_temp.py`, `scripts/modal_app.py`. Everything lives on the Volume
`e0-recovery`: `/inputs`, `/outputs`, and `/hf` for the model cache.

## Before you start

- Check out branch `siddhant/recovery-modal`. `data/` is not in git; the traces come from
  the HF dataset `MarkSh3n/e0-shared` (see "Once"). The file must have 2202 lines, and
  its SHA-256 must start with `c8cb3d8f9b21453b`, the copy the prefixes were designed
  on. If the hash differs, stop.
- Run the steps in order. **Stop after step 2 if the temp check doesn't print
  `VERDICT: T=0.7 is clearly best`.** **Stop after step 4** and post the last ~30 log
  lines of each benchmark (tok/s, tier projections). Step 5 only after sign-off.
- If a step fails, send the full error; don't work around it. If the T4 smoke test fails
  inside vLLM, add `--gpu h100` to rerun it; that costs about $0.50.
- `--budget` for step 5 is the Modal credit left at that point (Settings → Usage).

## Once: Windows (PowerShell)

Run each line on its own, in order. Skip the Mac/Linux block below.

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # only if uv is missing; then open a new terminal
uv tool install modal
modal setup
cd $HOME
git clone https://github.com/marksh3n/Ghait---MJJS.git    # skip if cloned; then git fetch
cd Ghait---MJJS
git switch siddhant/recovery-modal
uvx --from huggingface_hub hf download MarkSh3n/e0-shared traces_B.jsonl --repo-type dataset --local-dir data/hf
(Get-FileHash data\hf\traces_B.jsonl -Algorithm SHA256).Hash         # must start with C8CB3D8F9B21453B
(Get-Content data\hf\traces_B.jsonl | Measure-Object -Line).Lines    # must be 2202
modal volume create e0-recovery                                       # "already exists" is fine
modal volume put --force e0-recovery data/hf/traces_B.jsonl /inputs/traces_B.jsonl
modal volume ls e0-recovery /inputs
```

- If `modal` is not found after `uv tool install modal`, run `uv tool update-shell` and
  open a new terminal.
- If Modal's progress bars cause charmap/encoding errors, run
  `$env:PYTHONIOENCODING = "utf-8"` first.
- Then go to "Steps". Run every step from the `Ghait---MJJS` folder. Each step is a single
  line and works in PowerShell as written.

## Once: Mac / Linux

```bash
brew install uv            # if uv is missing (or: curl -LsSf https://astral.sh/uv/install.sh | sh)
uv tool install modal
modal setup
uvx --from huggingface_hub hf download MarkSh3n/e0-shared traces_B.jsonl \
    --repo-type dataset --local-dir data/hf
shasum -a 256 data/hf/traces_B.jsonl                     # starts with c8cb3d8f9b21453b?
scripts/modal_upload.sh                                  # checks 2202 lines, -> /inputs/traces_B.jsonl
```

## Steps

```bash
# 1. prefixes + tiers (CPU container, ~pennies); writes /inputs/prefixes_ceiling.jsonl
modal run scripts/modal_app.py --step build            # add --extra "--force" to rebuild
modal volume get e0-recovery /inputs/prefixes_ceiling.jsonl data/recovery_v2/

# 2. temperature gate (T4): exits non-zero unless T=0.7 is clearly best
modal run scripts/modal_app.py --step temp

# 3. smoke (T4): 2 prefixes, n=2, 256-token cap -> /outputs/smoke.jsonl; run twice, the
#    second run must find nothing to do
modal run scripts/modal_app.py --step smoke

# 4. benchmark: 5 minutes, then projections for tier 1 and tier 1+2
modal run scripts/modal_app.py --step bench --gpu h100
modal run scripts/modal_app.py --step bench --gpu h200

# 5. full run, detached; --budget is the credit left in $
modal run --detach scripts/modal_app.py --step run --gpu h100 --budget 25
```

The benchmark writes to the same `/outputs/rollouts.jsonl` as the full run with the same
config, so the full run skips its rows. `--minutes N` (default 5 for `bench`) stops after N
minutes and drops prefixes still running; they rerun later, so a short benchmark does not
pay for a long tail.

## Watching and stopping

```bash
modal app logs e0-recovery                  # live log: tok/s, running/waiting, KV %, $ spent
modal run scripts/modal_app.py --step status  # done/left per tier + spend, CPU only
modal app stop e0-recovery                  # stop; everything written so far is kept
modal volume get e0-recovery /outputs ./outputs
```

On the dashboard (modal.com/apps), open app `e0-recovery`. The running function
(`on_h100` or `on_h200`) shows the same log lines, plus GPU utilisation under
*Containers*.

## Budget guard

- The run stops starting new problems once `spent + estimated cost of the running work
  and the next problem >= budget - $2`. Work already running finishes and is written.
- If `spent >= budget - $0.50`, the run stops at once and drops whatever is still running.
- Spend is charged at the all-in rate `--rate-usd-hr`: H100 4.40, H200 4.99, T4 1.05.
  Each attempt also adds 4 minutes of startup.
- If Modal preempts the container, it restarts the call with the same arguments. The
  restart resumes from the file and inherits the spend already recorded in
  `rollouts.jsonl.ledger.json`. The last few minutes before the preemption are not recorded
  (the ledger is saved every 60 s, the Volume committed every 2 min).
- A new `modal run` is a new ledger key, so `--budget` there means the credit left at that
  point.
