# Development runs for the confirmation spec (Colab, CPU runtime)

These need the real dev data, so they run in E0-Pass1, not on the Mac. A CPU runtime is enough:
nothing here loads the model. Push the branch first so `checkout(BRANCH)` picks up `src/confirm`.
Then run setup cells 1–2 as usual (`BRANCH = 'siddhant/recovery-modal'`). Outputs go to
`MyDrive/e0_confirm`, and nothing is uploaded to HF.

```python
# confirm A (CPU, ~3 min): branch, p32k dev arrays, ceiling rollouts, Jaden's two 6144 exports; unit tests
checkout(BRANCH)
%cd /content/Ghait---MJJS
import os, shutil, subprocess
from huggingface_hub import hf_hub_download
for f in ['acts_B.npy', 'meta_B.parquet', 'meta_B.manifest.json']:
    shutil.copy(hf_hub_download(DST, f'p32k/{f}', repo_type='dataset'), f'data/{f}')
ROLL = hf_hub_download('MarkSh3n/e0-shared', 'recovery_ceiling', repo_type='dataset')
CONF = '/content/drive/MyDrive/e0_confirm'
os.makedirs(CONF, exist_ok=True)
auth = REPO_URL.replace('https://', f'https://{tok}@')
subprocess.run(f'git -C {REPO_DIR} fetch -q {auth} +refs/heads/jaden/confirmation-planning:refs/remotes/origin/jaden/confirmation-planning',
               shell=True, capture_output=True)
for n in ['pop342', 'cohort227']:
    subprocess.run(f'git -C {REPO_DIR} show origin/jaden/confirmation-planning:experiments/jaden_confirmation/data/'
                   f'development/e0_6144_preds_{n}.csv > /content/{n}.csv', shell=True)
!wc -l /content/pop342.csv /content/cohort227.csv
!python tests/test_confirm.py | tail -1; python tests/test_confirm_pipeline.py | tail -1; python tests/test_oracle.py | tail -1
```

```python
# confirm B (CPU, ~20-30 min): where the 6144 tree numbers come from (task 1)
!python -u src/confirm/provenance_6144.py --data data --out {CONF}/provenance_6144 \
    --jaden /content/pop342.csv /content/cohort227.csv 2>&1 | tee {CONF}/provenance_6144.log
```

```python
# confirm C (CPU, ~15-30 min): fit the frozen monitors on all dev data, thresholds on dev only
!python -u src/confirm/fit_monitors.py --data data --out {CONF}/monitors 2>&1 | tee {CONF}/fit.log
!sha256sum {CONF}/monitors/manifest.json {CONF}/monitors/*.joblib
```

```python
# confirm D (CPU, ~10 min): the confirmation analysis, unchanged, on the dev data (dry run)
!python -u src/confirm/dev_bundle.py --data data --rollouts {ROLL} --monitors {CONF}/monitors \
    --out {CONF}/dev_bundle 2>&1 | tail -25
!python -u src/confirm/analyze.py --bundle {CONF}/dev_bundle --out {CONF}/dev_dryrun 2>&1 | tee {CONF}/dryrun.log | tail -150
```

```python
# confirm E (CPU, ~10 min): the dev oracle notes and headline figure again, with the corrected wording
# and the token CIs / selection cost the old summaries lacked (task 5)
!python src/probe_cmp/oracle.py --rollouts {ROLL} --traces data/traces_B.jsonl \
    --preds /content/drive/MyDrive/e0_probe_cmp/preds --problems all main pilot --out {CONF}/oracle_v4 2>&1 | tail -60
```

Send Siddhant the printed output of B–E: the logs are also in `MyDrive/e0_confirm`. The
`manifest.json` sha256 goes into `docs/CONFIRMATION_SPEC.md` before anything is pushed for
pre-registration.
