# Development runs for the confirmation spec (Colab, CPU runtime)

These need the real dev data, so they run in E0-Pass1, not on the Mac. A CPU runtime is enough:
nothing here loads the model. Push the branch first so `checkout(BRANCH)` picks up `src/confirm`.
Cell 0 is self-contained, so no other setup cell is needed in a fresh runtime. Outputs go to
`MyDrive/e0_confirm`, and nothing is uploaded to HF.

```python
# confirm 0 (CPU runtime, once per runtime): GitHub PAT, repo on BRANCH, Drive, HF token, traces_B
import getpass, hashlib, os, shutil, subprocess
from google.colab import drive
BRANCH = 'siddhant/recovery-modal'
REPO_DIR, REPO_URL = '/content/Ghait---MJJS', 'https://github.com/marksh3n/Ghait---MJJS.git'
tok = globals().get('tok') or getpass.getpass('GitHub PAT: ').strip()
if not os.path.isdir(REPO_DIR):
    r = subprocess.run(['git', 'clone', '-q', REPO_URL.replace('https://', f'https://{tok}@'), REPO_DIR],
                       capture_output=True, text=True)
    print('clone', r.returncode, r.stderr[-300:].replace(tok, '***'))
    subprocess.run(['git', '-C', REPO_DIR, 'remote', 'set-url', 'origin', REPO_URL])  # token stays out of .git/config

def checkout(branch):
    auth = REPO_URL.replace('https://', f'https://{tok}@')
    r = subprocess.run(f'git -C {REPO_DIR} fetch -q {auth} +refs/heads/{branch}:refs/remotes/origin/{branch} && '
                       f'git -C {REPO_DIR} checkout -q -f -B {branch} origin/{branch} && '
                       f'git -C {REPO_DIR} log -1 --format="%h %ad %s"', shell=True, capture_output=True, text=True)
    print(branch, '|', (r.stdout + r.stderr).strip().replace(tok, '***'))

checkout(BRANCH)
%cd /content/Ghait---MJJS
!python -c "import math_verify" 2>/dev/null || python -m pip install -q math-verify
drive.mount('/content/drive')
from huggingface_hub import HfApi, hf_hub_download, login
try:
    HfApi().whoami()
except Exception:
    login(token=getpass.getpass('HF token: '), add_to_git_credential=False)
api = HfApi()
DST = f"{api.whoami()['name']}/e0-shared"  # your copy, which holds the p32k arrays
os.makedirs('data', exist_ok=True)
if not os.path.exists('data/traces_B.jsonl'):
    shutil.copy(hf_hub_download('MarkSh3n/e0-shared', 'traces_B.jsonl', repo_type='dataset'), 'data/traces_B.jsonl')
print('traces_B sha256', hashlib.sha256(open('data/traces_B.jsonl', 'rb').read()).hexdigest()[:16], '(want c8cb3d8f9b21453b)')
!test -f src/confirm/spec.py && echo "src/confirm present" || echo "src/confirm MISSING: push the branch, then rerun this cell"
```

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
# confirm C2 (CPU, ~5 min): live cuts and mixed problems kept by each pre-answer rule. COUNTS ONLY:
# labels and prefixes, no monitor scores, so no S7 AUROC or gain is computed
!python -u src/confirm/count_pre_answer.py --data data --rollouts {ROLL} 2>&1 | tee {CONF}/pre_answer_counts.log
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

```python
# confirm F (CPU, ~1 min): effect and discordance for Jaden's paired design, and power of his tests
!python -u src/confirm/discordance.py --bundle {CONF}/dev_bundle --traces data/traces_B.jsonl \
    --out {CONF}/discordance.json 2>&1 | tee {CONF}/discordance.log
```

The output of B–F is also logged in `MyDrive/e0_confirm`. The
`manifest.json` sha256 goes into `docs/CONFIRMATION_SPEC.md` before anything is pushed for
pre-registration.
