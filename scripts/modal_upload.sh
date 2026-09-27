#!/usr/bin/env bash
# put the inputs of the ceiling run on the e0-recovery Volume. data/ is gitignored, so
# traces_B.jsonl comes from outside the repo:
#   TRACES=/path/to/traces_B.jsonl scripts/modal_upload.sh     (default data/hf/traces_B.jsonl)
# the prefix file is normally built on Modal (--step build); a local one is uploaded only
# when passed: scripts/modal_upload.sh path/to/prefixes_ceiling.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."
VOL=e0-recovery
TRACES=${TRACES:-data/hf/traces_B.jsonl}
if [ "$(wc -l < "$TRACES")" -ne 2202 ]; then
  echo "$TRACES should have 2202 lines (367 problems x 6)" >&2
  exit 1
fi
modal volume create "$VOL" 2>/dev/null || true
modal volume put --force "$VOL" "$TRACES" /inputs/traces_B.jsonl
if [ $# -ge 1 ]; then
  modal volume put --force "$VOL" "$1" /inputs/prefixes_ceiling.jsonl
fi
modal volume ls "$VOL" /inputs
