#!/usr/bin/env bash
# Upload the ceiling-run inputs to the e0-recovery Volume.
#   TRACES=/path/to/traces_B.jsonl scripts/modal_upload.sh   (default data/hf/traces_B.jsonl)
#   scripts/modal_upload.sh path/to/prefixes_ceiling.jsonl  also uploads a local prefix file;
#                                                            normally built by --step build
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
