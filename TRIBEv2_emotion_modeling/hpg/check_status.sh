#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/blue/mzding/yujunchen/projects/TRIBEv2_emotion_modeling}"

cd "$PROJECT_DIR"
echo "Queue:"
squeue -u yujunchen
echo
echo "Latest run:"
cat latest_run.txt 2>/dev/null || true
echo
if [[ -f latest_run.txt ]]; then
  RUN_DIR="$(cat latest_run.txt)"
  find "$RUN_DIR" -maxdepth 3 -type f | sort | sed -n '1,80p'
fi
