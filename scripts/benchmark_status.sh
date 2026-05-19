#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT="${ASSET_ROOT:-/data1/kechuang/EyeTimelineAssets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ASSET_ROOT/benchmark_outputs}"
LATEST_RUN_FILE="$OUTPUT_ROOT/latest_run.txt"

if [[ -f "$LATEST_RUN_FILE" ]]; then
  RUN_DIR="$(cat "$LATEST_RUN_FILE")"
else
  RUN_DIR="$(find "$OUTPUT_ROOT/runs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
fi

echo "OUTPUT_ROOT=$OUTPUT_ROOT"
if [[ -z "${RUN_DIR:-}" || ! -d "$RUN_DIR" ]]; then
  echo "No benchmark run directory found."
  exit 0
fi

echo "RUN_DIR=$RUN_DIR"
if [[ -f "$RUN_DIR/status.json" ]]; then
  python - "$RUN_DIR/status.json" <<'PY'
import json, sys
p = sys.argv[1]
s = json.load(open(p))
print(f"updated_at={s.get('updated_at')}")
print(f"total={s.get('total')} completed={s.get('completed')} failed={s.get('failed')} skipped={s.get('skipped')} remaining={s.get('remaining')}")
running = s.get("running") or {}
if running:
    print("running:")
    for k, v in running.items():
        print(f"  {k} gpu={v}")
PY
fi

if [[ -f "$OUTPUT_ROOT/run_logs/latest.pid" ]]; then
  PID="$(cat "$OUTPUT_ROOT/run_logs/latest.pid")"
  if ps -p "$PID" >/dev/null 2>&1; then
    echo "process=running pid=$PID"
  else
    echo "process=not-running last_pid=$PID"
  fi
fi

if [[ -f "$OUTPUT_ROOT/summary.md" ]]; then
  echo
  echo "== summary =="
  sed -n '1,80p' "$OUTPUT_ROOT/summary.md"
fi

if [[ -f "$OUTPUT_ROOT/run_logs/latest.log" ]]; then
  LOG="$(cat "$OUTPUT_ROOT/run_logs/latest.log")"
  if [[ -f "$LOG" ]]; then
    echo
    echo "== latest log tail =="
    tail -n "${TAIL_LINES:-80}" "$LOG"
  fi
fi
