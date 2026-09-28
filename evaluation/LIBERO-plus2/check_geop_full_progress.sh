#!/usr/bin/env bash
# Periodic progress check + partial aggregate for GeoP LIBERO-plus full run.
set -euo pipefail
EVAL_DIR="${1:-$(cat /tmp/geop_full_eval_dir.txt 2>/dev/null || true)}"
if [[ -z "${EVAL_DIR}" || ! -d "${EVAL_DIR}" ]]; then
  echo "Usage: $0 <EVAL_LOG_DIR>" >&2
  exit 2
fi
CLIENT_PYTHON="${CLIENT_PYTHON:-/opt/conda/envs/libero-plus/bin/python}"
PROJ=/home/a26160/SRC/InternVLA-A-series
LIBERO_HOME="${LIBERO_HOME:-/home/a26160/SRC/LIBERO-plus}"

echo "=== $(date) ==="
echo "EVAL_DIR=$EVAL_DIR"
echo "Workers:"
grep -E 'DONE:|FAILED:|Starting shard|HEALTHCHECK|FATAL' "$EVAL_DIR"/worker_gpu*/worker.log 2>/dev/null | tail -40 || true
echo "Shard JSON count:"
find "$EVAL_DIR/logs" -name '*.json' ! -name '*_failures.json' 2>/dev/null | wc -l
echo "Crashes (sample):"
grep -h 'CRASHED:' "$EVAL_DIR"/worker_gpu*/client_*.log 2>/dev/null | head -5 || echo "(none)"
if [[ -d "$EVAL_DIR/logs" ]]; then
  PYTHONPATH="${LIBERO_HOME}:${PROJ}:${PYTHONPATH:-}" \
    "${CLIENT_PYTHON}" "${PROJ}/evaluation/LIBERO-plus2/aggregate_results.py" --root "${EVAL_DIR}" || true
  if [[ -f "$EVAL_DIR/overall_results.json" ]]; then
    python3 - <<PY
import json
d=json.load(open("${EVAL_DIR}/overall_results.json"))
print("leaderboard:", d.get("leaderboard_summary_percent"))
tot=d.get("overall",{}).get("total",{})
print("overall total:", tot)
PY
  fi
fi
# still running?
if pgrep -f 'run_eval_libero_plus_venv.sh' >/dev/null; then
  echo "STATUS: RUNNING"
else
  echo "STATUS: STOPPED"
fi
