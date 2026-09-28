#!/usr/bin/env bash
set -euo pipefail
# Full LIBERO-plus eval for libero_plus_geop SFT checkpoints.
# Unlike libero warmup (per-suite stats keys), mix SFT stats.json has a single key: libero_plus.
cd /home/a26160/SRC/InternVLA-A-series

export CKPT_PATH="${CKPT_PATH:-/home/a26160/SRC/InternVLA-A-series/outputs/internvla_a1_5/2026_09_25_08_41_19-internvla_a1_5-libero-plus-abs-geop_libero50k-libero-plus-geop-sft/checkpoints/050000/pretrained_model}"
export LIBERO_HOME="${LIBERO_HOME:-/home/a26160/SRC/LIBERO-plus}"
export SERVER_VENV="${SERVER_VENV:-/opt/conda/envs/internvla_a1_5}"
export CLIENT_VENV="${CLIENT_VENV:-/opt/conda/envs/libero-plus}"
# Processor loads Qwen/Qwen3.5-2B from HF cache / hub (matches train_config).
unset VLM_MODEL_PATH || true
export STATS_KEY_MODE=libero_plus
export ROBOT_TYPE_MODE=libero_plus
export INFERENCE_BACKEND=standard
export ACTION_LOSS_ONLY_FLAG=--action_loss_only
export ROTATE_IMAGES=true
export RENDER_BACKEND=osmesa
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
unset MAX_TASKS_PER_SUITE || true
export SHARDS_PER_SUITE="${SHARDS_PER_SUITE:-8}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
export REPLAN_STEPS="${REPLAN_STEPS:-8}"
export SAVE_FAILURE_VIDEOS=false
export SAVE_ACTIONS_FLAG=--no-save_actions
export SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-120}"
export BASE_PORT="${BASE_PORT:-5800}"
export EVAL_LOG_DIR="${EVAL_LOG_DIR:-${CKPT_PATH}/../libero_plus_full_$(date +%Y%m%d%H%M)}"

# Hygiene also applied inside run_eval_libero_plus_venv.sh; set here so parent shell is clean.
unset PYTHONHOME
export PYTHONNOUSERSITE=1

echo "$EVAL_LOG_DIR" > /tmp/geop_libero_plus_sft_eval_dir.txt
echo "EVAL_LOG_DIR=$EVAL_LOG_DIR"
echo "CKPT_PATH=$CKPT_PATH"
exec bash evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh
