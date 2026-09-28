#!/usr/bin/env bash
set -euo pipefail
cd /home/a26160/SRC/InternVLA-A-series

export CKPT_PATH=/home/a26160/SRC/InternVLA-A-series/outputs/internvla_a1_5/2026_09_19_03_16_25-internvla_a1_5-libero-abs-geop_warmup-libero-geop-sft/checkpoints/025000/pretrained_model
export LIBERO_HOME=/home/a26160/SRC/LIBERO-plus
export SERVER_VENV=/home/a26160/ENV/internvla_a1_5
export CLIENT_VENV=/opt/conda/envs/libero-plus
export VLM_MODEL_PATH=/home/a26160/CKPT/Qwen3.5-2B
export STATS_KEY_MODE=suite
export ROBOT_TYPE_MODE=suite
export INFERENCE_BACKEND=standard
export ACTION_LOSS_ONLY_FLAG=--action_loss_only
export ROTATE_IMAGES=true
export RENDER_BACKEND=egl
export GPU_IDS=0
export SUITES=libero_goal
export MAX_TASKS_PER_SUITE=2
export SHARDS_PER_SUITE=2
export NUM_TRIALS_PER_TASK=1
export REPLAN_STEPS=8
export SAVE_FAILURE_VIDEOS=false
export SAVE_ACTIONS_FLAG=--no-save_actions
export SERVER_STARTUP_WAIT=180
export BASE_PORT=5784
export EVAL_LOG_DIR="${CKPT_PATH}/../libero_plus_smoke_$(date +%Y%m%d%H%M)"

echo "$EVAL_LOG_DIR" > /tmp/geop_smoke_eval_dir.txt
echo "EVAL_LOG_DIR=$EVAL_LOG_DIR"
exec bash evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh
