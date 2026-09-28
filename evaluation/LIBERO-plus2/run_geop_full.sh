#!/usr/bin/env bash
set -euo pipefail
cd /home/a26160/SRC/InternVLA-A-series

export CKPT_PATH=/home/a26160/SRC/InternVLA-A-series/outputs/internvla_a1_5/2026_09_19_03_16_25-internvla_a1_5-libero-abs-geop_warmup-libero-geop-sft/checkpoints/050000/pretrained_model
export LIBERO_HOME=/home/a26160/SRC/LIBERO-plus
export SERVER_VENV=/home/a26160/ENV/internvla_a1_5
export CLIENT_VENV=/opt/conda/envs/libero-plus
# Leave unset: processor uses Qwen/Qwen3.5-2B from train_config / HF cache.
unset VLM_MODEL_PATH || true
# Warmup GeoP ckpt has per-suite stats keys (libero_spatial/...).
# For libero_plus mix SFT (single key), use run_geop_libero_plus_sft_full.sh instead.
export STATS_KEY_MODE=suite
# Must match training schemas (libero_*/end_effector), not panda/joint.
export ROBOT_TYPE_MODE=suite
export INFERENCE_BACKEND=standard
export ACTION_LOSS_ONLY_FLAG=--action_loss_only
# GeoP rlds20 videos are arm-at-top; live env is arm-at-bottom → rotate 180.
export ROTATE_IMAGES=true
# OSMesa (libosmesa6): same in-process env loop as starVLA LIBERO-plus.
# No EGL context, so eval_libero_plus.py does not fork per task.
export RENDER_BACKEND=osmesa
# Machine has 4x H200 free.
export GPU_IDS=0,1,2,3
export SUITES="libero_spatial libero_object libero_goal libero_10"
unset MAX_TASKS_PER_SUITE
export SHARDS_PER_SUITE=8
export NUM_TRIALS_PER_TASK=1
export REPLAN_STEPS=8
export SAVE_FAILURE_VIDEOS=false
export SAVE_ACTIONS_FLAG=--no-save_actions
export SERVER_STARTUP_WAIT=10
export BASE_PORT=5800
export EVAL_LOG_DIR="${CKPT_PATH}/../libero_plus_full_$(date +%Y%m%d%H%M)"

echo "$EVAL_LOG_DIR" > /tmp/geop_full_eval_dir.txt
echo "EVAL_LOG_DIR=$EVAL_LOG_DIR"
exec bash evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh
