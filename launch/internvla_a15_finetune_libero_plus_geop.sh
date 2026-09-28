#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Continue GeoP SFT on LIBERO-plus mix from a LIBERO 4-suite Phase-2 ckpt.
#
# Defaults:
#   - weights: 4DWVLA .../checkpoints/050000/pretrained_model
#   - data:    /tmp/DATA/libero_plus_geop  (HF_LEROBOT_HOME=/tmp/DATA)
#   - env:     /home/a26160/ENV/internvla_a1_5
#
# Usage:
#   bash launch/internvla_a15_finetune_libero_plus_geop.sh
#   SMOKE=1 bash launch/internvla_a15_finetune_libero_plus_geop.sh
#   WAN_SMOKE=1 bash launch/internvla_a15_finetune_libero_plus_geop.sh
#   USE_DEEPSPEED=1 bash launch/internvla_a15_finetune_libero_plus_geop.sh
#
# Env overrides:
#   WARMUP_CKPT  HF_LEROBOT_HOME  DATASET_REPO_ID  WAN_CHECKPOINT_PATH
#   PYTHON  STEPS  BATCH_SIZE  OUTPUT_DIR  OPTIMIZER_LR  USE_DEEPSPEED
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${PROJ_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

DEFAULT_CKPT_ROOT="/home/a26160/CKPT/4DWVLA/2026_09_19_03_16_25-internvla_a1_5-libero-abs-geop_warmup-libero-geop-sft"
DEFAULT_WARMUP_CKPT="${DEFAULT_CKPT_ROOT}/checkpoints/050000/pretrained_model"

export HF_HOME="${HF_HOME:-/tmp/zwy}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/tmp/DATA}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export USE_LIBUV="${USE_LIBUV:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-libnccl-tuner-disabled.so}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/itvla-triton-cache}"

VENV_PYTHON="${VENV_PYTHON:-/home/a26160/ENV/internvla_a1_5/bin/python}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${VENV_PYTHON}" ]]; then
    PYTHON="${VENV_PYTHON}"
  else
    PYTHON="python"
  fi
fi
if [[ -x "${VENV_PYTHON}" ]]; then
  export PATH="$(dirname "${VENV_PYTHON}"):${PATH}"
fi
# Drop the container/conda base interpreter override. PYTHONHOME=/opt/conda makes
# this env's python import /opt/conda torch, which does not match causal_conv1d.
unset PYTHONHOME
# The venv's site-packages lerobot does not include dataset schema YAMLs.
# Prefer this repo's src and do not inherit an external PYTHONPATH.
export PYTHONPATH="${PROJ_ROOT}/src"
export PYTHONNOUSERSITE=1

PY_SITE="$("${PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
TORCH_FILE="$("${PYTHON}" -c 'import torch; print(torch.__file__)')"
case "${TORCH_FILE}" in
  /opt/conda/*)
    echo "ERROR: torch resolved to conda base: ${TORCH_FILE}" >&2
    echo "causal_conv1d in ${VENV_PYTHON} was built against this env's torch." >&2
    echo "Unset PYTHONHOME (it is often /opt/conda) and retry." >&2
    exit 1
    ;;
esac
echo "torch: ${TORCH_FILE}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
# torchcodec needs FFmpeg shared libs (core7 -> libavcodec.so.61). The internvla
# venv does not ship them; the lerobot conda env does.
FFMPEG_LIB="${FFMPEG_LIB:-/opt/conda/envs/lerobot/lib}"
FFMPEG_LD=""
if [[ -d "${FFMPEG_LIB}" ]]; then
  FFMPEG_LD="${FFMPEG_LIB}:"
fi
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:\
${PY_SITE}/torch/lib:\
${PY_SITE}/nvidia/cuda_runtime/lib:\
${PY_SITE}/nvidia/cuda_nvrtc/lib:\
${PY_SITE}/nvidia/npp/lib:${CUDA_HOME}/lib64:${FFMPEG_LD}${LD_LIBRARY_PATH:-}"

WARMUP_CKPT="${WARMUP_CKPT:-${DEFAULT_WARMUP_CKPT}}"
if [[ ! -d "${WARMUP_CKPT}" ]]; then
  echo "ERROR: WARMUP_CKPT not found: ${WARMUP_CKPT}" >&2
  exit 1
fi

POLICY="internvla_a1_5"
PRETRAINED_DETAIL="geop_libero50k"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-Qwen/Qwen3.5-2B}"
WAN_CHECKPOINT_PATH="${WAN_CHECKPOINT_PATH:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
WAN_CONFIG_PATH="${WAN_CONFIG_PATH:-${WAN_CHECKPOINT_PATH}}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${WAN_CHECKPOINT_PATH}/Wan2.2_VAE.pth}"
VIDEO_MICRO_BATCH_SIZE="${VIDEO_MICRO_BATCH_SIZE:-4}"

DATASET_REPO_ID="${DATASET_REPO_ID:-libero_plus_geop}"
ACTION_TYPE=abs
# Do not reuse the 4-suite stats stored in the 50k ckpt.
USE_EXTERNAL_STATS="${USE_EXTERNAL_STATS:-false}"

# Lower than the original Phase-2 LR (5e-5) for continued training on Plus.
OPTIMIZER_LR="${OPTIMIZER_LR:-2e-5}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2e-6}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-36812}"

WAN_SMOKE="${WAN_SMOKE:-0}"
SMOKE="${SMOKE:-0}"

if [[ "${WAN_SMOKE}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  PROC_PER_NODE="${PROC_PER_NODE:-1}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  STEPS="${STEPS:-2}"
  NUM_WORKERS="${NUM_WORKERS:-2}"
  SAVE_FREQ="${SAVE_FREQ:-2}"
  LOG_FREQ="${LOG_FREQ:-1}"
  SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-1}"
  WANDB_ENABLE="${WANDB_ENABLE:-false}"
  JOB_SUFFIX="libero-plus-geop-wan-smoke"
elif [[ "${SMOKE}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  PROC_PER_NODE="${PROC_PER_NODE:-1}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  STEPS="${STEPS:-100}"
  NUM_WORKERS="${NUM_WORKERS:-2}"
  SAVE_FREQ="${SAVE_FREQ:-100}"
  LOG_FREQ="${LOG_FREQ:-10}"
  SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-50}"
  WANDB_ENABLE="${WANDB_ENABLE:-false}"
  JOB_SUFFIX="libero-plus-geop-sft-smoke"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  PROC_PER_NODE="${PROC_PER_NODE:-8}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  STEPS="${STEPS:-100000}"
  NUM_WORKERS="${NUM_WORKERS:-32}"
  SAVE_FREQ="${SAVE_FREQ:-5000}"
  LOG_FREQ="${LOG_FREQ:-10}"
  SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-2000}"
  WANDB_ENABLE="${WANDB_ENABLE:-true}"
  JOB_SUFFIX="libero-plus-geop-sft"
fi

NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

cd "${PROJ_ROOT}"

_wan_preflight() {
  local missing=0
  if [[ ! -f "${WAN_CHECKPOINT_PATH}/config.json" ]]; then
    echo "ERROR: WAN config.json missing: ${WAN_CHECKPOINT_PATH}/config.json" >&2
    missing=1
  fi
  if [[ ! -f "${WAN_VAE_PATH}" ]]; then
    echo "ERROR: WAN VAE missing: ${WAN_VAE_PATH}" >&2
    missing=1
  fi
  if [[ "${missing}" -ne 0 ]]; then
    cat >&2 <<EOF
Download Wan2.2-TI2V-5B into ${WAN_CHECKPOINT_PATH} (~32 GB):
  hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ${WAN_CHECKPOINT_PATH}
Or set WAN_CHECKPOINT_PATH / WAN_VAE_PATH to an existing install.
EOF
    exit 1
  fi
}
_wan_preflight

for repo in ${DATASET_REPO_ID}; do
  ds_root="${HF_LEROBOT_HOME}/${repo}"
  if [[ ! -d "${ds_root}" ]]; then
    echo "ERROR: dataset not found: ${ds_root}" >&2
    echo "Expected GeoP mix at /tmp/DATA/libero_plus_geop (set HF_LEROBOT_HOME=/tmp/DATA)." >&2
    exit 1
  fi
  if [[ ! -f "${ds_root}/meta/info.json" ]]; then
    echo "ERROR: missing ${ds_root}/meta/info.json" >&2
    exit 1
  fi
  if [[ ! -f "${ds_root}/meta/keypoints_meta.json" ]]; then
    echo "ERROR: missing keypoints_meta.json — run util_scripts/setup_libero_plus_geop.sh first." >&2
    exit 1
  fi
done

JOB_NAME="${JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-libero-plus-${ACTION_TYPE}-${PRETRAINED_DETAIL}-${JOB_SUFFIX}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/outputs/${POLICY}/${JOB_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"

echo "=== Continue SFT: LIBERO-plus GeoP 7D (from LIBERO 50k) ==="
echo "PROJ_ROOT=${PROJ_ROOT}"
echo "PYTHON=${PYTHON}"
echo "HF_HOME=${HF_HOME}"
echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "WARMUP_CKPT=${WARMUP_CKPT}"
echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "WAN_CHECKPOINT_PATH=${WAN_CHECKPOINT_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "OPTIMIZER_LR=${OPTIMIZER_LR}"
USE_DEEPSPEED="${USE_DEEPSPEED:-0}"
echo "SMOKE=${SMOKE} WAN_SMOKE=${WAN_SMOKE} PROC=${NUM_PROCESSES} BS=${BATCH_SIZE} STEPS=${STEPS} USE_DEEPSPEED=${USE_DEEPSPEED}"
echo "num_keypoint_joints=8, kpt_4d_mode=pos_rot (56D)"

if [[ "${NODE_RANK}" == "0" ]]; then
  for repo in ${DATASET_REPO_ID}; do
    info_json="${HF_LEROBOT_HOME}/${repo}/meta/info.json"
    "${PYTHON}" -c "
import json
p = '${info_json}'
target = 'libero_plus'
with open(p) as f:
    d = json.load(f)
if d.get('robot_type') != target:
    d['robot_type'] = target
    with open(p, 'w') as f:
        json.dump(d, f, indent=4)
    print(f'[info-patch] robot_type -> {target} in {p}')
else:
    print(f'[info-patch] robot_type already {target}')
if 'observation.keypoint_3d' not in d.get('features', {}):
    raise SystemExit(f'[info-patch] ERROR: observation.keypoint_3d missing in {p}')
"
  done
fi

mkdir -p "$(dirname "${LOG_FILE}")"

LAUNCH_ARGS=()
if [[ "${USE_DEEPSPEED}" == "1" ]]; then
  # accelerate rejects --multi_gpu together with --use_deepspeed.
  # num_processes still selects the GPU count; deepspeed_launcher owns the processes.
  LAUNCH_ARGS+=(--use_deepspeed --deepspeed_config_file="${SCRIPT_DIR}/ds_zero2.json")
elif [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  LAUNCH_ARGS+=(--multi_gpu)
fi
LAUNCH_ARGS+=(
  --num_processes="${NUM_PROCESSES}"
  --num_machines="${NODE_COUNT}"
  --machine_rank="${NODE_RANK}"
  --main_process_ip="${MASTER_ADDR}"
  --main_process_port="${MASTER_PORT}"
)

ARGS=(
  "${LAUNCH_ARGS[@]}"
  src/lerobot/scripts/lerobot_train.py

  --output_dir="${OUTPUT_DIR}"
  --num_workers="${NUM_WORKERS}"
  --job_name="${JOB_NAME}"

  --policy.type="${POLICY}"
  --policy.repo_id=lerobot_lab/"${POLICY}"
  --policy.push_to_hub=false
  --policy.pretrained_path="${WARMUP_CKPT}"
  --policy.gradient_checkpointing=true
  --policy.dtype=bfloat16
  --policy.vlm_model_name_or_path="${VLM_MODEL_PATH}"

  --policy.optimizer_lr="${OPTIMIZER_LR}"
  --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP}"
  --policy.scheduler_decay_steps="${STEPS}"
  --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}"

  --policy.train_expert_only=false
  --policy.knowledge_insulation=false
  --policy.knowledge_insulation_kpt=false
  --policy.freeze_vision_encoder=false
  --policy.enable_vqa_loss=true
  --policy.tokenize_state=true

  --policy.action_loss_only=false
  --policy.video_loss_weight=1
  --policy.video_loss_only=false
  --policy.freeze_wan_dit=true
  --policy.freeze_learnable_tokens=true
  --policy.num_learnable_tokens=50
  --policy.wan_checkpoint_path="${WAN_CHECKPOINT_PATH}"
  --policy.wan_config_path="${WAN_CONFIG_PATH}"
  --policy.vae_path="${WAN_VAE_PATH}"
  --policy.video_micro_batch_size="${VIDEO_MICRO_BATCH_SIZE}"

  --policy.enable_keypoint_predictor=true
  --policy.num_keypoint_joints=8
  --policy.kpt_4d_mode=pos_rot
  --policy.keypoint_history_max_len=200
  --policy.action_loss_weight=10.0
  --policy.kpt_loss_weight=1.0
  --policy.kpt_future_loss_weight=1.5
  --policy.kpt_to_action_detach=false
  --policy.init_kpt_expert_from_action=false
  --policy.freeze_keypoint_modules=false
  --policy.action_expert_lr_scale=1.0
  --policy.kpt_expert_lr_scale=1.0
  --policy.track_encoder_lr_scale=1.0

  --dataset.type="${POLICY}"
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.enable_keypoint_predictor=true
  --dataset.num_keypoint_joints=8
  --dataset.kpt_4d_mode=pos_rot
  --dataset.keypoint_history_max_len=200
  --dataset.action_mode="${ACTION_TYPE}"
  --dataset.use_external_stats="${USE_EXTERNAL_STATS}"
  --dataset.dist_loading=false
  --dataset.tokenize_state=true
  --dataset.use_fast_action_tokens=true
  --dataset.video_backend=torchcodec
  # --dataset.image_transforms.enable=true
  # --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":0.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}'

  --seed=42
  --batch_size="${BATCH_SIZE}"
  --steps="${STEPS}"
  --save_freq="${SAVE_FREQ}"
  --log_freq="${LOG_FREQ}"

  --wandb.enable="${WANDB_ENABLE}"
  --wandb.project="${POLICY}"
  --wandb.mode=offline
)

set -o pipefail
"${PYTHON}" -m accelerate.commands.launch "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
train_exit=${PIPESTATUS[0]}

decode_err=$(grep -c '\[video_decode_error\]' "${LOG_FILE}" 2>/dev/null) || decode_err=0
zero_frames=$(grep -c 'using_zeros' "${LOG_FILE}" 2>/dev/null) || zero_frames=0
echo "post_check: video_decode_error=${decode_err} using_zeros=${zero_frames} exit=${train_exit}"
exit "${train_exit}"
