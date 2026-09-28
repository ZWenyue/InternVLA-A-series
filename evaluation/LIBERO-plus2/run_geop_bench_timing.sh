#!/usr/bin/env bash
# Module-timing smoke for LIBERO-plus2 (EGL + co-located venv topology).
#
# Starts one policy server on GPU 0, runs bench_module_timing.py for NUM_RUNS
# episodes, prints per-module wall times, then tears the server down.
#
# Usage:
#   bash evaluation/LIBERO-plus2/run_geop_bench_timing.sh
#   NUM_RUNS=5 MAX_STEPS=80 bash evaluation/LIBERO-plus2/run_geop_bench_timing.sh
#   RENDER_BACKEND=osmesa bash evaluation/LIBERO-plus2/run_geop_bench_timing.sh   # A/B
set -euo pipefail
cd /home/a26160/SRC/InternVLA-A-series

export CKPT_PATH="${CKPT_PATH:-/home/a26160/SRC/InternVLA-A-series/outputs/internvla_a1_5/2026_09_19_03_16_25-internvla_a1_5-libero-abs-geop_warmup-libero-geop-sft/checkpoints/025000/pretrained_model}"
export LIBERO_HOME="${LIBERO_HOME:-/home/a26160/SRC/LIBERO-plus}"
export SERVER_VENV="${SERVER_VENV:-/home/a26160/ENV/internvla_a1_5}"
export CLIENT_VENV="${CLIENT_VENV:-/opt/conda/envs/libero-plus}"
export VLM_MODEL_PATH="${VLM_MODEL_PATH:-/home/a26160/CKPT/Qwen3.5-2B}"
export STATS_KEY="${STATS_KEY:-libero_spatial}"
export ROBOT_TYPE="${ROBOT_TYPE:-panda}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
export ACTION_LOSS_ONLY_FLAG="${ACTION_LOSS_ONLY_FLAG:---action_loss_only}"
# Match run_geop_full.sh after the EGL switch: co-located infer + EGL render.
export RENDER_BACKEND="${RENDER_BACKEND:-egl}"
export GPU_ID="${GPU_ID:-0}"
export PORT="${PORT:-5820}"
export NUM_RUNS="${NUM_RUNS:-5}"
export MAX_STEPS="${MAX_STEPS:-80}"
export REPLAN_STEPS="${REPLAN_STEPS:-8}"
export TASK_SUITE="${TASK_SUITE:-libero_spatial}"
export TASK_ID="${TASK_ID:-0}"
export SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-120}"
export DISABLE_KEYPOINTS="${DISABLE_KEYPOINTS:-}"

PROJ="$(pwd)"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${CKPT_PATH}/../libero_plus_bench_timing_$(date +%Y%m%d%H%M)}"
mkdir -p "${EVAL_LOG_DIR}"
JSON_OUT="${EVAL_LOG_DIR}/module_timing.json"
SERVER_LOG="${EVAL_LOG_DIR}/server.log"
BENCH_LOG="${EVAL_LOG_DIR}/bench.log"
LIBERO_CONFIG_DIR="${EVAL_LOG_DIR}/libero_config"
mkdir -p "${LIBERO_CONFIG_DIR}"

LP_BENCH="${LIBERO_HOME}/libero/libero"
cat > "${LIBERO_CONFIG_DIR}/config.yaml" <<YAML
benchmark_root: ${LP_BENCH}
bddl_files: ${LP_BENCH}/bddl_files
init_states: ${LP_BENCH}/init_files
datasets: ${LP_BENCH}/../datasets
assets: ${LP_BENCH}/assets
YAML

SERVER_PYTHON="${SERVER_VENV}/bin/python"
CLIENT_PYTHON="${CLIENT_VENV}/bin/python"
if [[ ! -x "${SERVER_PYTHON}" ]]; then
  echo "ERROR: SERVER_PYTHON not executable: ${SERVER_PYTHON}" >&2
  exit 2
fi
if [[ ! -x "${CLIENT_PYTHON}" ]]; then
  echo "ERROR: CLIENT_PYTHON not executable: ${CLIENT_PYTHON}" >&2
  exit 2
fi

# NVIDIA EGL ICD: without this MuJoCo sees 0 devices and raises
# "MUJOCO_EGL_DEVICE_ID must be between 0 and -1, got 0".
EGL_VENDOR_DIR="${EGL_VENDOR_DIR:-${CLIENT_VENV}/egl_vendor.d}"
NVIDIA_EGL_LIB="${NVIDIA_EGL_LIB:-/usr/local/nvidia/lib64/libEGL_nvidia.so.0}"
ensure_egl_vendor() {
  if [[ ! -f "${NVIDIA_EGL_LIB}" ]]; then
    echo "ERROR: ${NVIDIA_EGL_LIB} missing; cannot use RENDER_BACKEND=egl" >&2
    exit 2
  fi
  mkdir -p "${EGL_VENDOR_DIR}"
  if [[ ! -f "${EGL_VENDOR_DIR}/10_nvidia.json" ]]; then
    cat > "${EGL_VENDOR_DIR}/10_nvidia.json" <<JSON
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "${NVIDIA_EGL_LIB}"
    }
}
JSON
    echo "[$(date)] created ${EGL_VENDOR_DIR}/10_nvidia.json"
  fi
}
if [[ "${RENDER_BACKEND}" == "egl" ]]; then
  ensure_egl_vendor
fi

KPT_FLAGS=(--enable_keypoints --kpt_r_pad "${KPT_R_PAD:-0.9613106459379196}")
if [[ -n "${DISABLE_KEYPOINTS}" ]]; then
  KPT_FLAGS=(--no-enable_keypoints)
fi

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "============================================"
echo "LIBERO-plus2 module timing bench"
echo "  ckpt:    ${CKPT_PATH}"
echo "  GPU:     ${GPU_ID}"
echo "  port:    ${PORT}"
echo "  render:  ${RENDER_BACKEND}"
echo "  runs:    ${NUM_RUNS}  max_steps=${MAX_STEPS}  replan=${REPLAN_STEPS}"
echo "  suite:   ${TASK_SUITE} task_id=${TASK_ID}"
echo "  out:     ${EVAL_LOG_DIR}"
echo "============================================"

T0=$(date +%s)
(
  export PYTHONPATH="${PROJ}:${PROJ}/src:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  # Server must not init MuJoCo/EGL.
  unset MUJOCO_GL PYOPENGL_PLATFORM MUJOCO_EGL_DEVICE_ID __EGL_VENDOR_LIBRARY_DIRS
  exec "${SERVER_PYTHON}" evaluation/LIBERO2/policy_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --host 0.0.0.0 --port "${PORT}" --device cuda \
    --resize_size 224 \
    --stats_key "${STATS_KEY}" --robot_type "${ROBOT_TYPE}" \
    ${VLM_MODEL_PATH:+--vlm_model_path "${VLM_MODEL_PATH}"} \
    ${ACTION_LOSS_ONLY_FLAG} \
    --inference_backend "${INFERENCE_BACKEND}" \
    --idle_timeout -1
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "[$(date)] server pid=${SERVER_PID} (log: ${SERVER_LOG})"

echo "[$(date)] waiting up to ${SERVER_STARTUP_WAIT}s for server..."
READY=0
for ((i = 0; i < SERVER_STARTUP_WAIT; i += 2)); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: server exited early; see ${SERVER_LOG}" >&2
    exit 1
  fi
  if (
    export PYTHONPATH="${PROJ}:${PROJ}/src:${PYTHONPATH:-}"
    "${SERVER_PYTHON}" - "${PORT}" <<'PY'
import sys
from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy
port = int(sys.argv[1])
c = WebsocketClientPolicy(host="127.0.0.1", port=port)
m = c.get_server_metadata()
assert m.get("preprocessing_owner") == "server_canonical", m
print("ok")
PY
  ) >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
T_READY=$(date +%s)
if [[ "${READY}" -ne 1 ]]; then
  echo "ERROR: server healthcheck failed after ${SERVER_STARTUP_WAIT}s; see ${SERVER_LOG}" >&2
  exit 1
fi
echo "[$(date)] server ready in $((T_READY - T0))s"

echo "[$(date)] starting ${NUM_RUNS}-run timing bench..."
set +e
(
  export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"
  export PYTHONPATH="${LIBERO_HOME}:${PROJ}:${PYTHONPATH:-}"
  export RENDER_BACKEND="${RENDER_BACKEND}"
  export MAGICK_HOME="${CLIENT_VENV}"
  if [[ "${RENDER_BACKEND}" == "egl" ]]; then
    # Physical EGL index (matches nvidia-smi ordinal on this host).
    export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
    export __EGL_VENDOR_LIBRARY_DIRS="${EGL_VENDOR_DIR}"
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  fi
  if [[ "${RENDER_BACKEND}" == "osmesa" ]]; then
    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
    export GALLIUM_DRIVER=llvmpipe
    unset MUJOCO_EGL_DEVICE_ID __EGL_VENDOR_LIBRARY_DIRS
  fi
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export CLIENT_VENV="${CLIENT_VENV}"
  export LD_LIBRARY_PATH="${CLIENT_VENV}/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
  "${CLIENT_PYTHON}" evaluation/LIBERO-plus2/bench_module_timing.py \
    --host 127.0.0.1 --port "${PORT}" \
    --num_runs "${NUM_RUNS}" \
    --max_steps "${MAX_STEPS}" \
    --replan_steps "${REPLAN_STEPS}" \
    --task_suite_name "${TASK_SUITE}" \
    --task_id "${TASK_ID}" \
    --gripper_convention libero_native \
    --no-rotate_images \
    "${KPT_FLAGS[@]}" \
    --json_out "${JSON_OUT}"
)
BENCH_RC=$?
set -e
T_END=$(date +%s)

echo "[$(date)] bench exit=${BENCH_RC}  wall=$((T_END - T0))s (incl. server startup)"
echo "Timing JSON: ${JSON_OUT}"
echo "Server log : ${SERVER_LOG}"
exit "${BENCH_RC}"
