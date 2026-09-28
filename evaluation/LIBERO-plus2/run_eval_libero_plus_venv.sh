#!/usr/bin/env bash
set -euo pipefail

# ─── Required parameters ───
: "${CKPT_PATH:?ERROR: CKPT_PATH not set}"
: "${LIBERO_HOME:?ERROR: LIBERO_HOME not set}"
: "${SERVER_VENV:?ERROR: SERVER_VENV not set}"
: "${CLIENT_VENV:?ERROR: CLIENT_VENV not set}"

# ─── Optional parameters with defaults ───
PROJ="${PROJ:-$(cd "$(dirname "$0")/../.." && pwd)}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-5784}"
SHARDS_PER_SUITE="${SHARDS_PER_SUITE:-8}"
STATS_KEY_MODE="${STATS_KEY_MODE:-panda}"
ROBOT_TYPE_MODE="${ROBOT_TYPE_MODE:-panda}"
GRIPPER_CONVENTION="${GRIPPER_CONVENTION:-libero_native}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
SEED="${SEED:-7}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
ACTION_LOSS_ONLY_FLAG="${ACTION_LOSS_ONLY_FLAG:---action_loss_only}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
SAVE_ACTIONS_FLAG="${SAVE_ACTIONS_FLAG:---save_actions}"
# SAVE_FAILURE_VIDEOS: set to "true" to save MP4 for failed episodes (default off; JSON always saved)
SAVE_FAILURE_VIDEOS="${SAVE_FAILURE_VIDEOS:-false}"
# Output lives under the checkpoint step dir: .../checkpoints/<step>/libero_plus_YYYYMMDDHHMM/
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${CKPT_PATH}/../libero_plus_$(date +%Y%m%d%H%M)}"
CATEGORIES="${CATEGORIES:-}"
DISABLE_KEYPOINTS="${DISABLE_KEYPOINTS:-}"
# Must match libero_geop_rlds20 keypoints_meta.json bbox_radius (robot0_base-relative).
KPT_R_PAD="${KPT_R_PAD:-0.9613106459379196}"
ROTATE_IMAGES="${ROTATE_IMAGES:-false}"
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-90}"
# Render backend: auto | egl | osmesa
#   egl    = NVIDIA GPU, fast, but subject to MuJoCo EGL SIGABRT in forked children
#   osmesa = Mesa llvmpipe CPU, immune to EGL SIGABRT, ~slower (needs libosmesa6)
RENDER_BACKEND="${RENDER_BACKEND:-auto}"

# Prefer explicit python bins (conda envs here may lack bin/activate).
SERVER_PYTHON="${SERVER_PYTHON:-${SERVER_VENV}/bin/python}"
CLIENT_PYTHON="${CLIENT_PYTHON:-${CLIENT_VENV}/bin/python}"
if [[ ! -x "${SERVER_PYTHON}" ]]; then
    echo "ERROR: SERVER_PYTHON not executable: ${SERVER_PYTHON}" >&2
    exit 2
fi
if [[ ! -x "${CLIENT_PYTHON}" ]]; then
    echo "ERROR: CLIENT_PYTHON not executable: ${CLIENT_PYTHON}" >&2
    exit 2
fi

# PYTHONHOME=/opt/conda (common under conda activate) makes SERVER_PYTHON load
# base-site torch, which does not match this env's causal_conv1d CUDA extension.
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PATH="$(dirname "${SERVER_PYTHON}"):${PATH}"
# Do not inherit a polluted PYTHONPATH (e.g. /opt/conda/.../site-packages).
SERVER_PYTHONPATH="${PROJ}:${PROJ}/src"
TORCH_FILE="$("${SERVER_PYTHON}" -c 'import torch; print(torch.__file__)')"
# torch must resolve to SERVER_VENV; anything else (e.g. conda base at
# /opt/conda/lib/... via PYTHONHOME) means a polluted env.
case "${TORCH_FILE}" in
  "${SERVER_VENV}"/*) ;;
  *)
    echo "ERROR: server torch not from SERVER_VENV (${SERVER_VENV}): ${TORCH_FILE}" >&2
    echo "Unset PYTHONHOME / remove /opt/conda from PYTHONPATH and retry." >&2
    exit 2
    ;;
esac
echo "Server torch: ${TORCH_FILE}"
# robosuite 1.4 asserts on every env reset with mujoco>=3 (joint-type enum compare).
CLIENT_MUJOCO="$(env -u PYTHONPATH "${CLIENT_PYTHON}" -c 'import mujoco; print(mujoco.__version__, mujoco.__file__)')"
echo "Client mujoco: ${CLIENT_MUJOCO}"
case "${CLIENT_MUJOCO}" in
  2.3.*) ;;
  *)
    echo "ERROR: client mujoco must be 2.3.x for robosuite 1.4 (got ${CLIENT_MUJOCO})" >&2
    echo "Fix: env -u PYTHONPATH ${CLIENT_VENV}/bin/pip install mujoco==2.3.7" >&2
    exit 2
    ;;
esac
SERVER_SITE="$("${SERVER_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:\
${SERVER_SITE}/torch/lib:\
${SERVER_SITE}/nvidia/cuda_runtime/lib:\
${SERVER_SITE}/nvidia/cuda_nvrtc/lib:\
${SERVER_SITE}/nvidia/npp/lib:\
${CUDA_HOME}/lib64:\
${CLIENT_VENV}/lib:\
${LD_LIBRARY_PATH:-}"

# NVIDIA EGL ICD required when RENDER_BACKEND=egl (otherwise MuJoCo sees 0 devices).
EGL_VENDOR_DIR="${EGL_VENDOR_DIR:-${CLIENT_VENV}/egl_vendor.d}"
NVIDIA_EGL_LIB="${NVIDIA_EGL_LIB:-/usr/local/nvidia/lib64/libEGL_nvidia.so.0}"
if [[ "${RENDER_BACKEND}" == "egl" || "${RENDER_BACKEND}" == "auto" ]]; then
    if [[ -f "${NVIDIA_EGL_LIB}" ]]; then
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
    elif [[ "${RENDER_BACKEND}" == "egl" ]]; then
        echo "ERROR: RENDER_BACKEND=egl but ${NVIDIA_EGL_LIB} is missing" >&2
        exit 2
    fi
fi

TASK_CLS="${LIBERO_HOME}/libero/libero/benchmark/task_classification.json"
# Override for smoke / partial runs, e.g. SUITES="libero_goal"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
# Optional: cap tasks per suite after building shards (smoke). Empty = full suite.
MAX_TASKS_PER_SUITE="${MAX_TASKS_PER_SUITE:-}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================"
echo "LIBERO-plus2 Evaluation"
echo "  Checkpoint: ${CKPT_PATH}"
echo "  GPUs: ${GPU_IDS} (${NUM_GPUS} total)"
echo "  Output: ${EVAL_LOG_DIR}"
echo "  Stats key: ${STATS_KEY_MODE}"
echo "  Robot type: ${ROBOT_TYPE_MODE}"
echo "  Gripper: ${GRIPPER_CONVENTION}"
echo "  Keypoints: $([ -z "${DISABLE_KEYPOINTS}" ] && echo "ENABLED (R_pad=${KPT_R_PAD})" || echo "DISABLED")"
echo "  Rotation: $([ "${ROTATE_IMAGES}" = "true" ] && echo "ENABLED (180°)" || echo "DISABLED (raw)")"
echo "  Render backend: ${RENDER_BACKEND} (${NUM_GPUS} concurrent client workers)"
echo "============================================"

# ─── LIBERO config.yaml (flat format, B8) ───
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

# ─── Build task queue ───
QUEUE_FILE="${EVAL_LOG_DIR}/task_queue.txt"
mkdir -p "${EVAL_LOG_DIR}"
> "${QUEUE_FILE}"
for suite in ${SUITES}; do
    TOTAL=$(python3 -c "import json; d=json.load(open('${TASK_CLS}')); s=d['${suite}']; print(len(s) if isinstance(s,list) else sum(len(v) for v in s.values()))")
    if [ -n "${MAX_TASKS_PER_SUITE}" ]; then
        [ "${TOTAL}" -gt "${MAX_TASKS_PER_SUITE}" ] && TOTAL=${MAX_TASKS_PER_SUITE}
    fi
    SHARD_SIZE=$(( (TOTAL + SHARDS_PER_SUITE - 1) / SHARDS_PER_SUITE ))
    for s in $(seq 0 $((SHARDS_PER_SUITE - 1))); do
        START=$((s * SHARD_SIZE))
        END=$(( (s+1) * SHARD_SIZE ))
        [ ${END} -gt ${TOTAL} ] && END=${TOTAL}
        [ ${START} -ge ${END} ] && continue
        echo "${suite} ${START} ${END}" >> "${QUEUE_FILE}"
    done
done
echo "Task queue: $(wc -l < "${QUEUE_FILE}") shards"

# ─── Pre-assign shards to GPUs (round-robin) ───
for i in $(seq 0 $((NUM_GPUS - 1))); do
    > "${EVAL_LOG_DIR}/gpu${i}_queue.txt"
done
SHARD_IDX=0
while IFS=' ' read -r SUITE START END; do
    GPU_IDX=$((SHARD_IDX % NUM_GPUS))
    echo "${SUITE} ${START} ${END}" >> "${EVAL_LOG_DIR}/gpu${GPU_IDX}_queue.txt"
    SHARD_IDX=$((SHARD_IDX + 1))
done < "${QUEUE_FILE}"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "GPU ${GPU_ARRAY[$i]}: $(wc -l < "${EVAL_LOG_DIR}/gpu${i}_queue.txt") shards"
done

# Resolve stats_key / robot_type for a suite. STATS_KEY_MODE=suite binds to the
# suite name (required for multi-subset GeoP stats.json).
resolve_stats_key() {
    local suite="$1"
    if [[ "${STATS_KEY_MODE}" == "suite" ]]; then
        echo "${suite}"
    else
        echo "${STATS_KEY_MODE}"
    fi
}
resolve_robot_type() {
    local suite="$1"
    if [[ "${ROBOT_TYPE_MODE}" == "suite" ]]; then
        echo "${suite}"
    else
        echo "${ROBOT_TYPE_MODE}"
    fi
}

# ─── GPU worker: restart server when suite (stats_key) changes ───
gpu_worker() {
    local GPU_IDX=$1
    local SLOT_IDX=$2
    local PORT=$((BASE_PORT + SLOT_IDX))
    local WORKER_DIR="${EVAL_LOG_DIR}/worker_gpu${GPU_IDX}"
    local GPU_QUEUE="${EVAL_LOG_DIR}/gpu${SLOT_IDX}_queue.txt"
    mkdir -p "${WORKER_DIR}"

    if [ ! -s "${GPU_QUEUE}" ]; then
        echo "[$(date)] GPU ${GPU_IDX}: No shards assigned, skipping" | tee -a "${WORKER_DIR}/worker.log"
        return 0
    fi

    local KPT_FLAGS=""
    if [ -z "${DISABLE_KEYPOINTS}" ]; then
        KPT_FLAGS="--enable_keypoints --kpt_r_pad ${KPT_R_PAD}"
    else
        KPT_FLAGS="--no-enable_keypoints"
    fi
    local CAT_FLAG=""
    if [ -n "${CATEGORIES}" ]; then
        CAT_FLAG="--categories ${CATEGORIES}"
    fi
    local ROTATE_FLAG=""
    if [ "${ROTATE_IMAGES}" = "true" ]; then
        ROTATE_FLAG="--rotate_images"
    fi
    local FAILURE_VIDEO_FLAG=""
    if [ "${SAVE_FAILURE_VIDEOS}" = "true" ]; then
        FAILURE_VIDEO_FLAG="--save_failure_videos"
    fi

    local SERVER_PID=""
    local CURRENT_STATS_KEY=""
    local SHARD_NUM=0

    start_server_for_suite() {
        local suite="$1"
        local stats_key
        stats_key=$(resolve_stats_key "${suite}")
        local robot_type
        robot_type=$(resolve_robot_type "${suite}")

        if [[ -n "${SERVER_PID}" ]]; then
            kill "${SERVER_PID}" 2>/dev/null || true
            wait "${SERVER_PID}" 2>/dev/null || true
            SERVER_PID=""
        fi

        (
            cd "${PROJ}"
            unset PYTHONHOME
            export PYTHONNOUSERSITE=1
            export PYTHONPATH="${SERVER_PYTHONPATH}"
            CUDA_VISIBLE_DEVICES=${GPU_IDX} exec "${SERVER_PYTHON}" evaluation/LIBERO2/policy_server/server_policy.py \
                --ckpt_path "${CKPT_PATH}" \
                --host 0.0.0.0 --port ${PORT} --device cuda \
                --resize_size ${RESIZE_SIZE} \
                --stats_key "${stats_key}" --robot_type "${robot_type}" \
                ${VLM_MODEL_PATH:+--vlm_model_path "${VLM_MODEL_PATH}"} \
                ${ACTION_LOSS_ONLY_FLAG} \
                --inference_backend "${INFERENCE_BACKEND}" \
                --idle_timeout -1 \
                > "${WORKER_DIR}/server_${suite}.log" 2>&1
        ) &
        SERVER_PID=$!
        echo "${SERVER_PID}" > "${WORKER_DIR}/server.pid"
        CURRENT_STATS_KEY="${stats_key}"
        echo "[$(date)] GPU ${GPU_IDX}: Server PID=${SERVER_PID}, port=${PORT}, stats_key=${stats_key}, waiting ${SERVER_STARTUP_WAIT}s..." | tee -a "${WORKER_DIR}/worker.log"
        sleep ${SERVER_STARTUP_WAIT}

        local HC_OK=0
        for attempt in 1 2 3; do
            if (
                cd "${PROJ}"
                unset PYTHONHOME
                export PYTHONNOUSERSITE=1
                export PYTHONPATH="${SERVER_PYTHONPATH}"
                "${SERVER_PYTHON}" -c "
from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy
c = WebsocketClientPolicy(host='127.0.0.1', port=${PORT})
m = c.get_server_metadata()
am = m.get('action_mode')
assert am in ('joint', 'end_effector'), f'action_mode={am}'
assert m.get('preprocessing_owner') == 'server_canonical'
assert m.get('stats_key') == '${stats_key}', f'stats_key={m.get(\"stats_key\")}'
assert int(m.get('resize_size') or 0) == int(${RESIZE_SIZE}), f'resize_size={m.get(\"resize_size\")}'
print(f'HEALTHCHECK PASSED (gpu${GPU_IDX}, port ${PORT}, stats_key=${stats_key}, action_mode={am}, robot_type={m.get(\"robot_type\")})')
"
            ) >> "${WORKER_DIR}/worker.log" 2>&1; then
                HC_OK=1
                break
            fi
            echo "[$(date)] GPU ${GPU_IDX}: Healthcheck attempt ${attempt} failed, retrying in 30s..." | tee -a "${WORKER_DIR}/worker.log"
            sleep 30
        done

        if [ ${HC_OK} -eq 0 ]; then
            echo "FATAL: GPU ${GPU_IDX} healthcheck failed for suite=${suite} stats_key=${stats_key}" | tee -a "${WORKER_DIR}/worker.log"
            kill ${SERVER_PID} 2>/dev/null || true
            SERVER_PID=""
            return 1
        fi
        return 0
    }

    while IFS=' ' read -r SUITE START END; do
        SHARD_NUM=$((SHARD_NUM + 1))
        local want_key
        want_key=$(resolve_stats_key "${SUITE}")
        if [[ "${want_key}" != "${CURRENT_STATS_KEY}" ]] || [[ -z "${SERVER_PID}" ]]; then
            if ! start_server_for_suite "${SUITE}"; then
                echo "[$(date)] FAILED: gpu${GPU_IDX} could not start server for ${SUITE}; skipping shard" | tee -a "${WORKER_DIR}/worker.log"
                continue
            fi
        fi

        echo "[$(date)] GPU ${GPU_IDX}: Starting shard ${SHARD_NUM} — ${SUITE}[${START}:${END}] (stats_key=${CURRENT_STATS_KEY})" | tee -a "${WORKER_DIR}/worker.log"

        set +e
        (
            cd "${PROJ}"
            export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"
            export RENDER_BACKEND="${RENDER_BACKEND}"
            export RENDER_N_WORKERS="${NUM_GPUS}"
            export MAGICK_HOME="${CLIENT_VENV}"
            if [ "${RENDER_BACKEND}" = "osmesa" ]; then
                export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
                export GALLIUM_DRIVER=llvmpipe
                unset MUJOCO_EGL_DEVICE_ID __EGL_VENDOR_LIBRARY_DIRS
            fi
            if [ "${RENDER_BACKEND}" = "egl" ]; then
                export MUJOCO_EGL_DEVICE_ID=${SLOT_IDX}
                export __EGL_VENDOR_LIBRARY_DIRS="${EGL_VENDOR_DIR}"
            fi
            unset PYTHONHOME
            export PYTHONNOUSERSITE=1
            export LD_LIBRARY_PATH="${CLIENT_VENV}/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
            export PYTHONPATH="${LIBERO_HOME}:${PROJ}"
            export CLIENT_VENV="${CLIENT_VENV}"
            "${CLIENT_PYTHON}" evaluation/LIBERO-plus2/eval_libero_plus.py \
                --host 127.0.0.1 --port ${PORT} \
                --task_suite_name "${SUITE}" \
                --num_trials_per_task ${NUM_TRIALS_PER_TASK} \
                --num_steps_wait ${NUM_STEPS_WAIT} \
                --seed ${SEED} \
                --replan_steps ${REPLAN_STEPS} \
                --start_idx ${START} --end_idx ${END} \
                --task_classification_path "${TASK_CLS}" \
                --eval_log_dir "${EVAL_LOG_DIR}" \
                --gripper_convention "${GRIPPER_CONVENTION}" \
                ${KPT_FLAGS} ${CAT_FLAG} \
                ${ROTATE_FLAG} \
                ${FAILURE_VIDEO_FLAG} ${SAVE_ACTIONS_FLAG} \
                > "${WORKER_DIR}/client_${SUITE}_${START}_${END}.log" 2>&1
        )
        CLIENT_EXIT=$?
        set -e

        if [ ${CLIENT_EXIT} -ne 0 ]; then
            echo "[$(date)] FAILED: gpu${GPU_IDX} ${SUITE}[${START}:${END}] exit=${CLIENT_EXIT}" | tee -a "${WORKER_DIR}/worker.log"
        else
            echo "[$(date)] DONE: gpu${GPU_IDX} ${SUITE}[${START}:${END}]" | tee -a "${WORKER_DIR}/worker.log"
        fi
    done < "${GPU_QUEUE}"

    echo "[$(date)] GPU ${GPU_IDX}: All shards complete, shutting down server" | tee -a "${WORKER_DIR}/worker.log"
    if [[ -n "${SERVER_PID}" ]]; then
        kill ${SERVER_PID} 2>/dev/null || true
        wait ${SERVER_PID} 2>/dev/null || true
    fi
    return 0
}

cleanup_servers() {
    local pidfile pid
    for pidfile in "${EVAL_LOG_DIR}"/worker_gpu*/server.pid; do
        [ -f "${pidfile}" ] || continue
        pid=$(cat "${pidfile}")
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[$(date)] cleanup: killing leftover server PID=${pid}"
            kill "${pid}" 2>/dev/null || true
            sleep 3
            kill -9 "${pid}" 2>/dev/null || true
        fi
        rm -f "${pidfile}"
    done
}
trap cleanup_servers EXIT
trap 'kill $(jobs -p) 2>/dev/null; cleanup_servers; exit 130' INT TERM

# ─── Launch all GPU workers in parallel ───
echo "Starting eval at $(date) — ${NUM_GPUS} GPUs, $(wc -l < "${QUEUE_FILE}") shards"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_worker ${GPU_ARRAY[$i]} $i &
done
wait
echo "All shards complete at $(date)"

# ─── Aggregate results ───
echo "Aggregating results..."
(
    cd "${PROJ}"
    unset PYTHONHOME
    export PYTHONNOUSERSITE=1
    export PYTHONPATH="${LIBERO_HOME}:${PROJ}"
    "${CLIENT_PYTHON}" evaluation/LIBERO-plus2/aggregate_results.py --root "${EVAL_LOG_DIR}"
)
echo "Results: ${EVAL_LOG_DIR}/overall_results.json"
echo "Done at $(date)"
