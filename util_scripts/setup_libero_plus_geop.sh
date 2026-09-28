#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 20 Hz LIBERO-plus GeoP from libero_plus_mix_lerobot (joints kept, wrist kept).
#
# Mirrors setup_libero_geop_rlds.sh, but for the single mixed LeRobot dataset
# produced by lerobot/examples/port_datasets/port_libero_plus.py.
#
#   1. Export MountedPanda MJCF (LIBERO-plus) if needed
#   2. Copy mix dataset -> output, FK keypoints with R_pad
#   3. Optionally symlink into data/libero_plus
#
# Usage:
#   source /home/a26160/ENV/internvla_a1_5/bin/activate
#   bash util_scripts/setup_libero_plus_geop.sh
#   FORCE=1 bash util_scripts/setup_libero_plus_geop.sh
#   SMOKE=1 bash util_scripts/setup_libero_plus_geop.sh
#
# Env overrides:
#   SOURCE=/tmp/DATA/libero_plus_mix_lerobot
#   OUTPUT_DIR=/tmp/DATA/libero_plus_geop
#   MJCF_PATH=/tmp/DATA/panda_libero_plus.xml
#   LIBERO_PLUS_ROOT=.../evaluation/LIBERO-plus
#   SKIP_MJCF=1  SKIP_KEYPOINTS=1  SKIP_SYMLINK=1
#   FORCE=1  SMOKE=1
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE="${SOURCE:-/tmp/DATA/libero_plus_mix_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/DATA/libero_plus_geop}"
MJCF_PATH="${MJCF_PATH:-/tmp/DATA/panda_libero_plus.xml}"
# Prefer the vendored LIBERO-plus under evaluation/ (default SRC/LIBERO-plus often missing).
if [[ -z "${LIBERO_PLUS_ROOT:-}" ]]; then
  if [[ -d "${PROJ_ROOT}/evaluation/LIBERO-plus" ]]; then
    LIBERO_PLUS_ROOT="${PROJ_ROOT}/evaluation/LIBERO-plus"
  else
    LIBERO_PLUS_ROOT="/home/a26160/SRC/LIBERO-plus"
  fi
fi
PYTHON="${PYTHON:-python}"

SKIP_MJCF="${SKIP_MJCF:-0}"
SKIP_KEYPOINTS="${SKIP_KEYPOINTS:-0}"
SKIP_SYMLINK="${SKIP_SYMLINK:-0}"
SMOKE="${SMOKE:-0}"
FORCE="${FORCE:-0}"

export LD_LIBRARY_PATH_EGL="${LD_LIBRARY_PATH_EGL:-/usr/local/nvidia/lib64}"
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-${HOME}/.local/share/glvnd/egl_vendor.d}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${LIBERO_PLUS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_py() {
  env -i \
    HOME="${HOME}" USER="${USER:-}" \
    PATH="${PATH}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH_EGL}" \
    __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS}" \
    MUJOCO_GL="${MUJOCO_GL}" \
    PYTHONPATH="${PYTHONPATH}" \
    LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT}" \
    "${PYTHON}" "$@"
}

echo "=== LIBERO-plus GeoP from mix LeRobot (20 Hz) ==="
echo "SOURCE=${SOURCE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MJCF_PATH=${MJCF_PATH}"
echo "LIBERO_PLUS_ROOT=${LIBERO_PLUS_ROOT}"
echo "coordinate_system=robot0_base-relative"
echo "cameras=image + wrist_image (no rename)"

if [[ ! -d "${SOURCE}" ]]; then
  echo "ERROR: SOURCE not found: ${SOURCE}" >&2
  echo "Convert RLDS first, e.g.:" >&2
  echo "  python examples/port_datasets/port_libero_plus.py \\" >&2
  echo "    --raw-dir /tmp/DATA/libero_plus_mixdata/libero_mix/1.0.0 \\" >&2
  echo "    --repo-id local/libero_plus_mix --root ${SOURCE}" >&2
  exit 1
fi

cd "${PROJ_ROOT}"

if [[ "${SKIP_MJCF}" != "1" ]]; then
  echo ""
  echo ">>> Step 1/3: Export MountedPanda MJCF"
  if [[ ! -d "${LIBERO_PLUS_ROOT}" ]]; then
    echo "ERROR: LIBERO-plus root not found: ${LIBERO_PLUS_ROOT}" >&2
    exit 1
  fi
  run_py util_scripts/export_panda_mjcf.py \
    --output "${MJCF_PATH}" \
    --libero-plus "${LIBERO_PLUS_ROOT}"
else
  echo ">>> Step 1/3: SKIP (MJCF)"
  [[ -f "${MJCF_PATH}" ]] || { echo "ERROR: missing ${MJCF_PATH}" >&2; exit 1; }
fi

if [[ "${SMOKE}" == "1" ]]; then
  echo ""
  echo ">>> SMOKE: FK on first parquet only (via generate_libero_keypoints dry path)"
  # Lightweight: run full generate but user should use a tiny SOURCE for smoke.
  # Prefer dedicated smoke helper if present; else run generate with --force on a smoke output.
  SMOKE_OUT="${OUTPUT_DIR}_smoke"
  KPT_ARGS=(
    --source "${SOURCE}"
    --output "${SMOKE_OUT}"
    --xml "${MJCF_PATH}"
    --force
  )
  # Cap is not built into generate_libero_plus_geop; for true smoke use a small SOURCE
  # (e.g. /tmp/DATA/libero_plus_mix_smoke). Document that here.
  if [[ "$(basename "${SOURCE}")" != *"smoke"* ]]; then
    echo "WARNING: SMOKE=1 on full mix is expensive. Prefer SOURCE=.../libero_plus_mix_smoke" >&2
  fi
  run_py util_scripts/generate_libero_plus_geop.py "${KPT_ARGS[@]}"
  echo "SMOKE OK -> ${SMOKE_OUT}"
  exit 0
fi

if [[ "${SKIP_KEYPOINTS}" != "1" ]]; then
  echo ""
  echo ">>> Step 2/3: Generate keypoints (R_pad, keep wrist_image)"
  KPT_ARGS=(
    --source "${SOURCE}"
    --output "${OUTPUT_DIR}"
    --xml "${MJCF_PATH}"
  )
  if [[ "${FORCE}" == "1" ]]; then
    KPT_ARGS+=(--force)
  fi
  run_py util_scripts/generate_libero_plus_geop.py "${KPT_ARGS[@]}"
else
  echo ">>> Step 2/3: SKIP (keypoints)"
fi

if [[ "${SKIP_SYMLINK}" != "1" ]]; then
  echo ""
  echo ">>> Step 3/3: Symlink into ${PROJ_ROOT}/data/libero_plus"
  mkdir -p "${PROJ_ROOT}/data"
  src="${OUTPUT_DIR}"
  dst="${PROJ_ROOT}/data/libero_plus"
  if [[ ! -d "${src}" ]]; then
    echo "WARNING: ${src} missing, skip symlink"
  elif [[ -L "${dst}" || -e "${dst}" ]]; then
    echo "  ${dst} exists, skipping (remove manually to relink)"
  else
    ln -s "${src}" "${dst}"
    echo "  linked ${dst} -> ${src}"
  fi
else
  echo ">>> Step 3/3: SKIP (symlink)"
fi

echo ""
echo "=== Done ==="
echo "20 Hz GeoP mix: ${OUTPUT_DIR}"
echo "DATASET_REPO_ID='libero_plus'  (or point HF_LEROBOT_HOME at parent and use repo id)"
echo "Train with:  export HF_LEROBOT_HOME=$(dirname "${OUTPUT_DIR}")"
echo "  and dataset name: $(basename "${OUTPUT_DIR}")"
