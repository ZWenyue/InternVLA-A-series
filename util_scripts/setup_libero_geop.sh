#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 20 Hz LIBERO GeoP from libero_rlds_lerobot (joints kept, wrist_image kept).
#
#   1. Export MountedPanda MJCF (LIBERO-plus) if needed
#   2. Copy 4 suites -> output, FK keypoints with shared robot0_base R_pad
#   3. Optionally symlink into data/
#
# Usage:
#   bash util_scripts/setup_libero_geop_rlds.sh
#   FORCE=1 bash util_scripts/setup_libero_geop_rlds.sh
#   SMOKE=1 bash util_scripts/setup_libero_geop_rlds.sh
#
# Env overrides:
#   SOURCE_ROOT=/tmp/DATA/libero_rlds_lerobot
#   OUTPUT_DIR=/tmp/DATA/libero_geop_rlds20
#   MJCF_PATH=/tmp/DATA/panda_libero_plus.xml
#   SKIP_MJCF=1  SKIP_KEYPOINTS=1  SKIP_SYMLINK=1
#   FORCE=1  SMOKE=1
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE_ROOT="${SOURCE_ROOT:-/tmp/DATA/libero_rlds_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/DATA/libero_geop_rlds20}"
MJCF_PATH="${MJCF_PATH:-/tmp/DATA/panda_libero_plus.xml}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/home/a26160/SRC/LIBERO-plus}"
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

echo "=== LIBERO GeoP from RLDS (20 Hz) ==="
echo "SOURCE_ROOT=${SOURCE_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MJCF_PATH=${MJCF_PATH}"
echo "coordinate_system=robot0_base-relative"
echo "cameras=image + wrist_image (no rename)"

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "ERROR: SOURCE_ROOT not found: ${SOURCE_ROOT}" >&2
  exit 1
fi

cd "${PROJ_ROOT}"

if [[ "${SKIP_MJCF}" != "1" ]]; then
  echo ""
  echo ">>> Step 1/3: Export MountedPanda MJCF"
  run_py util_scripts/export_panda_mjcf.py \
    --output "${MJCF_PATH}" \
    --libero-plus "${LIBERO_PLUS_ROOT}"
else
  echo ">>> Step 1/3: SKIP (MJCF)"
  [[ -f "${MJCF_PATH}" ]] || { echo "ERROR: missing ${MJCF_PATH}" >&2; exit 1; }
fi

if [[ "${SMOKE}" == "1" ]]; then
  echo ""
  echo ">>> SMOKE: FK on libero_spatial only"
  run_py util_scripts/smoke_test_libero_fk.py \
    --source "${SOURCE_ROOT}/libero_spatial_no_noops_1.0.0_lerobot" \
    --xml "${MJCF_PATH}"
  echo "SMOKE OK. Re-run without SMOKE=1 for full conversion."
  exit 0
fi

if [[ "${SKIP_KEYPOINTS}" != "1" ]]; then
  echo ""
  echo ">>> Step 2/3: Generate keypoints (shared R_pad, keep wrist_image)"
  KPT_ARGS=(
    --source-root "${SOURCE_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --xml "${MJCF_PATH}"
  )
  if [[ "${FORCE}" == "1" ]]; then
    KPT_ARGS+=(--force)
  fi
  run_py util_scripts/generate_libero_geop_from_rlds.py "${KPT_ARGS[@]}"
else
  echo ">>> Step 2/3: SKIP (keypoints)"
fi

if [[ "${SKIP_SYMLINK}" != "1" ]]; then
  echo ""
  echo ">>> Step 3/3: Symlink suites into ${PROJ_ROOT}/data/"
  mkdir -p "${PROJ_ROOT}/data"
  for suite in libero_spatial libero_object libero_goal libero_10; do
    src="${OUTPUT_DIR}/${suite}"
    dst="${PROJ_ROOT}/data/${suite}"
    if [[ ! -d "${src}" ]]; then
      echo "WARNING: ${src} missing, skip"
      continue
    fi
    if [[ -L "${dst}" || -e "${dst}" ]]; then
      echo "  ${dst} exists, skipping (remove manually to relink to 20 Hz)"
    else
      ln -s "${src}" "${dst}"
      echo "  linked ${dst} -> ${src}"
    fi
  done
else
  echo ">>> Step 3/3: SKIP (symlinks)"
fi

echo ""
echo "=== Done ==="
echo "20 Hz GeoP suites: ${OUTPUT_DIR}/libero_{spatial,object,goal,10}"
echo "DATASET_REPO_ID='libero_spatial libero_object libero_goal libero_10'"
echo "Train with:  export HF_LEROBOT_HOME=${OUTPUT_DIR}"
echo "  (launch/internvla_a15_geop_phase1_libero_warmup.sh defaults to this path)"
