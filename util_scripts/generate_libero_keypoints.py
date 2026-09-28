#!/usr/bin/env python3
"""Offline MuJoCo FK 7D keypoint generation for LIBERO LeRobot v3 datasets.

Reads per-frame arm joints from `observation.state.joint_position` (7D) and
gripper finger positions from `observation.state[6:8]`, sets MuJoCo qpos by
joint name on a LIBERO-plus MountedPanda MJCF, and writes flattened 7D
keypoints in the **robot0_base** frame (Franka-GeoP style):

    observation.keypoint_3d  shape [K * 7]  (default K=8, px,py,pz,qx,qy,qz,qw)

Position normalization follows the InternVLA R_pad scheme (isotropic, origin
at robot0_base); quaternions are hemisphere-normalized (qw >= 0).

Typical pipeline:
    1. MUJOCO_GL=egl python util_scripts/export_panda_mjcf.py
    2. python util_scripts/generate_libero_keypoints.py \\
           --source /tmp/DATA/libero_lerobot_v3 \\
           --dest   /tmp/DATA/libero_geop/libero_merged_kpt
    3. python util_scripts/split_libero_merged.py ...

Requires: mujoco, pyarrow (via pandas)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 8
KEYPOINT_DIM = 7  # 3 (pos) + 4 (quat xyzw)
ROBOT_NQPOS = 9
BBOX_MARGIN = 0.15

JOINT_COLUMN = "observation.state.joint_position"
STATE_COLUMN = "observation.state"

EEF_BODY_CANDIDATES = ("gripper0_eef", "gripper0_right_eef")
BASE_BODY = "robot0_base"

KEYPOINT_ARM_BODY_NAMES = [
    "robot0_link1",
    "robot0_link2",
    "robot0_link3",
    "robot0_link4",
    "robot0_link5",
    "robot0_link6",
    "robot0_link7",
]

ARM_JOINT_NAMES = [f"robot0_joint{i}" for i in range(1, 8)]
GRIPPER_JOINT_NAMES = ("gripper0_finger_joint1", "gripper0_finger_joint2")

DEFAULT_XML = Path("/tmp/DATA/panda_libero_plus.xml")


def resolve_eef_body_name(model) -> str:
    for name in EEF_BODY_CANDIDATES:
        try:
            model.body(name)
            return name
        except KeyError:
            continue
    raise KeyError(f"No EEF body found (tried {', '.join(EEF_BODY_CANDIDATES)})")


def keypoint_body_names(model) -> list[str]:
    return [*KEYPOINT_ARM_BODY_NAMES, resolve_eef_body_name(model)]


def _mujoco_quat_wxyz_to_xyzw(xquat: np.ndarray) -> np.ndarray:
    """MuJoCo body quat (w, x, y, z) -> xyzw with qw >= 0."""
    raw_q = np.array([xquat[1], xquat[2], xquat[3], xquat[0]], dtype=np.float32)
    if raw_q[3] < 0:
        raw_q = -raw_q
    return raw_q


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    """Inverse of unit quaternion in wxyz."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion wxyz -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class LiberoMujocoFK:
    """MuJoCo FK for LIBERO MountedPanda bodies, expressed in robot0_base."""

    def __init__(self, model_xml: str | Path):
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.data = mujoco.MjData(self.model)
        self.body_names = keypoint_body_names(self.model)
        self.body_ids = [self.model.body(name).id for name in self.body_names]
        self.base_id = self.model.body(BASE_BODY).id

        self._arm_qadr = [int(self.model.jnt_qposadr[self.model.joint(n).id]) for n in ARM_JOINT_NAMES]
        self._grip_qadr = [int(self.model.jnt_qposadr[self.model.joint(n).id]) for n in GRIPPER_JOINT_NAMES]

    def compute(self, qpos9: np.ndarray) -> np.ndarray:
        """qpos9 [9] = 7 arm + 2 fingers -> keypoints [K, 7] in robot0_base."""
        qpos9 = np.asarray(qpos9, dtype=np.float64).reshape(-1)
        if qpos9.shape[0] != ROBOT_NQPOS:
            raise ValueError(f"Expected qpos length {ROBOT_NQPOS}, got {qpos9.shape}")

        for adr, val in zip(self._arm_qadr, qpos9[:7], strict=True):
            self.data.qpos[adr] = val
        for adr, val in zip(self._grip_qadr, qpos9[7:9], strict=True):
            self.data.qpos[adr] = val

        self._mujoco.mj_forward(self.model, self.data)

        base_pos = self.data.xpos[self.base_id].copy()
        base_quat = self.data.xquat[self.base_id].copy()  # wxyz
        R_base = _quat_wxyz_to_rotmat(base_quat)
        base_quat_inv = _quat_conjugate_wxyz(base_quat)

        keypoints = np.empty((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        for i, bid in enumerate(self.body_ids):
            pos_world = self.data.xpos[bid]
            keypoints[i, :3] = R_base.T @ (pos_world - base_pos)

            body_quat = self.data.xquat[bid]  # wxyz
            rel_wxyz = _quat_mul_wxyz(base_quat_inv, body_quat)
            keypoints[i, 3:7] = _mujoco_quat_wxyz_to_xyzw(rel_wxyz)
        return keypoints

    def compute_batch(self, qpos_batch: np.ndarray) -> np.ndarray:
        out = np.empty((len(qpos_batch), NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        for i, qpos in enumerate(qpos_batch):
            out[i] = self.compute(qpos)
        return out


def compute_r_pad(global_min: np.ndarray, global_max: np.ndarray, margin: float = BBOX_MARGIN) -> float:
    extremes = np.maximum(np.abs(global_min), np.abs(global_max))
    return float(extremes.max() * (1.0 + margin))


def _copy_dataset(source: Path, dest: Path, force: bool) -> None:
    if dest.exists():
        if force:
            logger.info("Removing existing destination %s (--force)", dest)
            shutil.rmtree(dest)
        else:
            raise FileExistsError(
                f"Destination {dest} already exists. Pass --force to overwrite, or use --skip-copy."
            )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        logger.info("Copying dataset %s -> %s (rsync -a)", source, dest)
        subprocess.run(["rsync", "-a", f"{source}/", f"{dest}/"], check=True)
    else:
        logger.warning("rsync not found; falling back to shutil.copytree")
        shutil.copytree(source, dest, symlinks=True)


def _get_parquet_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")
    return files


def _build_qpos(joint: np.ndarray, state: np.ndarray) -> np.ndarray:
    joint = np.asarray(joint, dtype=np.float64).reshape(-1)
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if joint.shape[0] != 7:
        raise ValueError(f"Expected 7 arm joints, got shape {joint.shape}")
    if state.shape[0] < 8:
        raise ValueError(f"Expected observation.state with at least 8 dims, got {state.shape}")
    qpos = np.empty(ROBOT_NQPOS, dtype=np.float64)
    qpos[:7] = joint
    qpos[7:9] = state[6:8]
    return qpos


def _read_frame_kinematics(df: pd.DataFrame, pq_path: Path | None = None) -> np.ndarray | None:
    if len(df) == 0:
        logger.warning("Skipping empty parquet: %s", pq_path)
        return None
    if JOINT_COLUMN not in df.columns:
        raise KeyError(
            f"Required column '{JOINT_COLUMN}' missing in {pq_path}. "
            f"Available: {list(df.columns)}"
        )
    if STATE_COLUMN not in df.columns:
        raise KeyError(f"Required column '{STATE_COLUMN}' missing in {pq_path}.")

    qpos = np.stack(
        [_build_qpos(j, s) for j, s in zip(df[JOINT_COLUMN].values, df[STATE_COLUMN].values, strict=True)]
    )
    if np.any(np.isnan(qpos)):
        raise ValueError(f"NaN in qpos while reading {pq_path}")
    return qpos


def pass1_compute_bbox(parquet_files: list[Path], fk: LiberoMujocoFK) -> tuple[np.ndarray, np.ndarray, int]:
    global_min = np.full(3, np.inf, dtype=np.float64)
    global_max = np.full(3, -np.inf, dtype=np.float64)
    total_frames = 0
    quat_norm_err_max = 0.0

    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path, columns=[JOINT_COLUMN, STATE_COLUMN])
        qpos = _read_frame_kinematics(df, pq_path)
        if qpos is None:
            continue

        kpts = fk.compute_batch(qpos)
        pos = kpts[:, :, :3].reshape(-1, 3)
        global_min = np.minimum(global_min, pos.min(axis=0))
        global_max = np.maximum(global_max, pos.max(axis=0))

        quat_norms = np.linalg.norm(kpts[:, :, 3:7].reshape(-1, 4), axis=1)
        quat_norm_err_max = max(quat_norm_err_max, float(np.abs(quat_norms - 1.0).max()))

        total_frames += len(df)
        logger.info("  Pass 1: %s — %d frames (total: %d)", pq_path.name, len(df), total_frames)

    if total_frames == 0:
        raise RuntimeError("Pass 1 found zero frames.")
    if quat_norm_err_max > 0.01:
        raise RuntimeError(f"Pass 1 quaternion norm error {quat_norm_err_max:.4f} > 0.01")
    logger.info("Pass 1 quaternion norm error max: %.6f", quat_norm_err_max)
    return global_min.astype(np.float32), global_max.astype(np.float32), total_frames


def pass2_write_keypoints(
    parquet_files: list[Path],
    fk: LiberoMujocoFK,
    r_pad: float,
    source_data_dir: Path,
    dest: Path,
) -> int:
    total_frames = 0
    oob_count = 0
    quat_err_count = 0

    for pq_path in parquet_files:
        rel = pq_path.relative_to(source_data_dir)
        dest_pq = dest / "data" / rel
        df = pd.read_parquet(pq_path)
        qpos = _read_frame_kinematics(df, dest_pq)
        if qpos is None:
            continue

        kpts = fk.compute_batch(qpos)
        kpts[:, :, :3] /= r_pad

        if (np.abs(kpts[:, :, :3]) > 1.01).any():
            oob_count += 1
            logger.warning(
                "  Position OOB in %s: max |pos| = %.4f",
                dest_pq.name,
                np.abs(kpts[:, :, :3]).max(),
            )

        quat_norms = np.linalg.norm(kpts[:, :, 3:7].reshape(-1, 4), axis=1)
        quat_err = float(np.abs(quat_norms - 1.0).max())
        if quat_err > 0.01:
            quat_err_count += 1
            logger.warning("  Quaternion norm error in %s: max = %.6f", dest_pq.name, quat_err)

        df["observation.keypoint_3d"] = [row.reshape(-1).astype(np.float32) for row in kpts]
        dest_pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest_pq, index=False)
        total_frames += len(df)
        logger.info("  Pass 2: %s — %d frames written", dest_pq.name, len(df))

    if oob_count:
        logger.warning("%d parquet files had OOB positions. Consider increasing --bbox-margin.", oob_count)
    if quat_err_count:
        logger.warning("%d parquet files had quaternion norm errors.", quat_err_count)
    return total_frames


def _keypoint_feature_names(body_names: list[str]) -> list[str]:
    return [
        f"{name}_{comp}"
        for name in body_names
        for comp in ("px", "py", "pz", "qx", "qy", "qz", "qw")
    ]


def _update_info_json(dest: Path, body_names: list[str]) -> None:
    info_path = dest / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["features"]["observation.keypoint_3d"] = {
        "dtype": "float32",
        "shape": [NUM_KEYPOINTS * KEYPOINT_DIM],
        "names": _keypoint_feature_names(body_names),
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info("Updated %s with observation.keypoint_3d [%d].", info_path, NUM_KEYPOINTS * KEYPOINT_DIM)


def _write_meta(
    dest: Path,
    r_pad: float,
    global_min: np.ndarray,
    global_max: np.ndarray,
    total_frames: int,
    xml_path: Path,
    bbox_margin: float,
    body_names: list[str],
) -> None:
    meta = {
        "bbox_radius": r_pad,
        "bbox_margin": bbox_margin,
        "global_min_base": global_min.tolist(),
        "global_max_base": global_max.tolist(),
        "normalization": "robot0_base_origin_isotropic_r_pad",
        "keypoint_dim": KEYPOINT_DIM,
        "keypoint_dim_layout": "px,py,pz,qx,qy,qz,qw",
        "rotation_representation": "quaternion_xyzw_hemisphere",
        "num_keypoints": NUM_KEYPOINTS,
        "keypoint_bodies": body_names,
        "base_body": BASE_BODY,
        "robot_nqpos": ROBOT_NQPOS,
        "qpos_layout": "joint_position[0:7] + observation.state[6:8]",
        "arm_joint_names": ARM_JOINT_NAMES,
        "gripper_joint_names": list(GRIPPER_JOINT_NAMES),
        "total_frames": total_frames,
        "mjcf_path": str(xml_path),
        "coordinate_system": "robot0_base-relative, divided by R_pad",
    }
    meta_path = dest / "meta" / "keypoints_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %s", meta_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Source LeRobot v3 dataset (with joint_position).")
    parser.add_argument("--dest", type=Path, required=True, help="Destination dataset copy with keypoints.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="MountedPanda MJCF from export_panda_mjcf.py.")
    parser.add_argument("--force", action="store_true", help="Overwrite --dest if it exists.")
    parser.add_argument("--skip-copy", action="store_true", help="Reuse existing --dest tree (partial rerun).")
    parser.add_argument("--bbox-margin", type=float, default=BBOX_MARGIN, help="R_pad safety margin.")
    parser.add_argument(
        "--r-pad",
        type=float,
        default=None,
        help="If set, use this R_pad (still runs Pass 1 for bbox meta). For shared multi-suite runs.",
    )
    args = parser.parse_args()

    if not args.xml.exists():
        raise FileNotFoundError(
            f"Panda MJCF not found at {args.xml}. Run:\n"
            f"  MUJOCO_GL=egl python util_scripts/export_panda_mjcf.py --output {args.xml}"
        )

    source = args.source.resolve()
    dest = args.dest.resolve()

    if not args.skip_copy:
        _copy_dataset(source, dest, args.force)
    elif not dest.exists():
        raise FileNotFoundError(f"--skip-copy but {dest} does not exist.")

    fk = LiberoMujocoFK(args.xml)
    source_parquets = _get_parquet_files(source / "data")
    logger.info("Found %d parquet files", len(source_parquets))
    logger.info("FK bodies (robot0_base-relative): %s", fk.body_names)

    logger.info("=== Pass 1: computing global bounding box (base frame) ===")
    global_min, global_max, total_p1 = pass1_compute_bbox(source_parquets, fk)
    if args.r_pad is not None:
        if args.r_pad <= 0:
            raise ValueError(f"--r-pad must be > 0, got {args.r_pad}")
        r_pad = float(args.r_pad)
        logger.info("Using forced --r-pad=%.6f (Pass-1 bbox kept for meta only)", r_pad)
    else:
        r_pad = compute_r_pad(global_min, global_max, margin=args.bbox_margin)
    logger.info("Global min (base): %s", global_min)
    logger.info("Global max (base): %s", global_max)
    logger.info("R_pad = %.6f m (margin=%.0f%%)", r_pad, args.bbox_margin * 100)

    logger.info("=== Pass 2: writing 7D keypoints ===")
    total_p2 = pass2_write_keypoints(source_parquets, fk, r_pad, source / "data", dest)

    _update_info_json(dest, fk.body_names)
    _write_meta(dest, r_pad, global_min, global_max, total_p2, args.xml.resolve(), args.bbox_margin, fk.body_names)

    logger.info("=== DONE ===")
    logger.info("  Frames: %d (pass1) / %d (pass2)", total_p1, total_p2)
    logger.info("  Output: %s", dest)
    logger.info("  keypoint_3d dim: %d (= %d keypoints x %d)", NUM_KEYPOINTS * KEYPOINT_DIM, NUM_KEYPOINTS, KEYPOINT_DIM)


if __name__ == "__main__":
    main()
