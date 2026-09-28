#!/usr/bin/env python3
"""Build 20 Hz GeoP keypoints for LIBERO-plus ``libero_mix`` (single LeRobot dataset).

Source (from ``port_libero_plus.py``)::

    <source>/   # e.g. /tmp/DATA/libero_plus_mix_lerobot
      data/
      meta/
      videos/
        observation.images.image
        observation.images.wrist_image
        observation.state              (8,)
        observation.state.joint_position (7,)
        action

Writes::

    <output>/   # e.g. /tmp/DATA/libero_plus_geop
      ...same tree + observation.keypoint_3d (K*7, robot0_base / R_pad)

Does **not** rename ``observation.images.wrist_image``. Same FK / R_pad contract
as ``generate_libero_geop_from_rlds.py`` (shared helpers in generate_libero_keypoints).

Usage:
    MUJOCO_GL=egl python util_scripts/generate_libero_plus_geop.py \\
        --source /tmp/DATA/libero_plus_mix_lerobot \\
        --output /tmp/DATA/libero_plus_geop \\
        --xml /tmp/DATA/panda_libero_plus.xml \\
        --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_libero_keypoints import (  # noqa: E402
    BBOX_MARGIN,
    DEFAULT_XML,
    LiberoMujocoFK,
    _copy_dataset,
    _get_parquet_files,
    _update_info_json,
    _write_meta,
    compute_r_pad,
    pass1_compute_bbox,
    pass2_write_keypoints,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SOURCE = Path("/tmp/DATA/libero_plus_mix_lerobot")
DEFAULT_OUTPUT = Path("/tmp/DATA/libero_plus_geop")
REQUIRED_COLUMNS = (
    "observation.state",
    "observation.state.joint_position",
)


def _validate_source(source: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path} (need LeRobot v3 layout)")
    with open(info_path) as f:
        info = json.load(f)
    features = info.get("features", {})
    missing = [c for c in REQUIRED_COLUMNS if c not in features]
    if missing:
        raise KeyError(
            f"Source {source} missing features required for FK: {missing}. "
            "Re-convert with port_libero_plus.py (must include joint_state)."
        )
    logger.info(
        "Source OK: %s episodes=%s frames=%s fps=%s",
        source,
        info.get("total_episodes"),
        info.get("total_frames"),
        info.get("fps"),
    )


def _patch_robot_type(dataset_root: Path, robot_type: str) -> None:
    info_path = dataset_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    if info.get("robot_type") != robot_type:
        info["robot_type"] = robot_type
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        logger.info("Patched robot_type=%s in %s", robot_type, info_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="LeRobot v3 LIBERO-plus mix dataset (from port_libero_plus.py).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination dataset root with observation.keypoint_3d.",
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="MountedPanda MJCF path.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing --output.")
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Reuse existing --output tree (only rewrite keypoints).",
    )
    parser.add_argument("--bbox-margin", type=float, default=BBOX_MARGIN)
    parser.add_argument(
        "--robot-type",
        type=str,
        default="libero_plus",
        help="robot_type written into meta/info.json.",
    )
    parser.add_argument(
        "--r-pad",
        type=float,
        default=None,
        help="If set, force this R_pad (Pass 1 bbox still recorded in meta).",
    )
    args = parser.parse_args()

    if not args.xml.exists():
        raise FileNotFoundError(
            f"MJCF not found: {args.xml}. Export first with export_panda_mjcf.py"
        )

    source = args.source.resolve()
    output = args.output.resolve()
    _validate_source(source)

    if args.skip_copy:
        if not output.is_dir():
            raise FileNotFoundError(f"--skip-copy but missing {output}")
        logger.info("Reuse existing %s", output)
    else:
        _copy_dataset(source, output, force=args.force)

    _patch_robot_type(output, args.robot_type)

    fk = LiberoMujocoFK(args.xml)
    logger.info("FK bodies (robot0_base-relative): %s", fk.body_names)

    # After copy, FK + write in place on the output tree (same as geop_from_rlds suites).
    parquets = _get_parquet_files(output / "data")
    logger.info("Found %d parquet file(s)", len(parquets))

    logger.info("=== Pass 1: global bbox ===")
    global_min, global_max, total_p1 = pass1_compute_bbox(parquets, fk)
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

    logger.info("=== Pass 2: write observation.keypoint_3d ===")
    total_p2 = pass2_write_keypoints(parquets, fk, r_pad, output / "data", output)
    _update_info_json(output, fk.body_names)
    _write_meta(
        output,
        r_pad,
        global_min,
        global_max,
        total_p2,
        args.xml.resolve(),
        args.bbox_margin,
        fk.body_names,
    )

    meta_path = output / "meta" / "keypoints_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["source"] = str(source)
    meta["fps_note"] = "20 Hz from libero_plus_mix LeRobot; camera key wrist_image kept"
    meta["r_pad_scope"] = "libero_plus_mix_single_dataset"
    meta["dataset_family"] = "libero_plus"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("=== DONE ===")
    logger.info("  Frames: %d (pass1) / %d (pass2)", total_p1, total_p2)
    logger.info("  Shared R_pad: %.6f m", r_pad)
    logger.info("  Output: %s", output)


if __name__ == "__main__":
    main()
