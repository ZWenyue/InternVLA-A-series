#!/usr/bin/env python3
"""Build 20 Hz LIBERO GeoP suites from libero_rlds_lerobot (with joints).

Source layout (already split, OpenVLA/IPEC camera names kept as-is)::

    <source-root>/
      libero_spatial_no_noops_1.0.0_lerobot/
      libero_object_no_noops_1.0.0_lerobot/
      libero_goal_no_noops_1.0.0_lerobot/
      libero_10_no_noops_1.0.0_lerobot/

Writes::

    <output-dir>/libero_{spatial,object,goal,10}/

with shared robot0_base-relative ``observation.keypoint_3d`` and one global R_pad.

Does **not** rename ``observation.images.wrist_image``.

Usage:
    MUJOCO_GL=egl python util_scripts/generate_libero_geop_from_rlds.py \\
        --source-root /tmp/DATA/libero_rlds_lerobot \\
        --output-dir  /tmp/DATA/libero_geop_rlds20 \\
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

# short_name -> source directory under --source-root
SUITE_SOURCES = {
    "libero_spatial": "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object": "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal": "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10": "libero_10_no_noops_1.0.0_lerobot",
}


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
        "--source-root",
        type=Path,
        default=Path("/tmp/DATA/libero_rlds_lerobot"),
        help="Directory containing the four *_no_noops_1.0.0_lerobot suites.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/DATA/libero_geop_rlds20"),
        help="Where to write libero_{spatial,object,goal,10}.",
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="MountedPanda MJCF path.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing suite outputs.")
    parser.add_argument("--bbox-margin", type=float, default=BBOX_MARGIN)
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Reuse existing suite dirs under --output-dir (only rewrite keypoints).",
    )
    args = parser.parse_args()

    if not args.xml.exists():
        raise FileNotFoundError(
            f"MJCF not found: {args.xml}. Export first with export_panda_mjcf.py"
        )

    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suite_pairs: list[tuple[str, Path, Path]] = []
    for short, src_name in SUITE_SOURCES.items():
        src = source_root / src_name
        dst = output_dir / short
        if not src.is_dir():
            raise FileNotFoundError(f"Missing source suite: {src}")
        suite_pairs.append((short, src, dst))

    # ── Copy suites (keep wrist_image) ────────────────────────────────────
    for short, src, dst in suite_pairs:
        if args.skip_copy:
            if not dst.is_dir():
                raise FileNotFoundError(f"--skip-copy but missing {dst}")
            logger.info("Reuse existing %s", dst)
        else:
            _copy_dataset(src, dst, force=args.force)
        _patch_robot_type(dst, short)

    # ── Shared Pass 1 over all suites ─────────────────────────────────────
    fk = LiberoMujocoFK(args.xml)
    logger.info("FK bodies (robot0_base-relative): %s", fk.body_names)

    all_parquets: list[Path] = []
    for short, _src, dst in suite_pairs:
        pqs = _get_parquet_files(dst / "data")
        logger.info("  %s: %d parquet file(s)", short, len(pqs))
        all_parquets.extend(pqs)

    logger.info("=== Pass 1 (global across %d suites) ===", len(suite_pairs))
    global_min, global_max, total_p1 = pass1_compute_bbox(all_parquets, fk)
    r_pad = compute_r_pad(global_min, global_max, margin=args.bbox_margin)
    logger.info("Global min (base): %s", global_min)
    logger.info("Global max (base): %s", global_max)
    logger.info("Shared R_pad = %.6f m (margin=%.0f%%)", r_pad, args.bbox_margin * 100)

    # ── Pass 2 per suite ──────────────────────────────────────────────────
    total_p2 = 0
    for short, _src, dst in suite_pairs:
        logger.info("=== Pass 2: %s ===", short)
        pqs = _get_parquet_files(dst / "data")
        # Source and dest data dirs are the same tree after copy — write in place.
        # pass2 expects source_data_dir for relative paths and dest as dataset root.
        n = pass2_write_keypoints(pqs, fk, r_pad, dst / "data", dst)
        total_p2 += n
        _update_info_json(dst, fk.body_names)
        _write_meta(
            dst,
            r_pad,
            global_min,
            global_max,
            n,
            args.xml.resolve(),
            args.bbox_margin,
            fk.body_names,
        )
        # Annotate shared-norm provenance
        meta_path = dst / "meta" / "keypoints_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["source_root"] = str(source_root)
        meta["fps_note"] = "20 Hz from libero_rlds_lerobot; camera key wrist_image kept"
        meta["r_pad_scope"] = "shared_across_libero_spatial_object_goal_10"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    logger.info("=== DONE ===")
    logger.info("  Frames: %d (pass1) / %d (pass2)", total_p1, total_p2)
    logger.info("  Shared R_pad: %.6f m", r_pad)
    logger.info("  Output: %s/libero_{spatial,object,goal,10}", output_dir)


if __name__ == "__main__":
    main()
