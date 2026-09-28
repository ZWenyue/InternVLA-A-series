#!/usr/bin/env python3
"""Export MountedPanda / Panda+RethinkMount MJCF for LIBERO GeoP FK keypoints.

Composes the robot MJCF via robosuite model APIs (no ``robosuite.make``), so it
works with newer MuJoCo that breaks ``env.reset()`` joint-address asserts.

Usage:
    env -i HOME="$HOME" PATH="$PATH" \\
      LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \\
      __EGL_VENDOR_LIBRARY_DIRS="$HOME/.local/share/glvnd/egl_vendor.d" \\
      MUJOCO_GL=egl PYTHONPATH=/path/to/LIBERO-plus \\
      python util_scripts/export_panda_mjcf.py --output /tmp/DATA/panda_libero_plus.xml

Requires: robosuite (1.4.x recommended), mujoco
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_OUTPUT = Path("/tmp/DATA/panda_libero_plus.xml")
DEFAULT_LIBERO_PLUS = Path("/home/a26160/SRC/LIBERO-plus")

EEF_BODY_CANDIDATES = ("gripper0_eef", "gripper0_right_eef")

KEYPOINT_BODIES = [
    "robot0_link1",
    "robot0_link2",
    "robot0_link3",
    "robot0_link4",
    "robot0_link5",
    "robot0_link6",
    "robot0_link7",
]

ARM_JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]
GRIPPER_JOINTS = ("gripper0_finger_joint1", "gripper0_finger_joint2")
BASE_BODY = "robot0_base"


def _ensure_libero_plus_on_path(libero_plus: Path) -> None:
    root = str(libero_plus.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_eef_body_name(model) -> str:
    for name in EEF_BODY_CANDIDATES:
        try:
            model.body(name)
            return name
        except KeyError:
            continue
    raise KeyError(f"No EEF body found (tried {', '.join(EEF_BODY_CANDIDATES)})")


def _load_mounted_panda_class(libero_plus: Path):
    """Load MountedPanda without importing libero.envs (avoids bddl/easydict)."""
    import importlib.util

    path = libero_plus / "libero" / "libero" / "envs" / "robots" / "mounted_panda.py"
    if not path.is_file():
        raise FileNotFoundError(f"MountedPanda source not found: {path}")
    spec = importlib.util.spec_from_file_location("_libero_mounted_panda", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MountedPanda from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MountedPanda


def _build_robot(libero_plus: Path):
    """Build Panda + RethinkMount + PandaGripper without creating a sim."""
    from robosuite.models.grippers import gripper_factory
    from robosuite.models.mounts import mount_factory

    robot = None
    robot_name = "Panda"
    try:
        from robosuite.models.robots.robot_model import register_robot

        MountedPanda = _load_mounted_panda_class(libero_plus)
        register_robot(MountedPanda)
        robot = MountedPanda(idn=0)
        robot_name = "MountedPanda"
    except Exception as exc:
        print(f"MountedPanda unavailable ({exc}); using robosuite Panda")
        from robosuite.models.robots import Panda

        robot = Panda(idn=0)
        robot_name = "Panda"

    mount_name = getattr(robot, "default_mount", None) or getattr(robot, "default_base", "RethinkMount")
    if callable(mount_name):
        mount_name = mount_name()
    if isinstance(mount_name, dict):
        mount_name = next(iter(mount_name.values()))
    robot.add_mount(mount_factory(mount_name))

    grip_name = robot.default_gripper
    if callable(grip_name):
        grip_name = grip_name()
    if isinstance(grip_name, dict):
        grip_name = next(iter(grip_name.values()))
    robot.add_gripper(gripper_factory(grip_name))
    return robot, robot_name


def export_panda_mjcf(output_path: Path, libero_plus: Path) -> Path:
    os.environ.setdefault("MUJOCO_GL", "egl")
    _ensure_libero_plus_on_path(libero_plus)

    import mujoco
    from robosuite.models.arenas import EmptyArena
    from robosuite.models.tasks import ManipulationTask

    robot, robot_name = _build_robot(libero_plus)
    task = ManipulationTask(
        mujoco_arena=EmptyArena(),
        mujoco_robots=[robot],
        mujoco_objects=[],
    )
    model_xml = task.get_xml()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(model_xml)

    model = mujoco.MjModel.from_xml_path(str(output_path))
    print(f"Exported {robot_name} MJCF to {output_path}")
    print(f"  nq={model.nq}, nv={model.nv}, nbody={model.nbody}")
    print(f"  base body: {BASE_BODY} id={model.body(BASE_BODY).id}")

    eef_name = resolve_eef_body_name(model)
    for name in [*KEYPOINT_BODIES, eef_name]:
        bid = model.body(name).id
        print(f"  {name}: body_id={bid}")

    for jname in [*ARM_JOINTS, *GRIPPER_JOINTS]:
        jid = model.joint(jname).id
        qadr = int(model.jnt_qposadr[jid])
        print(f"  {jname}: joint_id={jid} qposadr={qadr}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output MJCF path.")
    parser.add_argument(
        "--libero-plus",
        type=Path,
        default=Path(os.environ.get("LIBERO_PLUS_ROOT", DEFAULT_LIBERO_PLUS)),
        help="Path to LIBERO-plus repo root (contains libero/).",
    )
    args = parser.parse_args()
    if not args.libero_plus.is_dir():
        raise FileNotFoundError(f"LIBERO-plus root not found: {args.libero_plus}")
    export_panda_mjcf(args.output, args.libero_plus)


if __name__ == "__main__":
    main()
