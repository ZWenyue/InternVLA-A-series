#!/usr/bin/env python3
"""Run N short LIBERO-plus episodes and print per-module wall-clock timing.

Designed to sit next to ``run_geop_full.sh``: start a policy server, then::

    RENDER_BACKEND=egl python evaluation/LIBERO-plus2/bench_module_timing.py \\
        --host 127.0.0.1 --port 5800 --num_runs 5

Modules timed (per episode, then aggregated):
  env_create     OffScreenRenderEnv construction (run 0 only, or once recycled)
  env_reset      env.reset + set_init_state
  wait_sim       dummy wait steps (env.step only)
  keypoint       StandaloneFK / KeypointHistory.push
  preprocess     image rotate + state extract + kpt pack (client-side, before RPC)
  infer_rpc      websocket round-trip (server preprocess + model.sample_actions)
  env_step       policy-driven MuJoCo step (sim + render)
  episode_wall   full episode wall time
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_compat

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LOGGER = logging.getLogger("bench_module_timing")

TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


@dataclass
class Bucket:
    samples: list[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        self.samples.append(float(seconds))

    def summary(self) -> dict[str, float | int]:
        xs = self.samples
        if not xs:
            return {"n": 0, "sum_s": 0.0, "mean_s": 0.0, "std_s": 0.0, "min_s": 0.0, "max_s": 0.0}
        return {
            "n": len(xs),
            "sum_s": round(sum(xs), 4),
            "mean_s": round(statistics.mean(xs), 4),
            "std_s": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
            "min_s": round(min(xs), 4),
            "max_s": round(max(xs), 4),
        }


@dataclass
class EpisodeTimers:
    env_reset: Bucket = field(default_factory=Bucket)
    wait_sim: Bucket = field(default_factory=Bucket)
    keypoint: Bucket = field(default_factory=Bucket)
    preprocess: Bucket = field(default_factory=Bucket)
    infer_rpc: Bucket = field(default_factory=Bucket)
    env_step: Bucket = field(default_factory=Bucket)
    episode_wall: Bucket = field(default_factory=Bucket)
    n_policy_steps: int = 0
    n_infers: int = 0
    success: bool = False


def _get_libero_env(task, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def _install_timing_hooks(client, timers: EpisodeTimers) -> None:
    """Split ``_request_chunk`` into preprocess vs websocket RPC timing."""
    orig_request = client._request_chunk

    def timed_request_chunk(obs: dict[str, Any], lang: str) -> np.ndarray:
        t0 = time.perf_counter()
        primary = client._maybe_rotate(np.asarray(obs["agentview_image"], dtype=np.uint8))
        wrist = client._maybe_rotate(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8))
        state = client._extract_state(obs)
        example: dict[str, Any] = {
            "image": [primary, wrist],
            "lang": lang,
            "task": lang,
            "state": state,
        }
        if client.enable_keypoints and client._kpt_history is not None:
            kpt_history, his_len = client._kpt_history.get_history()
            if his_len > 0:
                example["kpt_history"] = kpt_history
                example["his_len"] = his_len
        payload = {"examples": [example], "do_sample": False}
        t_prep = time.perf_counter()
        timers.preprocess.add(t_prep - t0)

        t_rpc0 = time.perf_counter()
        response = client.client.predict_action(
            {
                "type": "infer",
                "request_id": f"libero-{client._step}",
                "payload": payload,
            }
        )
        timers.infer_rpc.add(time.perf_counter() - t_rpc0)
        timers.n_infers += 1

        if not response.get("ok", False):
            raise RuntimeError(f"Inference server error: {response}")
        data = response.get("data", {})
        if "actions" not in data:
            raise KeyError(
                f"Server response missing 'data.actions'. "
                f"protocol={response.get('protocol_version')} keys={list(data.keys())}"
            )
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 3:
            raise ValueError(f"Expected actions with ndim=3, got shape={actions.shape}")
        client._update_action_space(data)
        chunk = np.array(actions[0], dtype=np.float32, copy=True)
        if client._action_low is not None and client._action_high is not None:
            dim = min(chunk.shape[-1], client._action_low.shape[-1], client._action_high.shape[-1])
            chunk[:, :dim] = np.clip(chunk[:, :dim], client._action_low[:dim], client._action_high[:dim])
        return chunk.astype(np.float32)

    # Keep a reference so pylint/readers know we replaced the hot path intentionally.
    _ = orig_request
    client._request_chunk = timed_request_chunk  # type: ignore[method-assign]

    orig_push = client.push_keypoint

    def timed_push_keypoint() -> None:
        t0 = time.perf_counter()
        orig_push()
        timers.keypoint.add(time.perf_counter() - t0)

    client.push_keypoint = timed_push_keypoint  # type: ignore[method-assign]


def _run_one_episode(
    env,
    client,
    task_description: str,
    init_state,
    *,
    max_steps: int,
    num_steps_wait: int,
) -> EpisodeTimers:
    timers = EpisodeTimers()
    _install_timing_hooks(client, timers)

    t_ep0 = time.perf_counter()
    client.reset(task_description)

    t0 = time.perf_counter()
    env.reset()
    obs = env.set_init_state(init_state)
    timers.env_reset.add(time.perf_counter() - t0)

    done = False
    for t in range(max_steps + num_steps_wait):
        if t < num_steps_wait:
            client.push_keypoint()
            t0 = time.perf_counter()
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            timers.wait_sim.add(time.perf_counter() - t0)
            continue

        action = client.step(obs, task_description)
        timers.n_policy_steps += 1
        client.push_keypoint()
        t0 = time.perf_counter()
        obs, _, done, _ = env.step(action.tolist())
        timers.env_step.add(time.perf_counter() - t0)
        if done:
            break

    timers.episode_wall.add(time.perf_counter() - t_ep0)
    timers.success = bool(done)
    return timers


def _fmt_row(name: str, s: dict[str, float | int], episode_wall_sum: float) -> str:
    pct = (100.0 * float(s["sum_s"]) / episode_wall_sum) if episode_wall_sum > 0 else 0.0
    return (
        f"  {name:<12}  n={s['n']:>5}  "
        f"sum={s['sum_s']:>8.3f}s  mean={s['mean_s']:>7.4f}s  "
        f"std={s['std_s']:>7.4f}s  min={s['min_s']:>7.4f}s  max={s['max_s']:>7.4f}s  "
        f"({pct:5.1f}% of ep wall)"
    )


def _merge_buckets(episodes: list[EpisodeTimers]) -> dict[str, Bucket]:
    merged = {
        "env_reset": Bucket(),
        "wait_sim": Bucket(),
        "keypoint": Bucket(),
        "preprocess": Bucket(),
        "infer_rpc": Bucket(),
        "env_step": Bucket(),
        "episode_wall": Bucket(),
    }
    for ep in episodes:
        for name, bucket in merged.items():
            bucket.samples.extend(getattr(ep, name).samples)
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5800)
    p.add_argument("--num_runs", type=int, default=5, help="Number of episodes to time.")
    p.add_argument("--task_suite_name", type=str, default="libero_spatial", choices=list(TASK_SUITE_MAX_STEPS))
    p.add_argument("--task_id", type=int, default=0, help="Task index inside the suite.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--replan_steps", type=int, default=8)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument(
        "--max_steps",
        type=int,
        default=80,
        help="Cap episode length for faster bench (default 80). Use -1 for suite default.",
    )
    p.add_argument("--gripper_convention", type=str, default="libero_native")
    p.add_argument("--enable_keypoints", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--kpt_r_pad",
        type=float,
        default=0.9613106459379196,
        help="Keypoint position scale. Must match training keypoints_meta.json bbox_radius.",
    )
    p.add_argument("--rotate_images", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--json_out", type=str, default="", help="Optional path to write timing JSON.")
    p.add_argument("--reuse_env", action=argparse.BooleanOptionalAction, default=True,
                   help="Reuse one OffScreenRenderEnv across runs (default).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
    )
    np.random.seed(args.seed)

    from libero.libero import benchmark

    from evaluation.LIBERO2.model2libero_interface import LiberoModelClient

    max_steps = TASK_SUITE_MAX_STEPS[args.task_suite_name] if args.max_steps < 0 else args.max_steps
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = task_suite.n_tasks
    if not (0 <= args.task_id < n_tasks):
        raise SystemExit(f"task_id={args.task_id} out of range [0, {n_tasks})")

    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    init_state = initial_states[0]

    LOGGER.info(
        "Connecting to policy server %s:%d | suite=%s task_id=%d | runs=%d max_steps=%d replan=%d kpt=%s",
        args.host,
        args.port,
        args.task_suite_name,
        args.task_id,
        args.num_runs,
        max_steps,
        args.replan_steps,
        args.enable_keypoints,
    )

    t_conn0 = time.perf_counter()
    client = LiberoModelClient(
        host=args.host,
        port=args.port,
        rotate_images=args.rotate_images,
        gripper_convention=args.gripper_convention,
        replan_steps=args.replan_steps,
        enable_keypoints=args.enable_keypoints,
        kpt_r_pad=args.kpt_r_pad,
    )
    connect_s = time.perf_counter() - t_conn0
    LOGGER.info("Client connected in %.2fs | metadata keys=%s", connect_s, sorted(client.metadata.keys()))

    t_create0 = time.perf_counter()
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    env_create_s = time.perf_counter() - t_create0
    client.bind_env(env)
    LOGGER.info("Env created in %.2fs | task=%r", env_create_s, task_description)

    episodes: list[EpisodeTimers] = []
    for run_idx in range(args.num_runs):
        if not args.reuse_env and run_idx > 0:
            t_create0 = time.perf_counter()
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed + run_idx)
            env_create_s = time.perf_counter() - t_create0
            client.bind_env(env)
            LOGGER.info("Env re-created for run %d in %.2fs", run_idx, env_create_s)

        LOGGER.info("=== run %d/%d ===", run_idx + 1, args.num_runs)
        ep = _run_one_episode(
            env,
            client,
            task_description,
            init_state,
            max_steps=max_steps,
            num_steps_wait=args.num_steps_wait,
        )
        episodes.append(ep)
        LOGGER.info(
            "run %d done: wall=%.2fs success=%s policy_steps=%d infers=%d "
            "infer_rpc_mean=%.3fs env_step_mean=%.3fs",
            run_idx + 1,
            ep.episode_wall.samples[0],
            ep.success,
            ep.n_policy_steps,
            ep.n_infers,
            statistics.mean(ep.infer_rpc.samples) if ep.infer_rpc.samples else 0.0,
            statistics.mean(ep.env_step.samples) if ep.env_step.samples else 0.0,
        )

    merged = _merge_buckets(episodes)
    wall_sum = float(merged["episode_wall"].summary()["sum_s"])
    successes = sum(1 for ep in episodes if ep.success)

    print("\n" + "=" * 96)
    print(
        f"LIBERO-plus2 module timing | suite={args.task_suite_name} task_id={args.task_id} "
        f"| runs={args.num_runs} max_steps={max_steps} replan={args.replan_steps} "
        f"| RENDER_BACKEND={os.environ.get('RENDER_BACKEND', '?')} "
        f"| success={successes}/{args.num_runs}"
    )
    print(f"  connect_client  {connect_s:.3f}s (once)")
    print(f"  env_create      {env_create_s:.3f}s (first env)")
    print("-" * 96)
    for name in ("env_reset", "wait_sim", "keypoint", "preprocess", "infer_rpc", "env_step", "episode_wall"):
        print(_fmt_row(name, merged[name].summary(), wall_sum))
    print("-" * 96)

    total_policy_steps = sum(ep.n_policy_steps for ep in episodes)
    total_infers = sum(ep.n_infers for ep in episodes)
    infer_sum = float(merged["infer_rpc"].summary()["sum_s"])
    env_sum = float(merged["env_step"].summary()["sum_s"])
    print(f"  total_policy_steps={total_policy_steps}  total_infers={total_infers}")
    if total_infers > 0 and total_policy_steps > 0:
        print(
            f"  infer_rpc share of (infer+env_step) = "
            f"{100.0 * infer_sum / max(infer_sum + env_sum, 1e-9):.1f}%  |  "
            f"steps_per_infer ≈ {total_policy_steps / total_infers:.2f}"
        )
    print("=" * 96 + "\n")

    payload = {
        "config": {
            "host": args.host,
            "port": args.port,
            "num_runs": args.num_runs,
            "task_suite_name": args.task_suite_name,
            "task_id": args.task_id,
            "max_steps": max_steps,
            "replan_steps": args.replan_steps,
            "num_steps_wait": args.num_steps_wait,
            "enable_keypoints": args.enable_keypoints,
            "rotate_images": args.rotate_images,
            "render_backend": os.environ.get("RENDER_BACKEND"),
            "reuse_env": args.reuse_env,
        },
        "connect_client_s": round(connect_s, 4),
        "env_create_s": round(env_create_s, 4),
        "successes": successes,
        "modules": {name: bucket.summary() for name, bucket in merged.items()},
        "per_episode": [
            {
                "run": i,
                "success": ep.success,
                "n_policy_steps": ep.n_policy_steps,
                "n_infers": ep.n_infers,
                "episode_wall_s": ep.episode_wall.samples[0] if ep.episode_wall.samples else None,
                "infer_rpc_mean_s": (
                    round(statistics.mean(ep.infer_rpc.samples), 4) if ep.infer_rpc.samples else None
                ),
                "env_step_mean_s": (
                    round(statistics.mean(ep.env_step.samples), 4) if ep.env_step.samples else None
                ),
            }
            for i, ep in enumerate(episodes)
        ],
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Wrote timing JSON → %s", out)


if __name__ == "__main__":
    from evaluation.LIBERO2.render_backend import describe, egl_available, setup_render_env

    _client_venv = os.environ.get("CLIENT_VENV")
    _backend = setup_render_env(
        os.environ.get("RENDER_BACKEND"),
        device=os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"),
        client_venv=_client_venv,
    )
    _info = describe(_backend)
    _info["egl_available"] = egl_available(_client_venv)
    _info["egl_vendor_dirs"] = os.environ.get("__EGL_VENDOR_LIBRARY_DIRS")
    print(f"[render] backend={_backend} {_info}", file=sys.stderr)
    main()
