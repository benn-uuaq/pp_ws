"""
motion_tool.py - command-line tool for the calibration moves only, without a camera.

  axes  print the axis mapping for camera_side (does not connect to the robot)
  plan  print the grid around the current pose (does not move)
  step  move once along one camera axis       e.g. step depth 20
  run   visit the whole grid and return to the start pose (no measurement, checks reachability)

The move axes are the camera axes lateral/depth/vertical, not the robot base X/Y/Z.
Their mapping to base axes depends on camera_side (see motion_frame.py).
With a Helios camera attached, run can be extended to measure the marker at every pose.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from robot_camera_calibration import motion_frame as mf
from robot_camera_calibration import pose_math as pm
from robot_camera_calibration.motion_frame import format_base_mm, format_view_mm
from robot_camera_calibration.settings import default_config_path, load_config, step_m


# ======================================================================
# Output format
# ======================================================================
def axes_table(side: str) -> List[str]:
    axes = mf.view_axes(side)
    lines = [f"camera_side {side}: the camera is on the {side.upper()} side of the robot base",
             "  camera axis      base axis   meaning"]
    meaning = {"lateral": "right as seen by the camera", "depth": "towards the camera", "vertical": "up"}
    for i, name in enumerate(mf.VIEW_AXES):
        lines.append(f"  {name:<14}   {mf.base_axis_label(axes[:, i]):<9}   {meaning[name]}")
    return lines


# ======================================================================
# Robot motion
# ======================================================================
def ask_yes_no(text: str) -> bool:
    try:
        answer = input(f"{text} [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def wait_for_state(robot, timeout_s: float = 5.0):
    end = time.time() + timeout_s
    while time.time() < end:
        st = robot.get_state()
        if st is not None:
            return st
        time.sleep(0.05)
    return None


def current_pose(robot, out) -> Optional[List[float]]:
    st = wait_for_state(robot)
    if st is None:
        out(f"No robot state received: {robot.last_error or 'no response'}")
        return None
    if st.emergency_stopped or st.protective_stopped:
        out("The robot is in emergency / protective stop.")
        return None
    return list(st.tcp_pose)


def move_to(cfg: dict, robot, target: Sequence[float]) -> Tuple[bool, str]:
    rc = cfg["robot"]
    if not robot.move_linear(target, float(rc["acceleration"]), float(rc["velocity"])):
        return False, f"Failed to send the move command: {robot.last_error}"
    return robot.wait_until_reached(target, float(rc["position_tolerance_mm"]) / 1000.0,
                                    float(rc["move_timeout_s"]))


def _log_alarms(robot, out) -> None:
    for a in robot.pop_alarms():
        out(f"  [ROBOT ALARM] {a}")


# ======================================================================
# Commands
# ======================================================================
def cmd_axes(cfg: dict, out=print) -> int:
    for line in axes_table(cfg["sequence"]["camera_side"]):
        out(line)
    return 0


def cmd_plan(cfg: dict, robot, out=print) -> int:
    sc = cfg["sequence"]
    side = sc["camera_side"]
    start = current_pose(robot, out)
    if start is None:
        return 1
    for line in axes_table(side):
        out(line)
    out(f"Start pose: {pm.format_pose(start)}")
    offsets = mf.grid_offsets_view(step_m(cfg), int(sc["points_per_axis"]))
    poses = [mf.offset_pose(start, side, o) for o in offsets]
    out(f"Grid of {len(poses)} points (camera-axis spacing {format_view_mm(step_m(cfg))})")
    for i, (o, p) in enumerate(zip(offsets, poses), start=1):
        out(f"  [{i:2d}] {format_view_mm(o)}  ->  base {format_base_mm(np.subtract(p[:3], start[:3]))}")
    xyz = np.array([p[:3] for p in poses])
    out(f"Base range: X {xyz[:, 0].min() * 1000:.1f} ~ {xyz[:, 0].max() * 1000:.1f}, "
        f"Y {xyz[:, 1].min() * 1000:.1f} ~ {xyz[:, 1].max() * 1000:.1f}, "
        f"Z {xyz[:, 2].min() * 1000:.1f} ~ {xyz[:, 2].max() * 1000:.1f} mm")
    return 0


def cmd_step(cfg: dict, robot, axis: str, distance_mm: float,
             confirm: Callable[[str], bool] = ask_yes_no, out=print) -> int:
    sc = cfg["sequence"]
    side = sc["camera_side"]
    mf.axis_index(axis)
    limit = float(sc["max_single_step_mm"])
    if abs(distance_mm) > limit:
        out(f"Cannot move more than {limit:g} mm at once (sequence.max_single_step_mm): {distance_mm:g}")
        return 1
    start = current_pose(robot, out)
    if start is None:
        return 1
    target = mf.step_pose(start, side, axis, distance_mm / 1000.0)
    delta = np.subtract(target[:3], start[:3])
    out(f"camera_side {side}: {axis} {distance_mm:+g} mm  ->  base {format_base_mm(delta)}")
    out(f"Current: {pm.format_pose(start)}")
    out(f"Target:  {pm.format_pose(target)}")
    if not confirm(f"Moving at {float(cfg['robot']['velocity']) * 1000:.0f} mm/s. Proceed?"):
        out("Cancelled.")
        return 1
    ok, why = move_to(cfg, robot, target)
    _log_alarms(robot, out)
    end = current_pose(robot, out)
    if end is not None:
        moved = np.subtract(end[:3], start[:3])
        out(f"Actual move: {format_view_mm(mf.base_to_view(side, moved))}  (base {format_base_mm(moved)})")
    if not ok:
        out(f"Did not reach the target: {why}")
        return 1
    return 0


def cmd_run(cfg: dict, robot, confirm: Callable[[str], bool] = ask_yes_no, out=print) -> int:
    rc, sc = cfg["robot"], cfg["sequence"]
    side = sc["camera_side"]
    start = current_pose(robot, out)
    if start is None:
        return 1
    offsets = mf.grid_offsets_view(step_m(cfg), int(sc["points_per_axis"]))
    steps = sc["step_mm"]
    for line in axes_table(side):
        out(line)
    out(f"Start pose: {pm.format_pose(start)}")
    if not confirm(f"Visiting {len(offsets)} points around the current pose (lateral ±{steps['lateral']:g} / "
                   f"depth ±{steps['depth']:g} / vertical ±{steps['vertical']:g} mm, "
                   f"{float(rc['velocity']) * 1000:.0f} mm/s). Clear the area and keep a hand on the emergency stop. "
                   "Proceed?"):
        out("Cancelled.")
        return 1

    reached, skipped = 0, []
    for i, o in enumerate(offsets, start=1):
        target = mf.offset_pose(start, side, o)
        out(f"[{i:2d}/{len(offsets)}] {format_view_mm(o)}")
        ok, why = move_to(cfg, robot, target)
        _log_alarms(robot, out)
        if not ok:
            if "STOP" in why:
                out(f"Robot stopped: {why}")
                return 1
            out(f"  [WARN] {why} -> skipped")
            skipped.append(i)
            continue
        reached += 1

    if sc["return_to_start"]:
        out("Returning to the start pose.")
        ok, why = move_to(cfg, robot, start)
        if not ok:
            out(f"  [WARN] failed to return to the start pose: {why}")
    out(f"Done: {reached}/{len(offsets)} points reached" + (f", skipped {skipped}" if skipped else ""))
    return 0 if not skipped else 1


# ======================================================================
# Entry point
# ======================================================================
def make_robot(cfg: dict, sim: bool, sim_start_pose: Optional[Sequence[float]] = None):
    if sim:
        from robot_camera_calibration.sim_robot import SimRobot
        return SimRobot(sim_start_pose or [0.45, 0.0, 0.30, 3.1416, 0.0, 0.0])
    from robot_camera_calibration.robot_interface import RobotInterface
    rc = cfg["robot"]
    return RobotInterface(rc["ip"], int(rc["port"]), rc["move_command"], rc["stop_command"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check the calibration moves (camera axes)")
    ap.add_argument("--config", default=None, help="config file (default: the package's config/calibration.yaml)")
    ap.add_argument("--camera-side", default=None, help="use instead of sequence.camera_side in the config (+x/-x/+y/-y)")
    ap.add_argument("--sim", action="store_true", help="run with the simulator instead of the robot")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation before moving")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("axes", help="print the axis mapping")
    sub.add_parser("plan", help="print the grid around the current pose (no motion)")
    p_step = sub.add_parser("step", help="move once along one camera axis")
    p_step.add_argument("axis", choices=mf.VIEW_AXES)
    p_step.add_argument("distance_mm", type=float, help="distance [mm], the sign gives the direction")
    sub.add_parser("run", help="move through the whole grid (no measurement)")
    args = ap.parse_args(argv)

    config_path = Path(args.config) if args.config else default_config_path()
    try:
        cfg = load_config(config_path, args.camera_side)
    except Exception as e:
        print(f"Config error ({config_path}): {e}")
        return 1
    print(f"Config: {config_path}")

    if args.command == "axes":
        return cmd_axes(cfg)

    robot = make_robot(cfg, args.sim)
    confirm = (lambda _text: True) if args.yes else ask_yes_no
    if not robot.connect():
        print(f"Waiting for the robot connection: {robot.last_error}")
    print(f"Robot: {robot.description}")
    try:
        if args.command == "plan":
            return cmd_plan(cfg, robot)
        if args.command == "step":
            return cmd_step(cfg, robot, args.axis, args.distance_mm, confirm)
        return cmd_run(cfg, robot, confirm)
    except KeyboardInterrupt:
        print("\nInterrupted - stopping the robot.")
        robot.stop_motion()
        return 1
    finally:
        robot.close()


if __name__ == "__main__":
    sys.exit(main())
