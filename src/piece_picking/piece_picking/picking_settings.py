"""
picking_settings.py - picking settings (config/picking.yaml) and the last UI inputs (ui_state.yaml).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Optional

import yaml

from piece_picking.picking_core import CALIB_KINDS
from piece_picking.servo_motion import DEFAULT_SERVO

PACKAGE = "piece_picking"
UI_STATE_FILE = "ui_state.yaml"
MODES = ("manual", "trigger")

DEFAULT_CONFIG = {
    "robot": {
        "ip": "192.168.1.101",
        "port": 30001,
        "rotation_convention": "rpy",
        "velocity": 0.05,
        "acceleration": 0.30,
        "move_command": "movel({pose}, a={a}, v={v})",
        "stop_command": "stopl(1.0)",
        "move_timeout_s": 60.0,
        "position_tolerance_mm": 1.0,
    },
    "calibration": {
        "use": "rgb",
        "result_files": {
            "rgb": "~/pp_ws/calibration/calibration_rgb.yaml",
            "depth": "~/pp_ws/calibration/calibration_depth.yaml",
        },
    },
    "vision": {
        "mode": "manual",
        "trigger_topic": "/vision/trigger",
        "result_topic": "/vision/result",
        "timeout_s": 5.0,
        "manual_point_m": [0.0, 0.0, 1.0],
        "manual_normal": [0.0, 0.0, 0.0],
    },
    "pick": {
        "approach_mm": 50.0,
        "retreat_mm": 50.0,           # after the pick: back off along the tool axis ...
        "lift_mm": 200.0,             # ... then lift along base +Z before the tool turns back
        "final_offset_mm": 0.0,
        "gripper_length_mm": 0.0,
        "apply_normal": False,
        "descent_speed_ratio": 0.3,
        "start_from_home": True,
    },
    "servo": dict(DEFAULT_SERVO),
    "gripper": {
        "suction_do": [4, 5, 6, 7],
        "grip_level": False,          # output value while gripping (False = grip with OFF, normally ON)
        "grip_wait_s": 0.5,
        "release_wait_s": 0.5,
    },
    "safety": {
        "max_move_mm": 800.0,
        "z_min_mm": None,
        "workspace_mm": None,
        "max_tilt_deg": 60.0,
        "max_rotation_deg": 90.0,
    },
    "output": {
        "directory": "~/pp_ws/picking",
        "poses_file": "poses.yaml",
    },
}
POSE_NAMES = {"home": "Home", "standby": "Pick standby", "place": "Place"}   # GUI row order


def default_config_path() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory
        p = Path(get_package_share_directory(PACKAGE)) / "config" / "picking.yaml"
        if p.exists():
            return p
    except Exception:
        pass
    return Path(__file__).resolve().parents[1] / "config" / "picking.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def validate(cfg: dict) -> dict:
    cc = cfg["calibration"]
    if cc["use"] not in CALIB_KINDS:
        raise ValueError(f"calibration.use must be one of {CALIB_KINDS}: '{cc['use']}'")
    if set(cc["result_files"]) != set(CALIB_KINDS):
        raise ValueError("calibration.result_files must have exactly the two entries rgb / depth")
    vc = cfg["vision"]
    if vc["mode"] not in MODES:
        raise ValueError(f"vision.mode must be one of {MODES}: '{vc['mode']}'")
    if len(vc["manual_point_m"]) != 3:
        raise ValueError("vision.manual_point_m must be three values [x, y, z]")
    if len(vc["manual_normal"]) != 3:
        raise ValueError("vision.manual_normal must be three values [nx, ny, nz]")
    box = cfg["safety"]["workspace_mm"]
    if box is not None and (len(box) != 3 or any(len(r) != 2 for r in box)):
        raise ValueError("safety.workspace_mm must be [[xmin, xmax], [ymin, ymax], [zmin, zmax]]")
    gc = cfg["gripper"]
    do = gc["suction_do"]
    outs = [] if do is None else ([do] if isinstance(do, int) else list(do))
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in outs):
        raise ValueError(f"gripper.suction_do must be a list of DO numbers (e.g. [4, 5, 6, 7]): {do}")
    if not isinstance(gc["grip_level"], bool):
        raise ValueError("gripper.grip_level must be true / false")
    return cfg


def suction_outputs(cfg: dict) -> List[int]:
    """List of suction DO numbers (null -> [], a single number -> [n])."""
    do = cfg["gripper"]["suction_do"]
    return [] if do is None else ([int(do)] if isinstance(do, int) else [int(i) for i in do])


def load_config(path: Optional[Path] = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            _deep_merge(cfg, yaml.safe_load(f) or {})
    return validate(cfg)


def output_dir(cfg: dict) -> Path:
    return Path(cfg["output"]["directory"]).expanduser()


def ui_state(cfg: dict) -> dict:
    vc, pc = cfg["vision"], cfg["pick"]
    return {
        "robot": {"ip": cfg["robot"]["ip"], "velocity": float(cfg["robot"]["velocity"])},
        "calibration": {"use": cfg["calibration"]["use"],
                        "result_files": {k: str(v) for k, v in cfg["calibration"]["result_files"].items()}},
        "vision": {"mode": vc["mode"], "manual_point_m": [float(v) for v in vc["manual_point_m"]],
                   "manual_normal": [float(v) for v in vc["manual_normal"]]},
        "pick": {"approach_mm": float(pc["approach_mm"]), "final_offset_mm": float(pc["final_offset_mm"]),
                 "retreat_mm": float(pc["retreat_mm"]), "lift_mm": float(pc["lift_mm"]),
                 "gripper_length_mm": float(pc["gripper_length_mm"]),
                 "apply_normal": bool(pc["apply_normal"]), "start_from_home": bool(pc["start_from_home"])},
    }


# ======================================================================
# Home / pick standby / place poses ([Save ...] buttons)
# ======================================================================
def poses_path(cfg: dict) -> Path:
    return output_dir(cfg) / cfg["output"]["poses_file"]


def load_poses(cfg: dict) -> dict:
    """{"home": {...}, "standby": {...}, "place": {...}} — each {tcp_pose, joints, tcp_offset, saved}. Empty dict if none."""
    path = poses_path(cfg)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    return {k: v for k, v in data.items() if k in POSE_NAMES and isinstance(v, dict)
            and len(v.get("tcp_pose") or []) == 6}


def save_pose(cfg: dict, name: str, tcp_pose, joints, tcp_offset) -> dict:
    import datetime
    if name not in POSE_NAMES:
        raise ValueError(f"Unknown pose name: {name}")
    poses = load_poses(cfg)
    poses[name] = {
        "saved": datetime.datetime.now().isoformat(timespec="seconds"),
        "tcp_pose": [float(v) for v in tcp_pose],        # [m, rad] active TCP
        "joints": None if joints is None else [float(v) for v in joints],   # [rad] (moves to saved poses use the joints)
        "tcp_offset": [float(v) for v in tcp_offset],    # active TCP offset when saved
    }
    path = poses_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Piece-picking home / pick standby / place poses (picking_gui [Save ...])\n")
        yaml.safe_dump(poses, f, sort_keys=False, allow_unicode=True)
    return poses


def apply_ui_state(cfg: dict) -> bool:
    path = output_dir(cfg) / UI_STATE_FILE
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            saved = yaml.safe_load(f) or {}
        vs = saved.get("vision") or {}
        if "manual_point_mm" in vs and "manual_point_m" not in vs:      # old file (mm input)
            vs["manual_point_m"] = [float(v) / 1000.0 for v in vs.pop("manual_point_mm")]
        merged = _deep_merge(copy.deepcopy(cfg), saved)
        validate(merged)
    except Exception:
        return False
    cfg.clear()
    cfg.update(merged)
    return True


def save_ui_state(cfg: dict) -> Path:
    path = output_dir(cfg) / UI_STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(ui_state(cfg), f, sort_keys=False, allow_unicode=True)
    return path
