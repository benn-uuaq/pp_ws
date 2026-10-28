"""
settings.py - config file (config/calibration.yaml) and the last UI inputs.

motion_tool and calib_gui share the same config. Values changed in the UI (robot IP, camera
position, ...) do not modify the config file; they are kept in ui_state.yaml in the output folder
and loaded again at the next start.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Optional

import yaml

from robot_camera_calibration import motion_frame as mf
from robot_camera_calibration import pose_math as pm

PACKAGE = "robot_camera_calibration"
UI_STATE_FILE = "ui_state.yaml"
CAMERA_KINDS = ("rgb", "depth")       # rgb = Triton RGB, depth = Helios ToF
CAMERA_LABELS = {"rgb": "RGB (Triton)", "depth": "Depth (Helios ToF)"}

DEFAULT_CONFIG = {
    "active_camera": "rgb",
    "robot": {
        "ip": "192.168.1.101",
        "port": 30001,
        "rotation_convention": "rpy",
        "velocity": 0.10,
        "acceleration": 0.50,
        "move_command": "movel({pose}, a={a}, v={v})",
        "stop_command": "stopl(1.0)",
        "move_timeout_s": 30.0,
        "position_tolerance_mm": 1.0,
        "settle_time_s": 0.5,
    },
    "camera": {
        "ip": "",
        "operating_mode": "Distance3000mmSingleFreq",
        "exposure_time": "",
        "image_accumulation": 1,
        "spatial_filter": True,
        "confidence_threshold": True,
        "timeout_ms": 2000,
        "packet_size": 1500,
        "packet_delay_ns": 20000,
    },
    "camera_rgb": {
        "ip": "192.168.1.159",
        "pixel_format": "BayerRG8",
        "intrinsics_file": "",
        "timeout_ms": 3000,
        "packet_size": 1500,
        "packet_delay_ns": 20000,
    },
    "marker": {
        "dictionary": "DICT_4X4_50",
        "id": 0,
        "size_mm": 60.0,
        "offset_from_flange_mm": [0.0, 0.0, 0.0],
        "rotation_deg": [0.0, 0.0, 0.0],
        "rotation_tolerance_deg": 5.0,
        "offset_reference": "flange",
        "sample_scale": 0.8,
        "min_points": 30,
    },
    "sequence": {
        "camera_side": "+x",
        "points_per_axis": 3,
        "step_mm": {"lateral": 50.0, "depth": 50.0, "vertical": 50.0},
        "return_to_start": True,
        "max_single_step_mm": 100.0,
        "frames_per_pose": 10,
        "max_point_std_mm": 3.0,
        "max_pixel_std_px": 2.0,
        "min_valid_samples": 12,
        "start_from_saved": True,
        "centering_tolerance_px": 15,
    },
    "jog": {
        "speed_mm_s": 20.0,
        "rot_speed_deg_s": 5.0,
        "acceleration": 0.2,
        "rot_acceleration": 0.5,
        "hold_s": 2.0,
        "steps": [[1.0, 0.5], [5.0, 1.0], [10.0, 2.0], [50.0, 5.0]],
    },
    "fit": {
        "outlier_factor": 3.0,
        "outlier_min_mm": 2.0,
        "max_rms_mm": 5.0,
        "ransac_px": 3.0,
    },
    "output": {
        "directory": "~/pp_ws/calibration",
        "result_files": {"rgb": "calibration_rgb.yaml", "depth": "calibration_depth.yaml"},
        "start_pose_file": "start_pose.yaml",
        "save_images": True,
    },
}


def default_config_path() -> Path:
    """Look for the config in the installed share first, otherwise use the one in the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        p = Path(get_package_share_directory(PACKAGE)) / "config" / "calibration.yaml"
        if p.exists():
            return p
    except Exception:
        pass
    return Path(__file__).resolve().parents[1] / "config" / "calibration.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def validate(cfg: dict) -> dict:
    if cfg["active_camera"] not in CAMERA_KINDS:
        raise ValueError(f"active_camera must be one of {CAMERA_KINDS}: '{cfg['active_camera']}'")
    if set(cfg["output"]["result_files"]) != set(CAMERA_KINDS):
        raise ValueError("output.result_files must have exactly the two entries rgb / depth")
    sc = cfg["sequence"]
    sc["camera_side"] = mf.normalize_side(sc["camera_side"])
    conv = cfg["robot"]["rotation_convention"]
    if conv not in pm.ROTATION_CONVENTIONS:
        raise ValueError(f"robot.rotation_convention must be one of {pm.ROTATION_CONVENTIONS}: '{conv}'")
    steps = sc["step_mm"]
    if not isinstance(steps, dict) or set(steps) != set(mf.VIEW_AXES):
        raise ValueError("sequence.step_mm must have the three entries lateral / depth / vertical "
                         "(the old [x, y, z] list format is not accepted because it ignores the camera position)")
    if len(cfg["marker"]["offset_from_flange_mm"]) != 3:
        raise ValueError("marker.offset_from_flange_mm must be three values [x, y, z]")
    if len(cfg["marker"]["rotation_deg"]) != 3:
        raise ValueError("marker.rotation_deg must be three values [rx, ry, rz]")
    if cfg["marker"]["offset_reference"] not in ("flange", "tcp"):
        raise ValueError("marker.offset_reference must be flange or tcp")
    return cfg


def load_config(path: Optional[Path] = None, camera_side: Optional[str] = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            _deep_merge(cfg, yaml.safe_load(f) or {})
    if camera_side:
        cfg["sequence"]["camera_side"] = camera_side
    return validate(cfg)


def step_m(cfg: dict) -> List[float]:
    """Grid spacing [m] in [lateral, depth, vertical]."""
    steps = cfg["sequence"]["step_mm"]
    return [float(steps[name]) / 1000.0 for name in mf.VIEW_AXES]


def output_dir(cfg: dict) -> Path:
    return Path(cfg["output"]["directory"]).expanduser()


def result_path(cfg: dict, kind: str) -> Path:
    """Result file per camera kind (rgb -> calibration_rgb.yaml, depth -> calibration_depth.yaml)."""
    return output_dir(cfg) / cfg["output"]["result_files"][kind]


# ======================================================================
# Calibration start pose (jog the robot, then [Save current pose])
# ======================================================================
def start_pose_path(cfg: dict) -> Path:
    return output_dir(cfg) / cfg["output"]["start_pose_file"]


def load_start_pose(cfg: dict) -> Optional[dict]:
    """Saved start pose {tcp_pose, tcp_offset, joints, saved} or None."""
    path = start_pose_path(cfg)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if len(data.get("tcp_pose", [])) != 6:
            return None
        return data
    except Exception:
        return None


def save_start_pose(cfg: dict, tcp_pose, tcp_offset, joints=None) -> Path:
    import datetime
    data = {
        "saved": datetime.datetime.now().isoformat(timespec="seconds"),
        "tcp_pose": [float(v) for v in tcp_pose],          # [m, rad], actual TCP in the base frame
        "tcp_offset": [float(v) for v in tcp_offset],      # active TCP offset when saved
        "joints": None if joints is None else [float(v) for v in joints],   # [rad]
    }
    path = start_pose_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Calibration start pose (calib_gui [Save current pose])\n")
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return path


# ======================================================================
# Last UI inputs
# ======================================================================
def ui_state(cfg: dict) -> dict:
    """Extract only the values the UI handles from the config."""
    return {
        "active_camera": cfg["active_camera"],
        "robot": {"ip": cfg["robot"]["ip"]},
        "camera": {"ip": cfg["camera"]["ip"]},
        "camera_rgb": {"ip": cfg["camera_rgb"]["ip"]},
        "marker": {"offset_from_flange_mm": [float(v) for v in cfg["marker"]["offset_from_flange_mm"]],
                   "rotation_deg": [float(v) for v in cfg["marker"]["rotation_deg"]],
                   "offset_reference": cfg["marker"]["offset_reference"]},
        "sequence": {"camera_side": cfg["sequence"]["camera_side"],
                     "step_mm": {k: float(v) for k, v in cfg["sequence"]["step_mm"].items()},
                     "start_from_saved": bool(cfg["sequence"]["start_from_saved"])},
        "jog": {"speed_mm_s": float(cfg["jog"]["speed_mm_s"]),
                "rot_speed_deg_s": float(cfg["jog"]["rot_speed_deg_s"])},
    }


def apply_ui_state(cfg: dict) -> bool:
    """If ui_state.yaml exists in the output folder, overlay it on the config."""
    path = output_dir(cfg) / UI_STATE_FILE
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = yaml.safe_load(f) or {}
        merged = _deep_merge(copy.deepcopy(cfg), state)
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
