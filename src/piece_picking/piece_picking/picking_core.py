"""
picking_core.py - load calibration results, camera coordinates -> robot base coordinates, compute picking targets.

There is one calibration result per camera (calib_gui of robot_camera_calibration):
  calibration_rgb.yaml   : base_T_cam = RGB (Triton) camera -> base
  calibration_depth.yaml : base_T_cam = depth (Helios ToF) camera -> base

If the vision result (mm) is in the same camera frame as the selected result, convert directly with  p_base = R p_cam + t.
If the frames differ, first align with the LUCID Chroma ToF -> RGB relation (orientation.yml, p_rgb = R_o p_tof + t_o [mm])
(the GUI currently uses only the same camera frame as the selected result).

Picking target (vision result: position [mm] in the camera frame + surface normal [direction vector])
  * The normal is a direction, so rotation only:  N_base = R N_cam  (no translation).
    The suction direction (tool +Z) points into the surface — the face is seen by the camera, so the sign is chosen pointing away from the camera.
  * Apply rotation (tilt): turn the reference tool orientation as little as possible so tool +Z matches the suction direction (suction, so the rotation about the axis is kept).
    No rotation (keep): reference tool orientation unchanged — vertical pick.
  * Approach and retreat follow the tool axis (suction direction). Even for tilted picks the pad enters perpendicular to the surface.
TODO: the products are inside a box, so an approach path / tool orientation sequence that avoids the box walls still needs to be worked out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np
import yaml

from robot_camera_calibration import pose_math as pm

ORIENTATION_LABELS = {"tilt": "tilted along the normal", "keep": "tool orientation unchanged (vertical pick)"}

FRAMES = ("rgb", "tof")
FRAME_LABELS = {"rgb": "RGB camera (Triton)", "tof": "ToF camera (Helios)"}
CALIB_KINDS = ("rgb", "depth")
CALIB_LABELS = {"rgb": "RGB camera result", "depth": "Depth (ToF) result"}
KIND_TO_FRAME = {"rgb": "rgb", "depth": "tof"}


# ======================================================================
# Calibration results
# ======================================================================
def _rigid(R, t) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or abs(np.linalg.det(R) - 1.0) > 1e-4:
        raise ValueError("R is not a rotation matrix (not orthogonal / det != 1)")
    return R, t


@dataclass
class CalibrationResult:
    path: Path
    R: np.ndarray                       # base_T_cam rotation
    t: np.ndarray                       # base_T_cam translation [m]
    meta: dict = field(default_factory=dict)
    camera: str = "depth"               # "rgb" | "depth" — which camera this result belongs to

    @property
    def frame(self) -> str:
        return KIND_TO_FRAME[self.camera]

    def summary_lines(self) -> List[str]:
        m = self.meta
        lines = [f"File: {self.path}"]
        if m.get("created"):
            lines.append(f"Created: {m['created']}")
        if "rms_error_mm" in m:
            line = (f"RMS {m['rms_error_mm']:.2f} mm, max {m.get('max_error_mm', float('nan')):.2f} mm "
                    f"({m.get('samples_used', '?')}/{m.get('samples_total', '?')} samples)")
            if "rms_reprojection_px" in m:
                line += f", reprojection {m['rms_reprojection_px']:.2f} px"
            lines.append(line)
        lines.append(f"Camera ({FRAME_LABELS[self.frame]}) position [" + ", ".join(f"{v * 1000:.1f}" for v in self.t) + "] mm")
        return lines


def load_calibration(path, expected: Optional[str] = None) -> CalibrationResult:
    """Read a calib_gui result file. With expected, check that it is the result of that camera.

    Older files without meta.camera_kind (aruco_rigid_result.yaml) are treated as depth results.
    """
    path = Path(path).expanduser()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "base_T_cam" not in data:
        raise ValueError(f"No base_T_cam entry: {path}")
    R, t = _rigid(data["base_T_cam"]["R"], data["base_T_cam"]["t"])
    meta = data.get("meta") or {}
    camera = str(meta.get("camera_kind", "depth"))
    if camera not in CALIB_KINDS:
        raise ValueError(f"Unknown camera kind '{camera}': {path}")
    if expected is not None and camera != expected:
        raise ValueError(f"Not a {CALIB_LABELS[expected]} but a {CALIB_LABELS[camera]} file: {path}")
    return CalibrationResult(path, R, t, meta, camera)


# ======================================================================
# RGB(Triton) <-> ToF(Helios)  (LUCID Chroma orientation.yml)
# ======================================================================
@dataclass
class ChromaOrientation:
    """Result of LUCID py_HLTRGB_2_orientation. Helios 3D point (mm) -> Triton camera coordinates (mm)."""
    path: Path
    R: np.ndarray           # rotationVector as a matrix
    t_mm: np.ndarray        # translationVector [mm]

    def rgb_to_tof_m(self, p_rgb_m: Sequence[float]) -> np.ndarray:
        """RGB camera coordinates [m] -> ToF camera coordinates [m]."""
        p_mm = np.asarray(p_rgb_m, dtype=float).reshape(3) * 1000.0
        return self.R.T @ (p_mm - self.t_mm) / 1000.0

    def tof_to_rgb_m(self, p_tof_m: Sequence[float]) -> np.ndarray:
        """ToF camera coordinates [m] -> RGB camera coordinates [m]."""
        p_mm = np.asarray(p_tof_m, dtype=float).reshape(3) * 1000.0
        return (self.R @ p_mm + self.t_mm) / 1000.0


def load_orientation(path) -> ChromaOrientation:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    fs = cv.FileStorage(str(path), cv.FileStorage_READ)
    try:
        rvec = fs.getNode("rotationVector").mat()
        tvec = fs.getNode("translationVector").mat()
    finally:
        fs.release()
    if rvec is None or tvec is None:
        raise ValueError(f"No rotationVector / translationVector in: {path}")
    R, _ = cv.Rodrigues(np.asarray(rvec, dtype=float).reshape(3))
    return ChromaOrientation(path, R, np.asarray(tvec, dtype=float).reshape(3))


# ======================================================================
# Camera coordinates -> base coordinates
# ======================================================================
def camera_point_to_base(p_cam_mm: Sequence[float], frame: str, calib: CalibrationResult,
                         orientation: Optional[ChromaOrientation] = None) -> Tuple[np.ndarray, str]:
    """Camera point [mm] -> base coordinates [m]. Returns: (point, description of the transform used)."""
    if frame not in FRAMES:
        raise ValueError(f"Frame must be one of {FRAMES}: '{frame}'")
    p = np.asarray(p_cam_mm, dtype=float).reshape(3) / 1000.0
    name = Path(calib.path).name
    if frame == calib.frame:
        return calib.R @ p + calib.t, name
    if orientation is None:
        raise ValueError(f"The input is in the {FRAME_LABELS[frame]} frame but the calibration is in the {FRAME_LABELS[calib.frame]} "
                         "frame. Select the result of the same camera, or a ToF↔RGB transform (orientation.yml) is needed.")
    p = orientation.rgb_to_tof_m(p) if frame == "rgb" else orientation.tof_to_rgb_m(p)
    return calib.R @ p + calib.t, f"orientation.yml + {name}"


def camera_normal_to_base(n_cam: Sequence[float], p_cam_mm: Sequence[float], frame: str,
                          calib: CalibrationResult,
                          orientation: Optional[ChromaOrientation] = None) -> Tuple[Optional[np.ndarray], bool]:
    """Vision normal (direction vector in the camera frame, any length) -> unit suction direction in the base frame.

    N_base = R N_cam (rotation only). The suction direction must point into the surface, so if the normal points
    against the viewing ray (camera -> product point), i.e. towards the camera = outward normal, it is flipped. A zero normal gives (None, False).
    Returns: (suction direction, whether it was flipped)
    """
    n = np.asarray(n_cam, dtype=float).reshape(3)
    norm = float(np.linalg.norm(n))
    if norm < 1e-6:
        return None, False
    n = n / norm
    ray = np.asarray(p_cam_mm, dtype=float).reshape(3)
    if np.linalg.norm(ray) < 1e-6:
        ray = np.array([0.0, 0.0, 1.0])
    flipped = float(np.dot(n, ray)) < 0.0
    if flipped:
        n = -n
    if frame != calib.frame:
        if orientation is None:
            raise ValueError(f"The normal is in the {FRAME_LABELS[frame]} frame but the calibration is in the {FRAME_LABELS[calib.frame]} frame.")
        n = orientation.R.T @ n if frame == "rgb" else orientation.R @ n
    return calib.R @ n, flipped


def describe_direction(approach_dir_base: Sequence[float]) -> str:
    """Suction direction -> "12.3° from vertical, product face points towards base +X" (to check the tilt direction)."""
    d = np.asarray(approach_dir_base, dtype=float)
    tilt = math.degrees(math.acos(float(np.clip(-d[2], -1.0, 1.0))))
    if tilt < 0.5:
        return f"{tilt:.1f}° from vertical (almost vertical)"
    face = -d[:2]                                     # horizontal part of the outward surface normal
    i = int(np.argmax(np.abs(face)))
    axis = ("+" if face[i] > 0 else "-") + "XY"[i]
    return (f"{tilt:.1f}° from vertical, product face points towards base {axis} "
            f"(azimuth {math.degrees(math.atan2(face[1], face[0])):.0f}°)")


# ======================================================================
# Picking target
# ======================================================================
@dataclass
class PickPlan:
    product_base: np.ndarray    # product point, base [m]
    target_pose: List[float]    # final TCP pose [m, rad] (value sent to the robot)
    approach_pose: List[float]  # approach TCP pose
    tip_target: np.ndarray = None    # suction tip position at the final point, base [m]
    tip_approach: np.ndarray = None  # suction tip position at the approach point
    gripper_mm: float = 0.0          # gripper length used (active TCP -> suction tip, tool +Z)
    tool_rotation: np.ndarray = None   # target tool rotation (3x3 in the base frame)
    mode: str = "keep"                 # "tilt" (tilted along the normal) | "keep" (reference tool orientation unchanged)
    rotation_change_deg: float = 0.0   # angle changed from the reference tool orientation
    tilt_deg: float = 0.0              # angle of the suction direction (tool +Z) from straight down (base -Z)
    retreat_pose: List[float] = None   # after the pick: backed off from the target along the tool axis (TCP)
    lift_pose: List[float] = None      # then lifted straight up along base +Z, same tool orientation (TCP)


def gripper_offset_base(tcp_pose: Sequence[float], gripper_mm: float, convention: str = "rpy") -> np.ndarray:
    """Vector from the active TCP to the suction tip in base coordinates [m]. The gripper extends along tool +Z."""
    return pm.rotation_from_pose(tcp_pose, convention) @ np.array([0.0, 0.0, float(gripper_mm) / 1000.0])


def _align_z(R_current: np.ndarray, z_target: np.ndarray) -> np.ndarray:
    """Turn the current tool rotation as little as possible so tool +Z matches z_target (suction: rotation about the axis is kept)."""
    z0 = R_current[:, 2]
    z1 = np.asarray(z_target, dtype=float) / np.linalg.norm(z_target)
    axis = np.cross(z0, z1)
    s, c = float(np.linalg.norm(axis)), float(np.clip(np.dot(z0, z1), -1.0, 1.0))
    if s < 1e-12:
        if c > 0:
            return R_current.copy()
        axis = R_current[:, 0]                      # exactly opposite: 180 deg about tool X
        return cv.Rodrigues(axis * math.pi)[0] @ R_current
    return cv.Rodrigues(axis / s * math.atan2(s, c))[0] @ R_current


def plan_pick(product_base_m: Sequence[float], reference_tcp_pose: Sequence[float],
              approach_mm: float, final_offset_mm: float, gripper_mm: float = 0.0,
              convention: str = "rpy", approach_dir_base: Optional[Sequence[float]] = None,
              retreat_mm: Optional[float] = None, lift_mm: float = 0.0) -> PickPlan:
    """Targets are set on the suction tip; the TCP pose sent to the robot is pulled back by the gripper length.

    * Tool rotation: if approach_dir_base (suction direction, base) is given, the reference tool orientation (reference_tcp_pose)
      is turned as little as possible so tool +Z points along it (tilt). Otherwise the reference orientation is kept (keep, vertical pick).
    * Suction direction z = tool +Z. Suction tip final point = product point - z * final_offset  (stops in front of the surface)
    * Suction tip approach point = final point - z * approach  (backed off along the tool axis)
    * TCP = suction tip - z * gripper length
    * After the pick: retreat = final point - z * retreat_mm (along the tool axis, default = approach), then
      lift = retreat + base Z * lift_mm with the same tool orientation, so the held product clears the box before the
      tool turns back upright.
    The approach and final points have the same tool orientation, so approach point -> product is a straight move along tool +Z.
    With the tool pointing straight down, this is the same as "above the product along +Z".
    TODO: approach path that avoids the box walls (now a straight approach along the tool axis).
    """
    p = np.asarray(product_base_m, dtype=float).reshape(3)
    R_ref = pm.rotation_from_pose(reference_tcp_pose, convention)
    if approach_dir_base is None:
        mode, R_tool = "keep", R_ref
        orient = [float(v) for v in reference_tcp_pose[3:6]]
    else:
        mode, R_tool = "tilt", _align_z(R_ref, approach_dir_base)
        orient = [float(v) for v in pm.rotation_to_pose_params(R_tool, convention)]
    z = R_tool[:, 2]
    tip_target = p - z * float(final_offset_mm) / 1000.0
    tip_approach = tip_target - z * float(approach_mm) / 1000.0
    g = z * float(gripper_mm) / 1000.0
    tilt = math.degrees(math.acos(float(np.clip(-z[2], -1.0, 1.0))))
    back = float(approach_mm if retreat_mm is None else retreat_mm) / 1000.0
    retreat = tip_target - z * back - g
    lift = retreat + np.array([0.0, 0.0, float(lift_mm) / 1000.0])
    return PickPlan(p, list(tip_target - g) + orient, list(tip_approach - g) + orient,
                    tip_target, tip_approach, float(gripper_mm), R_tool, mode,
                    pm.rotation_angle_deg(R_ref, R_tool), tilt,
                    [float(v) for v in retreat] + orient, [float(v) for v in lift] + orient)


def gripper_missing(active_tcp_offset: Sequence[float], gripper_mm: float) -> bool:
    """The pendant's active TCP is the flange (0) and the gripper length is also 0 — the flange goes down to the product and the gripper collides."""
    return gripper_mm <= 0.0 and abs(float(active_tcp_offset[2])) * 1000.0 <= 1.0


def gripper_tcp_warning(active_tcp_offset: Sequence[float], gripper_mm: float) -> str:
    """Warn if the pendant's active TCP already includes the tool length and a gripper length is entered too (applied twice)."""
    z_mm = float(active_tcp_offset[2]) * 1000.0
    if gripper_mm > 0.0 and abs(z_mm) > 1.0:
        return (f"The pendant's active TCP already has Z {z_mm:.1f} mm. Adding the gripper length {gripper_mm:.1f} mm "
                "applies it twice (the suction tip stops that much higher). Use only one of them.")
    if gripper_mm <= 0.0 and abs(z_mm) <= 1.0:
        return ("The gripper length is 0 and the pendant's active TCP is the flange. The flange goes down to the product position, "
                "so the suction gripper collides with the product / box by its length. Enter the gripper length.")
    return ""


def check_plan(plan: PickPlan, current_tcp_pose: Sequence[float], safety: dict) -> List[str]:
    """Problems to catch before moving (empty = OK).

    The move distance is checked on the TCP; the Z limit / workspace on the suction tip, the point closest to the table / box.
    """
    problems = []
    cur = np.asarray(current_tcp_pose[:3], dtype=float)
    max_move = safety.get("max_move_mm")
    tips = (plan.tip_approach, plan.tip_target)
    for (name, pose), tip in zip((("Approach point", plan.approach_pose), ("Target point", plan.target_pose)), tips):
        xyz = np.asarray(pose[:3], dtype=float)
        tip = xyz if tip is None else np.asarray(tip, dtype=float)
        if max_move is not None:
            dist = float(np.linalg.norm(xyz - cur)) * 1000.0
            if dist > float(max_move):
                problems.append(f"{name} is {dist:.0f} mm away from the current TCP "
                                f"(limit {float(max_move):.0f} mm). Check the input units / frame.")
        z_min = safety.get("z_min_mm")
        if z_min is not None and tip[2] * 1000.0 < float(z_min):
            problems.append(f"{name} suction tip Z {tip[2] * 1000:.1f} mm is below the limit {float(z_min):.1f} mm.")
        box = safety.get("workspace_mm")
        if box:
            for i, axis in enumerate("XYZ"):
                lo, hi = float(box[i][0]), float(box[i][1])
                v = tip[i] * 1000.0
                if not lo <= v <= hi:
                    problems.append(f"{name} suction tip {axis} {v:.1f} mm is outside the workspace [{lo:.0f}, {hi:.0f}].")
    max_tilt = safety.get("max_tilt_deg")
    if max_tilt is not None and plan.tilt_deg > float(max_tilt):
        problems.append(f"The suction direction tilts {plan.tilt_deg:.0f}° from vertical (limit {float(max_tilt):.0f}°). "
                        "Check the normal or turn off 'Apply rotation'.")
    max_rot = safety.get("max_rotation_deg")
    if max_rot is not None and plan.rotation_change_deg > float(max_rot):
        problems.append(f"The tool must rotate {plan.rotation_change_deg:.0f}° from the reference pose (limit {float(max_rot):.0f}°). "
                        "Check the normal or turn off 'Apply rotation'.")
    return problems


def format_xyz_mm(p_m: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v) * 1000.0:.1f}" for v in p_m[:3]) + "] mm"


def format_pose(pose: Sequence[float]) -> str:
    x, y, z = (float(v) * 1000.0 for v in pose[:3])
    a, b, c = (math.degrees(float(v)) for v in pose[3:6])
    return f"X {x:.1f}  Y {y:.1f}  Z {z:.1f} mm | RX {a:.2f}  RY {b:.2f}  RZ {c:.2f} deg"
