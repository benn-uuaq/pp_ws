r"""
pose_math.py - Robot pose math, taken unchanged from calib_core.py of the
original calibration tool (D:\RND_WORK\PiecePicking\robot_camera_calibration).

Only the camera-independent part is here. Marker detection, depth sampling
and the rigid fit stay out until the Helios camera is integrated.

* Robot poses are [x, y, z, rx, ry, rz] in metres and radians, base frame.
* ``convention``: "rpy" (R = Rz @ Ry @ Rx, Elite CS) or "rotvec" (UR style).
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

ROTATION_CONVENTIONS = ("rpy", "rotvec")


# ======================================================================
# Rotations and poses
# ======================================================================
def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Roll-pitch-yaw (fixed X-Y-Z axes) to a rotation matrix: Rz @ Ry @ Rx."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def rotvec_to_matrix(rotvec: Sequence[float]) -> np.ndarray:
    """Rotation vector (axis * angle, radians) to a rotation matrix (Rodrigues)."""
    v = np.asarray(rotvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=float)
    return np.eye(3) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * (kx @ kx)


def rotation_from_pose(pose: Sequence[float], convention: str) -> np.ndarray:
    """Rotation matrix of a 6D pose, using the robot's orientation convention."""
    rx, ry, rz = (float(v) for v in pose[3:6])
    if convention == "rpy":
        return rpy_to_matrix(rx, ry, rz)
    if convention == "rotvec":
        return rotvec_to_matrix([rx, ry, rz])
    raise ValueError(f"Unknown rotation convention '{convention}'. "
                     f"Use one of {ROTATION_CONVENTIONS}.")


def pose_to_matrix(pose: Sequence[float], convention: str) -> np.ndarray:
    """6D pose -> 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = rotation_from_pose(pose, convention)
    T[:3, 3] = np.asarray(pose[:3], dtype=float)
    return T


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle [deg] of the relative rotation between two rotation matrices."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def marker_point_in_base(tcp_pose: Sequence[float],
                         tcp_offset: Sequence[float],
                         marker_offset_m: Sequence[float],
                         convention: str) -> np.ndarray:
    """Position of the marker centre in the robot base frame [m].

    tcp_pose        : actual TCP pose reported by the robot (base frame).
    tcp_offset      : active TCP offset relative to the flange
                      (all zeros when the active TCP is the flange itself).
    marker_offset_m : marker centre expressed in the FLANGE frame [m].

    The flange pose is recovered as  T_base_flange = T_base_tcp @ inv(T_flange_tcp),
    so the marker offset is always relative to the flange, whatever TCP is
    active on the pendant.
    """
    T_base_tcp = pose_to_matrix(tcp_pose, convention)
    T_flange_tcp = pose_to_matrix(tcp_offset, convention)
    T_base_flange = T_base_tcp @ np.linalg.inv(T_flange_tcp)
    o = np.asarray(marker_offset_m, dtype=float).reshape(3)
    return T_base_flange[:3, :3] @ o + T_base_flange[:3, 3]


# ======================================================================
# Text formatting
# ======================================================================
def format_pose(pose: Sequence[float]) -> str:
    """Pose as 'X Y Z [mm] | RX RY RZ [deg]' for logs and the status panel."""
    x, y, z = (float(v) * 1000.0 for v in pose[:3])
    a, b, c = (math.degrees(float(v)) for v in pose[3:6])
    return (f"X {x:8.1f}  Y {y:8.1f}  Z {z:8.1f} mm | "
            f"RX {a:7.2f}  RY {b:7.2f}  RZ {c:7.2f} deg")


def format_mm(vec: Sequence[float], digits: int = 1) -> str:
    return "[" + ", ".join(f"{float(v) * 1000.0:.{digits}f}" for v in vec) + "] mm"


def format_deg(vec: Sequence[float], digits: int = 1) -> str:
    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in vec) + "] deg"


# ======================================================================
# Marker rotation (added for pp_ws)
# ======================================================================
def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> roll-pitch-yaw [rad] with R = Rz @ Ry @ Rx (inverse of rpy_to_matrix)."""
    R = np.asarray(R, dtype=float)
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(R[2, 0]) < 1.0 - 1e-9:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:                                   # gimbal lock: put everything into yaw
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw])


def rpy_deg_to_matrix(rpy_deg: Sequence[float]) -> np.ndarray:
    return rpy_to_matrix(*(math.radians(float(v)) for v in rpy_deg))


def reference_rotation(tcp_pose: Sequence[float], tcp_offset: Sequence[float],
                       convention: str, from_tcp: bool) -> np.ndarray:
    """Rotation of the marker reference frame in the base frame: the active TCP
    itself (from_tcp) or the flange recovered from the TCP pose and offset."""
    R_base_tcp = rotation_from_pose(tcp_pose, convention)
    if from_tcp:
        return R_base_tcp
    return R_base_tcp @ rotation_from_pose(tcp_offset, convention).T


# ======================================================================
# Jog helpers (added for pp_ws)
# ======================================================================
def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle) [rad]."""
    R = np.asarray(R, dtype=float)
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:                     # 180 deg: axis from the diagonal
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(axis))
        for j in range(3):
            if j != i and R[i, j] < 0:
                axis[j] = -axis[j]
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(angle))
    return axis * angle


def rotation_to_pose_params(R: np.ndarray, convention: str) -> np.ndarray:
    """Inverse of rotation_from_pose: the robot's (rx, ry, rz) for a rotation matrix."""
    if convention == "rpy":
        return matrix_to_rpy(R)
    if convention == "rotvec":
        return matrix_to_rotvec(R)
    raise ValueError(f"Unknown rotation convention '{convention}'.")


def rotate_pose_about_base(pose: Sequence[float], axis: int, angle_rad: float,
                           convention: str) -> List[float]:
    """Rotate the tool about base axis 0/1/2 (X/Y/Z) through the TCP; the TCP position stays."""
    rot = [rpy_to_matrix(angle_rad, 0, 0), rpy_to_matrix(0, angle_rad, 0), rpy_to_matrix(0, 0, angle_rad)][axis]
    R = rot @ rotation_from_pose(pose, convention)
    return [float(v) for v in pose[:3]] + [float(v) for v in rotation_to_pose_params(R, convention)]
