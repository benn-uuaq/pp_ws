"""
motion_frame.py - move axes that follow the camera position.

The calibration grid and single moves are defined not in robot base X/Y/Z but in **three directions
seen from the camera**, so that "50 mm towards the camera" means the same thing whichever side of
the base the camera is on.

  lateral  (left/right)  right side of the image when the camera looks at the robot
  depth    (depth)       horizontal direction towards the camera
  vertical (up/down)     up (base +Z)

camera_side is the horizontal position of the camera seen from the robot base (+x, -x, +y, -y).
depth is the horizontal unit vector d from the base towards the camera, and lateral is Z x d.
The three axes are mutually perpendicular unit vectors, but they follow the names "right / towards camera / up",
so they form a left-handed frame (lateral x depth = -vertical). They are only used for translations, so this is fine.

  camera_side   lateral   depth   vertical
  +x            +Y        +X      +Z
  -x            -Y        -X      +Z
  +y            -X        +Y      +Z
  -y            +X        -Y      +Z

When the camera looks down from above, camera_side is still chosen by its horizontal position. Then
depth points towards the spot right below the camera, and the sign of lateral may differ from the image's
right side depending on how the camera is mounted. The grid is symmetric, so the sign does not
affect the calibration result.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

VIEW_AXES = ("lateral", "depth", "vertical")

# Horizontal direction from the base towards the camera
CAMERA_SIDES = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
}

_BASE_NAMES = ("X", "Y", "Z")


def normalize_side(side) -> str:
    """Normalise notations like '+X', 'x', ' -y ' to '+x' / '-y'."""
    text = str(side).strip().lower().replace(" ", "")
    if text in ("x", "y"):
        text = "+" + text
    if text not in CAMERA_SIDES:
        raise ValueError(f"camera_side must be one of {', '.join(CAMERA_SIDES)}: '{side}'")
    return text


def view_axes(side) -> np.ndarray:
    """3x3 matrix whose columns are the base unit vectors of lateral, depth, vertical."""
    depth = np.array(CAMERA_SIDES[normalize_side(side)])
    vertical = np.array([0.0, 0.0, 1.0])
    lateral = np.cross(vertical, depth)
    return np.column_stack([lateral, depth, vertical])


def view_to_base(side, offset_view: Sequence[float]) -> np.ndarray:
    """Convert a [lateral, depth, vertical] displacement to a base [X, Y, Z] displacement."""
    return view_axes(side) @ np.asarray(offset_view, dtype=float).reshape(3)


def base_to_view(side, offset_base: Sequence[float]) -> np.ndarray:
    """Convert a base [X, Y, Z] displacement to a [lateral, depth, vertical] displacement."""
    return view_axes(side).T @ np.asarray(offset_base, dtype=float).reshape(3)


def axis_index(axis: str) -> int:
    name = str(axis).strip().lower()
    if name not in VIEW_AXES:
        raise ValueError(f"Axis name must be one of {', '.join(VIEW_AXES)}: '{axis}'")
    return VIEW_AXES.index(name)


def base_axis_label(vector: Sequence[float]) -> str:
    """Write a unit vector as a base axis name like '+X' (axis-aligned vectors only)."""
    v = np.asarray(vector, dtype=float)
    i = int(np.argmax(np.abs(v)))
    return ("+" if v[i] > 0 else "-") + _BASE_NAMES[i]


def describe(side) -> str:
    """One-line summary like 'lateral=+Y, depth=+X, vertical=+Z'."""
    axes = view_axes(side)
    return ", ".join(f"{name}={base_axis_label(axes[:, i])}" for i, name in enumerate(VIEW_AXES))


def nearest_side(direction_xy: Sequence[float]) -> str:
    """The camera_side closest to a horizontal direction vector."""
    d = np.asarray(direction_xy, dtype=float)[:2]
    return max(CAMERA_SIDES, key=lambda s: float(np.dot(CAMERA_SIDES[s][:2], d)))


def format_view_mm(offset_view: Sequence[float]) -> str:
    return " / ".join(f"{name} {float(v) * 1000.0:+6.1f}" for name, v in zip(VIEW_AXES, offset_view)) + " mm"


def format_base_mm(offset_base: Sequence[float]) -> str:
    return " / ".join(f"{name} {float(v) * 1000.0:+6.1f}" for name, v in zip("XYZ", offset_base)) + " mm"


# ======================================================================
# Grid
# ======================================================================
def _axis_offsets(n: int, step: float) -> List[float]:
    half = (n - 1) / 2.0
    return [(i - half) * step for i in range(n)]


def grid_offsets_view(step_m: Sequence[float], points_per_axis: int = 3) -> List[Tuple[float, float, float]]:
    """Grid displacements [m] in camera axes [lateral, depth, vertical].

    With 3 points per axis it is -step, 0, +step on each axis = 27 points. Visited as a "snake" (boustrophedon),
    outermost vertical, then depth, then lateral, so two consecutive points are always exactly one
    cell apart.
    """
    if points_per_axis < 2:
        raise ValueError("points_per_axis must be >= 2")
    ol = _axis_offsets(points_per_axis, float(step_m[0]))
    od = _axis_offsets(points_per_axis, float(step_m[1]))
    ov = _axis_offsets(points_per_axis, float(step_m[2]))

    offsets = []
    row = 0
    for iv, dv in enumerate(ov):
        ds = od if iv % 2 == 0 else od[::-1]
        for dd in ds:
            ls = ol if row % 2 == 0 else ol[::-1]
            for dl in ls:
                offsets.append((dl, dd, dv))
            row += 1
    return offsets


def offset_pose(base_pose: Sequence[float], side, offset_view: Sequence[float]) -> List[float]:
    """Pose translated from base_pose by a camera-axis displacement. The orientation is kept."""
    d = view_to_base(side, offset_view)
    return [float(base_pose[i]) + float(d[i]) for i in range(3)] + [float(v) for v in base_pose[3:6]]


def generate_grid_poses(base_pose: Sequence[float], side,
                        step_m: Sequence[float] = (0.05, 0.05, 0.05),
                        points_per_axis: int = 3) -> List[List[float]]:
    """Translation grid around base_pose. step_m is in [lateral, depth, vertical] order."""
    return [offset_pose(base_pose, side, o) for o in grid_offsets_view(step_m, points_per_axis)]


def step_pose(base_pose: Sequence[float], side, axis: str, distance_m: float) -> List[float]:
    """Pose moved by distance_m along one axis only (e.g. depth +0.02 = 20 mm towards the camera)."""
    offset = [0.0, 0.0, 0.0]
    offset[axis_index(axis)] = float(distance_m)
    return offset_pose(base_pose, side, offset)
