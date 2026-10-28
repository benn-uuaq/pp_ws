"""Checks the camera-relative move axes (motion_frame). Runs without a robot."""

import numpy as np
import pytest

from robot_camera_calibration import motion_frame as mf

# camera_side -> base axes of (lateral, depth, vertical)
EXPECTED = {
    "+x": ("+Y", "+X", "+Z"),
    "-x": ("-Y", "-X", "+Z"),
    "+y": ("-X", "+Y", "+Z"),
    "-y": ("+X", "-Y", "+Z"),
}


@pytest.mark.parametrize("side", sorted(EXPECTED))
def test_axes_table(side):
    axes = mf.view_axes(side)
    labels = tuple(mf.base_axis_label(axes[:, i]) for i in range(3))
    assert labels == EXPECTED[side]
    assert np.allclose(axes.T @ axes, np.eye(3))         # mutually perpendicular unit vectors


@pytest.mark.parametrize("text, side", [("+X", "+x"), ("x", "+x"), (" -y ", "-y"), ("+Y", "+y")])
def test_normalize_side(text, side):
    assert mf.normalize_side(text) == side


def test_unknown_side_is_rejected():
    with pytest.raises(ValueError):
        mf.normalize_side("+z")


@pytest.mark.parametrize("side", sorted(EXPECTED))
def test_depth_moves_toward_camera(side):
    start = [0.4, 0.1, 0.3, 3.14, 0.0, 0.0]
    target = mf.step_pose(start, side, "depth", 0.02)
    moved = np.subtract(target[:3], start[:3])
    assert np.allclose(moved, np.array(mf.CAMERA_SIDES[side]) * 0.02)
    assert target[3:] == start[3:]
    assert np.allclose(mf.base_to_view(side, moved), [0.0, 0.02, 0.0])


def test_grid_is_snake_and_contains_start():
    base = [0.4, 0.0, 0.3, 3.14, 0.0, 0.0]
    poses = mf.generate_grid_poses(base, "+x", (0.05, 0.05, 0.05), 3)
    assert len(poses) == 27
    steps = np.abs(np.diff(np.array(poses)[:, :3], axis=0)).sum(axis=1)
    assert np.allclose(steps, 0.05)                      # one cell at a time
    assert any(np.allclose(p, base) for p in poses)      # includes the start pose
    assert all(p[3:] == base[3:] for p in poses)         # orientation kept


@pytest.mark.parametrize("side, half_range_xyz", [
    ("+x", (0.06, 0.03, 0.04)),     # depth -> X, lateral -> Y
    ("-x", (0.06, 0.03, 0.04)),
    ("+y", (0.03, 0.06, 0.04)),     # depth -> Y, lateral -> X
    ("-y", (0.03, 0.06, 0.04)),
])
def test_grid_steps_follow_camera_side(side, half_range_xyz):
    base = [0.4, 0.0, 0.3, 3.14, 0.0, 0.0]
    step = (0.03, 0.06, 0.04)                            # lateral, depth, vertical
    xyz = np.array(mf.generate_grid_poses(base, side, step, 3))[:, :3] - base[:3]
    assert np.allclose(xyz.max(axis=0), half_range_xyz)
    assert np.allclose(xyz.min(axis=0), -np.array(half_range_xyz))


def test_grid_first_move_is_lateral():
    # outermost vertical, then depth, then lateral, so the first move is along lateral.
    offsets = mf.grid_offsets_view((0.05, 0.05, 0.05), 3)
    d = np.subtract(offsets[1], offsets[0])
    assert np.allclose(d, [0.05, 0.0, 0.0])
