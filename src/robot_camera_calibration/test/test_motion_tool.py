"""Run the motion_tool commands end to end with the simulated robot."""

import copy

import numpy as np
import pytest

from robot_camera_calibration import motion_tool as mt
from robot_camera_calibration import settings
from robot_camera_calibration.sim_robot import SimRobot

START = [0.45, 0.0, 0.30, 3.1416, 0.0, 0.0]


def _cfg(side="+x"):
    cfg = copy.deepcopy(settings.DEFAULT_CONFIG)
    cfg["sequence"]["camera_side"] = side
    cfg["robot"]["velocity"] = 2.0             # to finish the test quickly
    return cfg


def _yes(_text):
    return True


def test_config_file_loads_and_matches_defaults():
    cfg = mt.load_config(mt.default_config_path())
    assert cfg["sequence"]["camera_side"] in ("+x", "-x", "+y", "-y")
    assert set(cfg["sequence"]["step_mm"]) == {"lateral", "depth", "vertical"}


def test_old_xyz_step_list_is_rejected(tmp_path):
    p = tmp_path / "old.yaml"
    p.write_text("sequence:\n  step_mm: [50, 50, 50]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        mt.load_config(p)


def test_camera_side_override():
    cfg = mt.load_config(None, camera_side="-Y")
    assert cfg["sequence"]["camera_side"] == "-y"


@pytest.mark.parametrize("side, expected", [("+x", [0.02, 0, 0]), ("+y", [0, 0.02, 0]),
                                            ("-x", [-0.02, 0, 0]), ("-y", [0, -0.02, 0])])
def test_step_depth_moves_toward_camera(side, expected):
    robot = SimRobot(START)
    assert mt.cmd_step(_cfg(side), robot, "depth", 20.0, confirm=_yes, out=lambda *_: None) == 0
    assert np.allclose(np.subtract(robot.current_pose()[:3], START[:3]), expected, atol=1e-9)


def test_step_limit():
    robot = SimRobot(START)
    assert mt.cmd_step(_cfg(), robot, "lateral", 150.0, confirm=_yes, out=lambda *_: None) == 1
    assert robot.current_pose() == START


def test_step_cancelled_does_not_move():
    robot = SimRobot(START)
    assert mt.cmd_step(_cfg(), robot, "vertical", 10.0, confirm=lambda _t: False, out=lambda *_: None) == 1
    assert robot.current_pose() == START


def test_run_visits_grid_and_returns():
    robot = SimRobot(START)
    visited = []
    real_move = robot.move_linear

    def record(pose, a, v):
        visited.append(list(pose))
        return real_move(pose, a, v)

    robot.move_linear = record
    cfg = _cfg("+y")
    cfg["sequence"]["step_mm"] = {"lateral": 30.0, "depth": 60.0, "vertical": 40.0}
    assert mt.cmd_run(cfg, robot, confirm=_yes, out=lambda *_: None) == 0
    grid = np.array(visited[:27])[:, :3] - START[:3]
    assert np.allclose(grid.max(axis=0), [0.03, 0.06, 0.04])   # +y: lateral -> X, depth -> Y
    assert np.allclose(visited[-1], START)                      # back at the start pose
