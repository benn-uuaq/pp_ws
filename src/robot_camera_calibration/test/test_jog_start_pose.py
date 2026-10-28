"""Jog and start pose save / move (simulated robot, no screen)."""

import copy
import math
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication  # noqa: E402

from robot_camera_calibration import calib_gui  # noqa: E402
from robot_camera_calibration import motion_frame as mf  # noqa: E402
from robot_camera_calibration import pose_math as pm  # noqa: E402
from robot_camera_calibration import settings  # noqa: E402

# keep one Qt app until the module ends. If it were deleted per test, other tests' QObjects would disappear with it.
_APP = QApplication.instance() or QApplication([])


# ---- pose math -------------------------------------------------------------------
@pytest.mark.parametrize("conv", ["rpy", "rotvec"])
def test_rotate_pose_about_base(conv):
    pose = [0.4, 0.3, 0.6, 3.0, 0.1, -0.4]
    out = pm.rotate_pose_about_base(pose, 2, math.radians(10.0), conv)
    assert out[:3] == pose[:3]                                    # TCP position unchanged
    R0, R1 = pm.rotation_from_pose(pose, conv), pm.rotation_from_pose(out, conv)
    assert np.allclose(R1, pm.rpy_to_matrix(0, 0, math.radians(10.0)) @ R0, atol=1e-9)


def test_rotvec_roundtrip_including_180_deg():
    rng = np.random.default_rng(0)
    rots = [pm.rpy_to_matrix(*rng.uniform(-3, 3, 3)) for _ in range(20)]
    rots.append(pm.rpy_to_matrix(math.pi, 0, 0))                   # common pose with the tool facing down
    for R in rots:
        assert np.allclose(pm.rotvec_to_matrix(pm.matrix_to_rotvec(R)), R, atol=1e-6)


def test_start_pose_save_and_load(tmp_path):
    cfg = copy.deepcopy(settings.DEFAULT_CONFIG)
    cfg["output"]["directory"] = str(tmp_path)
    assert settings.load_start_pose(cfg) is None
    settings.save_start_pose(cfg, [0.1, 0.2, 0.3, 3.1, 0, 0], [0] * 6, [0.1] * 6)
    sp = settings.load_start_pose(cfg)
    assert sp["tcp_pose"] == [0.1, 0.2, 0.3, 3.1, 0, 0] and sp["joints"] == [0.1] * 6 and sp["saved"]


# ---- GUI -------------------------------------------------------------------------
@pytest.fixture
def window(tmp_path):
    app = _APP
    cfg = copy.deepcopy(settings.DEFAULT_CONFIG)
    cfg["active_camera"] = "depth"
    cfg["sequence"]["camera_side"] = "+y"
    cfg["output"]["directory"] = str(tmp_path)
    win = calib_gui.CalibrationWindow(settings.validate(cfg), sim=True)
    win._ask = lambda *a: True
    calib_gui.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    win._on_connect_robot()
    yield app, win
    win.close()


def _pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def _wait_still(app, robot, timeout=10.0):
    end = time.time() + timeout
    _pump(app, 0.05)
    while robot.get_state().is_moving() and time.time() < end:
        _pump(app, 0.02)


def test_jog_labels_follow_camera_side(window):
    app, win = window
    assert "(+Y)" in win.jog_lin_buttons[("depth", 1)].text()          # +y: towards camera = +Y
    win.side_checks["-x"].setChecked(True)
    assert "(-X)" in win.jog_lin_buttons[("depth", 1)].text()


def test_continuous_jog_moves_while_pressed(window):
    app, win = window
    win.combo_jog.setCurrentIndex(0)                                   # continuous
    win.spin_jog_speed.setValue(20)
    start = np.array(win.robot.current_pose()[:3])
    button = win.jog_lin_buttons[("lateral", 1)]                       # right (lateral +)
    button.setDown(True)
    button.pressed.emit()
    assert win._jog_active
    _pump(app, 0.5)
    button.setDown(False)
    button.released.emit()
    assert not win._jog_active
    moved = np.array(win.robot.current_pose()[:3]) - start
    lateral = mf.view_axes("+y")[:, 0]
    assert 7.0 < float(moved @ lateral) * 1000 < 13.0                  # about 0.5 s x 20 mm/s
    assert np.linalg.norm(moved - lateral * (moved @ lateral)) < 1e-6  # no motion in other directions
    _pump(app, 0.3)
    assert np.allclose(win.robot.current_pose()[:3], start + moved)   # stopped after release


def test_watchdog_stops_jog_when_release_is_missed(window):
    """If a continuous jog runs while the button is not pressed (release missed), it stops within 0.1 s."""
    app, win = window
    win.combo_jog.setCurrentIndex(0)
    win._on_jog_pressed("lin", 2, +1)                                 # button not pressed
    assert win._jog_active
    _pump(app, 0.3)
    assert not win._jog_active
    p = win.robot.current_pose()
    _pump(app, 0.2)
    assert np.allclose(win.robot.current_pose(), p)


def test_step_jog_moves_exact_distance(window):
    app, win = window
    win.combo_jog.setCurrentIndex(2)                                   # step 5 mm
    win.spin_jog_speed.setValue(100)
    start = np.array(win.robot.current_pose()[:3])
    win._on_jog_pressed("lin", 1, -1)                                 # away from the camera (depth -)
    win._on_jog_released()
    _wait_still(app, win.robot)
    moved = np.array(win.robot.current_pose()[:3]) - start
    assert np.allclose(moved, -mf.view_axes("+y")[:, 1] * 0.005, atol=1e-6)


def test_save_start_and_go_back(window, tmp_path):
    app, win = window
    home = list(win.robot.current_pose())
    win._on_save_start()
    assert (tmp_path / "start_pose.yaml").exists()
    assert np.allclose(win.start_pose["tcp_pose"], home)
    win.combo_jog.setCurrentIndex(3)                                   # 10 mm
    win.spin_jog_speed.setValue(100)
    win._on_jog_pressed("lin", 2, +1)
    _wait_still(app, win.robot)
    assert np.linalg.norm(np.subtract(win.robot.current_pose()[:3], home[:3])) > 0.009
    win._on_goto_start()
    _wait_still(app, win.robot)
    assert np.allclose(win.robot.current_pose()[:3], home[:3], atol=1e-6)


def test_stop_button_stops_jog_motion(window):
    app, win = window
    win.combo_jog.setCurrentIndex(0)
    win._on_jog_pressed("lin", 2, +1)
    _pump(app, 0.2)
    win._on_stop()
    p = win.robot.current_pose()
    _pump(app, 0.3)
    assert np.allclose(win.robot.current_pose(), p)
