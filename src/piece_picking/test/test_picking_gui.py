"""Run the picking GUI end to end with a simulated robot: select result -> compute target -> move to product -> return."""

import copy
import os
import time

import numpy as np
import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication  # noqa: E402

from piece_picking import picking_gui  # noqa: E402
from piece_picking import picking_settings as settings  # noqa: E402
from piece_picking.vision_client import VisionClient  # noqa: E402

R_RGB = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])   # camera looking down
T_RGB = np.array([-0.26, 0.42, 1.43])
R_DEPTH, T_DEPTH = R_RGB.copy(), np.array([-0.20, 0.42, 1.43])          # ToF 60 mm next to the RGB
START = [-0.30, 0.40, 0.75, np.pi, 0.0, -0.16]


def _result(path, R, t, kind):
    path.write_text(yaml.safe_dump({"base_T_cam": {"R": R.tolist(), "t": t.tolist()},
                                    "meta": {"created": "test", "camera_kind": kind, "start_tcp_pose": START}}),
                    encoding="utf-8")
    return str(path)


_APP = QApplication.instance() or QApplication([])      # keep one for the whole test run


@pytest.fixture
def window(tmp_path):
    app = _APP
    cfg = copy.deepcopy(settings.DEFAULT_CONFIG)
    cfg["calibration"]["result_files"] = {
        "rgb": _result(tmp_path / "calibration_rgb.yaml", R_RGB, T_RGB, "rgb"),
        "depth": _result(tmp_path / "calibration_depth.yaml", R_DEPTH, T_DEPTH, "depth")}
    cfg["output"]["directory"] = str(tmp_path / "out")
    win = picking_gui.PickingWindow(settings.validate(cfg), sim=True)
    win._ask = lambda *a: True
    picking_gui.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    picking_gui.QMessageBox.information = staticmethod(lambda *a, **k: None)
    yield app, win
    win.close()


def _wait(app, win, timeout=30.0):
    end = time.time() + timeout
    while (win._running() or win.worker is None) and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def test_both_results_autoloaded_and_selection_switches_frame(window):
    app, win = window
    assert np.allclose(win.calibs["rgb"].t, T_RGB) and np.allclose(win.calibs["depth"].t, T_DEPTH)
    assert win._use() == "rgb" and win._frame() == "rgb"
    win.calib_checks["depth"].setChecked(True)
    assert win._frame() == "tof" and "ToF" in win.lbl_frame.text()


def _set_product(win, product, R, t):
    p_cam_m = R.T @ (product - t)                     # input fields are in m (vision position_m as is)
    for sp, v in zip(win.spin_point, p_cam_m):
        sp.setValue(float(v))


@pytest.mark.parametrize("use, R, t", [("rgb", R_RGB, T_RGB), ("depth", R_DEPTH, T_DEPTH)])
def test_move_to_product_and_back(window, use, R, t):
    app, win = window
    win.calib_checks[use].setChecked(True)
    assert not win.chk_apply_normal.isChecked()                          # default: tool orientation unchanged (vertical)
    win.edit_robot_ip.setText("127.0.0.1")
    win._on_connect_robot()
    win.spin_speed.setValue(250)
    product = np.array([-0.25, 0.45, 0.20])
    _set_product(win, product, R, t)                # entered in the same camera frame as the selected result
    win.spin_gripper.setValue(100)                  # simulated robot active TCP = flange (0)
    win.spin_approach.setValue(100)
    win.spin_offset.setValue(20)
    win._recompute()
    assert win.plan is not None and not win.plan_error, win.plan_error

    win._on_move("target")
    _wait(app, win)
    pose = win.robot.current_pose()
    assert np.allclose(pose[:3], product + [0, 0, 0.02 + 0.10], atol=1e-3)   # offset + gripper (tool facing down)
    assert np.allclose(pose[3:], START[3:])                   # tool orientation kept
    assert np.allclose(win.return_pose[:3], START[:3])

    win.worker = None
    win._on_return()
    _wait(app, win)
    assert np.allclose(win.robot.current_pose()[:3], START[:3], atol=1e-3)


def test_gripper_length_applied_to_motion(window):
    """Gripper 150 mm: the robot TCP stands that much higher and the suction tip ends 20 mm above the product."""
    app, win = window
    win.edit_robot_ip.setText("127.0.0.1")
    win._on_connect_robot()
    win.spin_speed.setValue(250)
    win.spin_gripper.setValue(0)
    win._recompute()                                                          # (the GUI does this with a 0.25 s timer)
    assert "collides" in win.lbl_gripper.text()                              # 0 + flange TCP = danger warning
    assert "Gripper length is 0" in win.plan_error                            # ... and motion is blocked
    assert not win.btn_target.isEnabled() and not win.btn_cycle.isEnabled()
    win.spin_gripper.setValue(150)
    assert "Warning" not in win.lbl_gripper.text()
    product = np.array([-0.25, 0.45, 0.20])
    _set_product(win, product, R_RGB, T_RGB)
    win.spin_offset.setValue(20)
    win._recompute()
    assert win.plan is not None and not win.plan_error, win.plan_error
    win._on_move("target")
    _wait(app, win)
    tcp = np.array(win.robot.current_pose()[:3])
    assert np.allclose(tcp, product + [0, 0, 0.02 + 0.15], atol=1e-3)       # tool facing down
    assert np.allclose(win.plan.tip_target, product + [0, 0, 0.02])
    assert win.cfg["pick"]["gripper_length_mm"] == 150.0                     # input saved


def _connect(win):
    win.edit_robot_ip.setText("127.0.0.1")
    win._on_connect_robot()
    win.spin_speed.setValue(250)


def _set_normal(win, n_cam):
    for sp, v in zip(win.spin_normal, n_cam):
        sp.setValue(float(v))


def test_tilted_product_normal(window):
    """Normal of a face tilted 20 deg in the base: with Apply rotation the tool tilts 20 deg, the suction tip lands on the product, the TCP backs off along the tool axis."""
    from robot_camera_calibration.pose_math import rotation_from_pose
    app, win = window
    _connect(win)
    product = np.array([-0.25, 0.45, 0.25])
    _set_product(win, product, R_RGB, T_RGB)
    d = np.array([0.0, np.sin(np.radians(20)), -np.cos(np.radians(20))])     # suction direction (base)
    _set_normal(win, R_RGB.T @ d)                                             # normal in the camera frame
    win.spin_gripper.setValue(100)
    win.spin_approach.setValue(50)
    win.spin_offset.setValue(0)
    win._recompute()
    assert win.plan.mode == "keep" and "Apply rotation is off" in win.lbl_target.text()   # before ticking: vertical
    assert "20.0° from vertical" in win.lbl_target.text()                         # the normal is still shown
    win.chk_apply_normal.setChecked(True)
    assert win.plan is not None and not win.plan_error, win.plan_error
    assert win.plan.mode == "tilt" and win.plan.tilt_deg == pytest.approx(20.0, abs=1e-3)
    win._on_move("target")
    _wait(app, win)
    pose = win.robot.current_pose()
    R = rotation_from_pose(pose, "rpy")
    assert np.allclose(R[:, 2], d, atol=1e-4)
    assert np.allclose(np.array(pose[:3]) + R[:, 2] * 0.1, product, atol=1e-3)   # suction tip = product
    assert win.cfg["pick"]["apply_normal"] is True                               # input saved
    assert np.allclose(win.cfg["vision"]["manual_normal"], R_RGB.T @ d, atol=1e-6)


def test_apply_without_normal_picks_vertically(window):
    app, win = window
    _connect(win)
    win.chk_apply_normal.setChecked(True)
    _set_normal(win, [0, 0, 0])
    win._recompute()
    assert win.plan.mode == "keep" and "no normal" in win.lbl_target.text()
    assert np.allclose(win.plan.target_pose[3:], START[3:])


def test_tool_reference_is_home_when_starting_from_home(window):
    """When starting from home, the tool orientation reference is the home pose (the pose right before the approach) — same result wherever the robot is."""
    app, win = window
    _connect(win)
    win._on_save_pose("home")
    elsewhere = [-0.1, 0.5, 0.6, np.pi - 0.4, 0.2, 0.9]
    assert win._reference_pose(elsewhere) == (list(win.poses["home"]["tcp_pose"]), "home")
    win.chk_start_home.setChecked(False)
    assert win._reference_pose(elsewhere) == (elsewhere, "current")


def test_pick_place_home_cycle(window, tmp_path, monkeypatch):
    """Save home / place -> pick -> place -> home. Approach and retreat along the tool axis. Grip = DO4~7 OFF, release = ON."""
    app, win = window
    events = []
    monkeypatch.setattr(picking_gui.ServoMover, "digital_out",
                        lambda self, outs, level: events.append(("do", list(outs), level)) or True)
    _connect(win)
    win._on_save_pose("home")                                   # current (START) pose = home
    place = [-0.05, 0.55, 0.30, np.pi, 0.0, -0.16]
    win.poses = settings.save_pose(win.cfg, "place", place, None, [0] * 6)
    assert (tmp_path / "out" / "poses.yaml").exists()
    product = np.array([-0.25, 0.45, 0.25])
    _set_product(win, product, R_RGB, T_RGB)
    win.spin_gripper.setValue(150)
    win.spin_approach.setValue(100)
    win.spin_offset.setValue(0)
    win._recompute()
    assert win.btn_cycle.isEnabled(), win.plan_error

    visited = []
    real_move = win.robot.move_linear
    def move(p, a, v):
        visited.append(np.round(p[:3], 4).tolist())
        events.append(("move", len(visited) - 1))
        return real_move(p, a, v)
    win.robot.move_linear = move
    win._on_cycle()
    _wait(app, win, timeout=60)
    assert "Done" in win.log_box.toPlainText().splitlines()[-1]
    up = 0.15 + 0.10                                             # gripper + approach (tool facing down)
    back = 0.15 + 0.05                                           # gripper + retreat 50 mm along the tool
    expected = [product + [0, 0, up], product + [0, 0, 0.15], product + [0, 0, back], product + [0, 0, back + 0.20],
                np.add(place[:3], [0, 0, 0.10]), place[:3], np.add(place[:3], [0, 0, 0.10]), START[:3]]
    assert len(visited) == len(expected)
    for got, exp in zip(visited, expected):
        assert np.allclose(got, exp, atol=1e-3), (got, exp)
    assert np.allclose(win.robot.current_pose()[:3], START[:3], atol=1e-3)   # back home
    # OFF (grip) after reaching the product (1), then retreat (2) and lift (3); ON (release) after placing (5)
    assert events == [("move", 0), ("move", 1), ("do", [4, 5, 6, 7], False), ("move", 2), ("move", 3),
                      ("move", 4), ("move", 5), ("do", [4, 5, 6, 7], True), ("move", 6), ("move", 7)]
    log = win.log_box.toPlainText()
    assert "Grip: DO4, DO5, DO6, DO7 = OFF" in log and "Release: DO4, DO5, DO6, DO7 = ON" in log


def test_pick_standby_comes_before_the_approach(window, monkeypatch):
    """Home -> pick standby -> approach (the tool turns only on this move) -> product. The tilt starts from the standby orientation."""
    from robot_camera_calibration.pose_math import rotation_from_pose
    app, win = window
    monkeypatch.setattr(picking_gui.ServoMover, "digital_out", lambda self, outs, level: True)
    _connect(win)
    win._on_save_pose("home")                                   # START = home
    standby = [-0.25, 0.45, 0.60, np.pi, 0.0, 0.5]              # above the box, wrist turned differently from home
    win.poses = settings.save_pose(win.cfg, "standby", standby, None, [0] * 6)
    win.poses = settings.save_pose(win.cfg, "place", [-0.05, 0.55, 0.30, np.pi, 0.0, -0.16], None, [0] * 6)
    product = np.array([-0.25, 0.45, 0.25])
    _set_product(win, product, R_RGB, T_RGB)
    d = np.array([0.0, np.sin(np.radians(20)), -np.cos(np.radians(20))])
    _set_normal(win, R_RGB.T @ d)
    win.chk_apply_normal.setChecked(True)
    win.spin_gripper.setValue(150)
    win.spin_approach.setValue(100)
    win.spin_offset.setValue(0)
    win._recompute()
    assert win._reference_pose(START)[1] == "pick standby"
    assert "rotation from the pick standby pose" in win.lbl_target.text()
    assert win.plan.rotation_change_deg == pytest.approx(20.0, abs=1e-3)       # only the tilt, standby RZ kept

    poses = []
    real_move = win.robot.move_linear
    win.robot.move_linear = lambda p, a, v: (poses.append(list(p)), real_move(p, a, v))[1]
    win._on_cycle()
    _wait(app, win, timeout=60)
    assert "Done" in win.log_box.toPlainText().splitlines()[-1]
    assert np.allclose(poses[0], standby, atol=1e-6)                           # 1st: pick standby as saved
    assert np.allclose(poses[1][:3], win.plan.approach_pose[:3], atol=1e-6)    # 2nd: approach, already tilted
    assert np.allclose(rotation_from_pose(poses[1], "rpy")[:, 2], d, atol=1e-4)
    assert np.allclose(poses[2][3:], poses[1][3:])                             # approach -> product: no rotation
    # after the pick: 50 mm back along the (tilted) tool axis, then 200 mm straight up, tool still tilted
    assert np.allclose(np.subtract(poses[3][:3], poses[2][:3]), -d * 0.05, atol=1e-6)
    assert np.allclose(np.subtract(poses[4][:3], poses[3][:3]), [0, 0, 0.20], atol=1e-6)
    assert np.allclose(poses[3][3:], poses[2][3:]) and np.allclose(poses[4][3:], poses[2][3:])


def test_manual_suction_buttons(window, monkeypatch):
    app, win = window
    calls = []
    monkeypatch.setattr(picking_gui.ServoMover, "digital_out",
                        lambda self, outs, level: calls.append((list(outs), level)) or True)
    assert "DO4, DO5, DO6, DO7" in win.lbl_suction.text() and not win.btn_grip.isEnabled()
    _connect(win)
    win._update_buttons()                                     # (the GUI does this with a timer)
    assert win.btn_grip.isEnabled()
    assert win.btn_grip.text() == "Grip (OFF)" and win.btn_release.text() == "Release (ON)"
    win.btn_grip.click()
    _wait(app, win)
    win.worker = None
    win.btn_release.click()
    _wait(app, win)
    assert calls == [([4, 5, 6, 7], False), ([4, 5, 6, 7], True)]


def test_suction_config_validation():
    cfg = copy.deepcopy(settings.DEFAULT_CONFIG)
    assert settings.suction_outputs(cfg) == [4, 5, 6, 7]
    cfg["gripper"]["suction_do"] = 3
    assert settings.suction_outputs(settings.validate(cfg)) == [3]
    cfg["gripper"]["suction_do"] = None
    assert settings.suction_outputs(settings.validate(cfg)) == []
    for bad in (["4"], [-1], [True]):
        cfg["gripper"]["suction_do"] = bad
        with pytest.raises(ValueError):
            settings.validate(cfg)


def test_missing_result_blocks_motion(window, tmp_path):
    app, win = window
    win.sim = False                                           # without simulated fallback values
    win._load_calibration("rgb", tmp_path / "nope.yaml")
    win.calib_checks["rgb"].setChecked(True)
    win._recompute()
    assert win.plan is None and "missing" in win.plan_error
    assert not win.btn_target.isEnabled()


CANDIDATE = """{
    "object_id": 3,
    "rank": 3,
    "position_m": [
      0.011000001803040504,
      -0.1457500010728836,
      1.366000056267847
    ],
    "normal": [
      0.006712059490382671,
      -0.02153539098799228,
      0.999745477714539
    ],
    "uv": [1033, 678],
    "source_index": 123209
  },"""


def test_paste_vision_json_fills_m_and_normal(window, monkeypatch):
    app, win = window
    _connect(win)
    QApplication.clipboard().setText("")                        # with an empty clipboard the paste dialog opens
    monkeypatch.setattr(win, "_ask_json_text", lambda text: CANDIDATE)
    win.btn_paste.click()
    assert [sp.value() for sp in win.spin_point] == pytest.approx([0.011, -0.14575, 1.366], abs=1e-9)
    assert [sp.value() for sp in win.spin_normal] == pytest.approx([0.006712, -0.021535, 0.999745], abs=1e-9)
    assert win.spin_point[0].suffix() == " m"
    assert np.allclose(win._point_mm(), [11.0, -145.75, 1366.0])
    assert "object_id 3, rank 3" in win.log_box.toPlainText()
    assert win.plan is not None and "normal → approach direction" in win.lbl_target.text()


def test_paste_several_candidates_asks_which(window, monkeypatch):
    app, win = window
    text = ('[{"object_id": 7, "rank": 2, "position_m": [0.1, 0.0, 1.2], "normal": [0, 0, 1]},'
            ' {"object_id": 5, "rank": 1, "position_m": [-0.05, 0.02, 1.3]}]')
    monkeypatch.setattr(win, "_ask_json_text", lambda t: text)
    QApplication.clipboard().setText("")
    shown = {}

    def get_item(parent, title, label, items, current, editable):
        shown["items"], shown["current"] = list(items), current
        return items[1], True                                    # choose the second one (rank 2)
    monkeypatch.setattr(picking_gui.QInputDialog, "getItem", staticmethod(get_item))
    win._on_paste_json()
    assert shown["current"] == 0 and shown["items"][0].startswith("object_id 5, rank 1")   # rank 1 is the default
    assert [sp.value() for sp in win.spin_point] == pytest.approx([0.1, 0.0, 1.2])


def test_trigger_mode_is_todo(window):
    app, win = window
    win.chk_trigger.setChecked(True)
    assert win.btn_capture.isEnabled()
    assert win.spin_point[0].isReadOnly() and not win.btn_paste.isEnabled()
    with pytest.raises(NotImplementedError):
        VisionClient().trigger()
    win._on_capture()                                         # must only log a message and return
    assert "TODO" in win.log_box.toPlainText()
