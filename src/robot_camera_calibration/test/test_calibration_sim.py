"""Run the whole calibration sequence with a simulated robot + simulated Helios (no GUI)."""

import copy
import os

import numpy as np
import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication  # noqa: E402

from robot_camera_calibration import calib_core as core  # noqa: E402
from robot_camera_calibration import calib_gui  # noqa: E402
from robot_camera_calibration import pose_math as pm  # noqa: E402
from robot_camera_calibration import settings  # noqa: E402
from robot_camera_calibration.sim_camera import create_simulation  # noqa: E402

# share one Qt app with the GUI tests in the same process and keep it until the end.
_APP = QApplication.instance() or QApplication([])


def _fast_config(tmp_path, side, offset_mm):
    cfg = copy.deepcopy(settings.DEFAULT_CONFIG)
    cfg["sequence"]["camera_side"] = side
    cfg["sequence"]["frames_per_pose"] = 5
    cfg["marker"]["offset_from_flange_mm"] = offset_mm
    cfg["robot"]["velocity"] = 1.0
    cfg["robot"]["settle_time_s"] = 0.02
    cfg["output"]["directory"] = str(tmp_path)
    cfg["output"]["save_images"] = False
    return settings.validate(cfg)


def _run(cfg, sim_side=None, true_offset_mm=None, kind="depth", start_shift_m=None):
    assert _APP is not None
    robot, camera, _, (R_true, t_true) = create_simulation(cfg, sim_side, true_offset_mm, kind=kind)
    start_target = None
    if start_shift_m is not None:            # saved start pose = slightly away from the current pose
        start_target = list(robot.current_pose())
        start_target[:3] = list(np.add(start_target[:3], start_shift_m))
        robot._worker_start_target = start_target
    camera.start()
    try:
        det = core.MarkerDetector(cfg["marker"]["dictionary"], cfg["marker"]["id"])
        worker = calib_gui.CalibrationWorker(robot, camera, det, cfg, kind, start_target=start_target)
        out = {}
        worker.finished_run.connect(lambda ok, msg: out.update(ok=ok, msg=msg))
        worker.run()            # run in this thread (signals are delivered immediately)
        return out, worker, R_true, t_true
    finally:
        camera.stop()


@pytest.mark.parametrize("side", ["+x", "-x", "+y", "-y"])
def test_calibration_recovers_camera_pose_for_each_side(tmp_path, side):
    offset = [12.0, -30.0, 45.0]
    cfg = _fast_config(tmp_path, side, offset)
    out, worker, R_true, t_true = _run(cfg)
    assert out["ok"], out["msg"]
    data = yaml.safe_load(open(worker.result_path))
    R = np.array(data["base_T_cam"]["R"])
    t = np.array(data["base_T_cam"]["t"])
    assert pm.rotation_angle_deg(R, R_true) < 0.5
    assert np.linalg.norm(t - t_true) * 1000 < 3.0
    assert data["meta"]["samples_valid"] == 27
    assert data["meta"]["grid"]["camera_side"] == side
    assert "warning" not in data["meta"]
    assert (worker.run_dir / "samples.yaml").exists()


@pytest.mark.parametrize("side", ["+y", "-x"])
def test_rgb_calibration_by_pnp(tmp_path, side):
    """RGB (no depth): find base_T_rgb by PnP of the marker base coordinates known to the robot <-> image pixels."""
    offset = [12.0, -30.0, 45.0]
    cfg = _fast_config(tmp_path, side, offset)
    truth_rot = create_simulation(cfg, kind="rgb")[1].true_marker_rotation_deg
    cfg["marker"]["rotation_deg"] = [float(v) for v in truth_rot]
    out, worker, R_true, t_true = _run(cfg, kind="rgb")
    assert out["ok"], out["msg"]
    assert worker.result_path == tmp_path / "calibration_rgb.yaml"
    assert not (tmp_path / "calibration_depth.yaml").exists()
    data = yaml.safe_load(open(worker.result_path))
    R = np.array(data["base_T_cam"]["R"])
    t = np.array(data["base_T_cam"]["t"])
    assert pm.rotation_angle_deg(R, R_true) < 0.5
    assert np.linalg.norm(t - t_true) * 1000 < 3.0
    meta = data["meta"]
    assert meta["camera_kind"] == "rgb" and meta["samples_valid"] == 27
    assert meta["rms_reprojection_px"] < 1.0
    assert meta["marker"]["rotation_diff_deg"] < 2.0                  # marker orientation measured from the 2D corners
    assert np.array(meta["camera"]["camera_matrix"]).shape == (3, 3)


def test_calibration_starts_from_saved_pose(tmp_path):
    """If the saved start pose differs from the current pose, go there first and visit the grid around it."""
    cfg = _fast_config(tmp_path, "+y", [0.0, 0.0, 30.0])
    out, worker, R_true, t_true = _run(cfg, start_shift_m=[0.03, -0.02, 0.01])
    assert out["ok"], out["msg"]
    data = yaml.safe_load(open(worker.result_path))
    assert np.allclose(data["meta"]["start_tcp_pose"][:3], worker.start_target[:3], atol=1e-3)
    assert np.linalg.norm(np.array(data["base_T_cam"]["t"]) - t_true) * 1000 < 3.0
    assert any("Moving to the saved start pose" in line for line in worker._lines)


def test_depth_and_rgb_results_are_separate_files(tmp_path):
    cfg = _fast_config(tmp_path, "+y", [0.0, 0.0, 30.0])
    assert settings.result_path(cfg, "rgb").name == "calibration_rgb.yaml"
    assert settings.result_path(cfg, "depth").name == "calibration_depth.yaml"
    out, worker, _, _ = _run(cfg)
    assert out["ok"], out["msg"]
    assert worker.result_path == tmp_path / "calibration_depth.yaml"
    assert yaml.safe_load(open(worker.result_path))["meta"]["camera_kind"] == "depth"


def test_marker_rotation_is_measured_and_checked(tmp_path):
    cfg = _fast_config(tmp_path, "+y", [12.0, -30.0, 45.0])
    truth = create_simulation(cfg)[1].true_marker_rotation_deg
    cfg["marker"]["rotation_deg"] = [float(truth[0]), float(truth[1]), float(truth[2]) + 10.0]   # RZ off by 10 deg
    out, worker, _, _ = _run(cfg)
    assert out["ok"], out["msg"]
    marker = yaml.safe_load(open(worker.result_path))["meta"]["marker"]
    measured = pm.rpy_deg_to_matrix(marker["rotation_measured_deg"])
    assert pm.rotation_angle_deg(measured, pm.rpy_deg_to_matrix(truth)) < 1.0
    assert 9.0 < marker["rotation_diff_deg"] < 11.0
    assert "rotation_warning" in marker


def test_wrong_side_still_fits_but_warns(tmp_path):
    cfg = _fast_config(tmp_path, "+x", [0.0, 0.0, 30.0])
    out, worker, R_true, t_true = _run(cfg, sim_side="-y")     # the real camera is on -y
    assert out["ok"], out["msg"]
    data = yaml.safe_load(open(worker.result_path))
    assert np.linalg.norm(np.array(data["base_T_cam"]["t"]) - t_true) * 1000 < 3.0
    assert "-Y" in data["meta"]["warning"] or "−Y" in data["meta"]["warning"]


def test_ui_state_roundtrip(tmp_path):
    cfg = _fast_config(tmp_path, "+x", [1.0, 2.0, 3.0])
    cfg["robot"]["ip"] = "10.0.0.5"
    cfg["sequence"]["camera_side"] = "-y"
    settings.save_ui_state(cfg)
    fresh = _fast_config(tmp_path, "+x", [0.0, 0.0, 0.0])
    assert settings.apply_ui_state(fresh)
    assert fresh["robot"]["ip"] == "10.0.0.5"
    assert fresh["sequence"]["camera_side"] == "-y"
    assert fresh["marker"]["offset_from_flange_mm"] == [1.0, 2.0, 3.0]
