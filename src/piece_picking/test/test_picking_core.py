"""picking_core: loading calibrations, coordinate conversion, picking target, safety checks."""

import cv2 as cv
import numpy as np
import pytest
import yaml

from piece_picking import picking_core as pc


def _rot(rx, ry, rz):
    R, _ = cv.Rodrigues(np.array([rx, ry, rz], dtype=float))
    return R


R_CAM = _rot(np.pi, 0.02, -0.03)              # camera looking down
T_CAM = np.array([-0.20, 0.42, 1.43])


def _write_result(path, R=R_CAM, t=T_CAM, kind="depth"):
    meta = {"created": "test", "rms_error_mm": 1.2, "max_error_mm": 2.0,
            "samples_used": 27, "samples_total": 27, "grid": {"camera_side": "+y"}}
    if kind is not None:
        meta["camera_kind"] = kind
    path.write_text(yaml.safe_dump({"base_T_cam": {"R": R.tolist(), "t": t.tolist()}, "meta": meta}),
                    encoding="utf-8")
    return path


def _write_orientation(path, R, t_mm):
    fs = cv.FileStorage(str(path), cv.FileStorage_WRITE)
    fs.write("cameraMatrix", np.eye(3))
    fs.write("rotationVector", cv.Rodrigues(R)[0])
    fs.write("translationVector", np.asarray(t_mm, float).reshape(3, 1))
    fs.release()
    return path


def test_rgb_result_with_rgb_point(tmp_path):
    calib = pc.load_calibration(_write_result(tmp_path / "calibration_rgb.yaml", kind="rgb"), expected="rgb")
    assert calib.camera == "rgb" and calib.frame == "rgb"
    assert any("RMS 1.20" in line for line in calib.summary_lines())
    p_mm = np.array([30.0, -50.0, 900.0])
    p, how = pc.camera_point_to_base(p_mm, "rgb", calib)
    assert how == "calibration_rgb.yaml"
    assert np.allclose(p, R_CAM @ (p_mm / 1000) + T_CAM)


def test_old_file_without_kind_is_depth(tmp_path):
    calib = pc.load_calibration(_write_result(tmp_path / "aruco_rigid_result.yaml", kind=None))
    assert calib.camera == "depth" and calib.frame == "tof"


def test_wrong_kind_is_refused(tmp_path):
    with pytest.raises(ValueError):
        pc.load_calibration(_write_result(tmp_path / "x.yaml", kind="depth"), expected="rgb")


def test_rejects_non_rotation(tmp_path):
    with pytest.raises(ValueError):
        pc.load_calibration(_write_result(tmp_path / "bad.yaml", R=R_CAM * 1.01))


def test_frame_mismatch_needs_orientation(tmp_path):
    calib = pc.load_calibration(_write_result(tmp_path / "d.yaml", kind="depth"))
    with pytest.raises(ValueError):
        pc.camera_point_to_base([0, 0, 1000], "rgb", calib, None)
    # with the LUCID orientation (Helios mm -> Triton mm), an RGB input can also use the depth result
    R_o, t_o = _rot(0.01, -0.2, 0.005), np.array([-60.0, 5.0, 12.0])
    orient = pc.load_orientation(_write_orientation(tmp_path / "orientation.yml", R_o, t_o))
    p_tof_mm = np.array([10.0, 20.0, 1100.0])
    p, how = pc.camera_point_to_base(R_o @ p_tof_mm + t_o, "rgb", calib, orient)
    assert "orientation" in how
    assert np.allclose(p, R_CAM @ (p_tof_mm / 1000) + T_CAM, atol=1e-9)
    assert np.allclose(orient.tof_to_rgb_m(orient.rgb_to_tof_m([0.1, 0.2, 1.0])), [0.1, 0.2, 1.0])


def test_plan_keeps_tool_orientation_and_approaches_from_above():
    tcp = [-0.30, 0.40, 0.75, np.pi, 0.0, -0.16]
    plan = pc.plan_pick([-0.25, 0.45, 0.20], tcp, approach_mm=100, final_offset_mm=20)
    assert np.allclose(plan.target_pose[:3], [-0.25, 0.45, 0.22])
    assert np.allclose(plan.approach_pose[:3], [-0.25, 0.45, 0.32])
    assert plan.target_pose[3:] == tcp[3:] and plan.approach_pose[3:] == tcp[3:]


def test_gripper_length_with_tool_pointing_down():
    tcp = [-0.30, 0.40, 0.75, np.pi, 0.0, -0.16]                      # tool +Z = base -Z (down)
    plan = pc.plan_pick([-0.25, 0.45, 0.20], tcp, approach_mm=100, final_offset_mm=20, gripper_mm=150)
    assert np.allclose(plan.tip_target, [-0.25, 0.45, 0.22])          # suction tip 20 mm above the product
    assert np.allclose(plan.target_pose[:3], [-0.25, 0.45, 0.37])     # TCP 150 mm (gripper) higher
    assert np.allclose(plan.approach_pose[:3], [-0.25, 0.45, 0.47])
    assert plan.target_pose[3:] == tcp[3:]


def test_gripper_length_with_tilted_tool():
    from robot_camera_calibration.pose_math import rotation_from_pose
    tcp = [-0.30, 0.40, 0.75, np.pi - 0.3, 0.25, 0.4]                 # tilted tool
    plan = pc.plan_pick([-0.25, 0.45, 0.20], tcp, 100, 0, gripper_mm=180)
    R = rotation_from_pose(plan.target_pose, "rpy")
    tip = np.array(plan.target_pose[:3]) + R @ [0, 0, 0.18]           # 180 mm from the TCP along tool +Z
    assert np.allclose(tip, [-0.25, 0.45, 0.20], atol=1e-9)
    # approach along the tool axis (100 mm against the suction direction)
    assert np.allclose(np.array(plan.approach_pose[:3]) + R @ [0, 0, 0.18], tip - R[:, 2] * 0.1)


def _down_tilted(deg, about="x"):
    """Straight down (base -Z) tilted by deg about base X (or Y)."""
    a = np.radians(deg)
    return np.array([0.0, np.sin(a), -np.cos(a)]) if about == "x" else np.array([np.sin(a), 0.0, -np.cos(a)])


def test_normal_is_rotated_only_and_sign_is_fixed(tmp_path):
    calib = pc.load_calibration(_write_result(tmp_path / "rgb.yaml", kind="rgb"), expected="rgb")
    n = np.array([0.006712, -0.021535, 0.999746])                  # example vision normal (pointing away from the camera)
    p_mm = np.array([11.0, -145.75, 1366.0])
    d, flipped = pc.camera_normal_to_base(n, p_mm, "rgb", calib)
    assert not flipped and np.allclose(d, R_CAM @ (n / np.linalg.norm(n)))   # rotation only, no translation
    d2, flipped2 = pc.camera_normal_to_base(-3 * n, p_mm, "rgb", calib)       # outward normal / different length
    assert flipped2 and np.allclose(d2, d)
    assert pc.camera_normal_to_base([0, 0, 0], p_mm, "rgb", calib) == (None, False)


def test_tilt_aligns_tool_z_and_approach_goes_straight_along_tool_z():
    from robot_camera_calibration.pose_math import rotation_from_pose
    tcp = [-0.30, 0.40, 0.75, np.pi, 0.0, np.radians(-30)]              # currently straight down
    d = _down_tilted(20)
    p = np.array([-0.25, 0.45, 0.20])
    plan = pc.plan_pick(p, tcp, 50, 0, gripper_mm=150, approach_dir_base=d)
    R = rotation_from_pose(plan.target_pose, "rpy")
    assert plan.mode == "tilt" and np.allclose(R[:, 2], d, atol=1e-9)
    assert plan.tilt_deg == pytest.approx(20.0, abs=1e-6)
    assert plan.rotation_change_deg == pytest.approx(20.0, abs=1e-6)        # rotates only by the tilt (no wrist turn)
    assert np.allclose(np.array(plan.target_pose[:3]) + d * 0.15, p)       # suction tip = product
    assert plan.approach_pose[3:] == plan.target_pose[3:]                  # approach -> product: same orientation,
    assert np.allclose(np.subtract(plan.target_pose[:3], plan.approach_pose[:3]), d * 0.05)   # 50 mm straight along tool +Z
    out = pc.plan_pick(p, tcp, 50, 0, gripper_mm=150, approach_dir_base=d, retreat_mm=50, lift_mm=200)
    assert np.allclose(np.subtract(out.retreat_pose[:3], out.target_pose[:3]), -d * 0.05)   # back along the tool
    assert np.allclose(np.subtract(out.lift_pose[:3], out.retreat_pose[:3]), [0, 0, 0.2])   # then straight up
    assert out.lift_pose[3:] == out.target_pose[3:]                                          # still tilted
    keep = pc.plan_pick(p, tcp, 50, 0, gripper_mm=150)
    assert keep.mode == "keep" and keep.target_pose[3:] == tcp[3:] and keep.rotation_change_deg == 0.0


def test_tilt_and_rotation_limits():
    tcp = [0.0, 0.4, 0.6, np.pi, 0, 0]
    steep = pc.plan_pick([0, 0.4, 0.2], tcp, 100, 0, approach_dir_base=_down_tilted(75))
    problems = pc.check_plan(steep, tcp, {"max_tilt_deg": 60, "max_rotation_deg": 90})
    assert any("tilts" in p for p in problems)
    up = [0.0, 0.4, 0.6, 0, 0, 0]                                         # tool facing up -> must rotate 180 deg
    spun = pc.plan_pick([0, 0.4, 0.2], up, 100, 0, approach_dir_base=[0, 0, -1])
    assert any("must rotate" in p for p in pc.check_plan(spun, up, {"max_tilt_deg": 60, "max_rotation_deg": 90}))


def test_describe_direction():
    assert "almost vertical" in pc.describe_direction([0, 0, -1])
    text = pc.describe_direction(_down_tilted(-15, "y"))                # face points to +X (suction tilts towards -X)
    assert "15.0°" in text and "+X" in text


def test_z_limit_is_checked_at_the_suction_tip():
    tcp = [0.0, 0.0, 0.5, np.pi, 0, 0]
    plan = pc.plan_pick([0.1, 0.0, 0.05], tcp, 100, 0, gripper_mm=150)  # TCP Z 0.20, suction tip Z 0.05
    problems = pc.check_plan(plan, tcp, {"max_move_mm": 800, "z_min_mm": 100.0})
    assert len(problems) == 1 and "suction tip" in problems[0] and "Target point" in problems[0]


def test_gripper_tcp_warning():
    assert "twice" in pc.gripper_tcp_warning([0, 0, 0.15, 0, 0, 0], 150.0)   # pendant TCP + gripper length
    assert "collides" in pc.gripper_tcp_warning([0] * 6, 0.0)               # both 0 (danger)
    assert pc.gripper_tcp_warning([0] * 6, 150.0) == ""                     # flange TCP + gripper length
    assert pc.gripper_tcp_warning([0, 0, 0.15, 0, 0, 0], 0.0) == ""         # tool length in the pendant TCP
    assert pc.gripper_missing([0] * 6, 0.0)                                 # only this case blocks motion
    assert not pc.gripper_missing([0] * 6, 150.0)
    assert not pc.gripper_missing([0, 0, 0.15, 0, 0, 0], 0.0)


def test_check_plan_limits():
    tcp = [0.0, 0.0, 0.5, np.pi, 0, 0]
    far = pc.plan_pick([2.0, 0.0, 0.2], tcp, 100, 0)                     # 2 m away (unit mistake etc.)
    assert any("away from" in p for p in pc.check_plan(far, tcp, {"max_move_mm": 800}))
    low = pc.plan_pick([0.1, 0.0, -0.05], tcp, 100, 0)
    assert any("below the limit" in p for p in pc.check_plan(low, tcp, {"max_move_mm": 800, "z_min_mm": 0.0}))
    ok = pc.plan_pick([0.1, 0.1, 0.2], tcp, 100, 0)
    box = [[-500, 500], [-500, 500], [0, 800]]
    assert pc.check_plan(ok, tcp, {"max_move_mm": 800, "z_min_mm": 0.0, "workspace_mm": box}) == []
    out = pc.plan_pick([0.6, 0.1, 0.2], tcp, 100, 0)
    assert any("workspace" in p for p in pc.check_plan(out, tcp, {"workspace_mm": box}))
