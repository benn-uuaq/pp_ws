"""Checks calib_core (marker, 3D measurement, fit) and the Helios conversion. Runs without hardware."""

import math

import numpy as np
import pytest

from robot_camera_calibration import calib_core as core
from robot_camera_calibration import motion_frame as mf
from robot_camera_calibration import pose_math as pm
from robot_camera_calibration.camera_helios import HeliosCamera, INVALID_RAW

K = np.array([[465.0, 0, 320.0], [0, 465.0, 240.0], [0, 0, 1.0]])


def _matrix_to_rpy(R):
    pitch = math.asin(-R[2, 0])
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def test_rpy_matches_documented_rotvec_example():
    # pair from the docs: rotation vector [3.14, 1.57, 0]  <->  RPY [-2.80856, -0.16202, 0.9]
    Rv = pm.rotation_from_pose([0, 0, 0, 3.14, 1.57, 0.0], "rotvec")
    Rr = pm.rotation_from_pose([0, 0, 0, -2.80856, -0.16202, 0.9], "rpy")
    assert pm.rotation_angle_deg(Rv, Rr) < 0.01


def test_marker_point_with_and_without_tcp_offset():
    conv = "rpy"
    flange = [0.4, -0.1, 0.3, 0.2, -0.3, 1.1]
    offset = np.array([0.01, -0.02, 0.05])
    expected = pm.pose_to_matrix(flange, conv)[:3, :3] @ offset + np.array(flange[:3])
    assert np.allclose(pm.marker_point_in_base(flange, [0] * 6, offset, conv), expected, atol=1e-12)
    tcp_off = [0.0, 0.0, 0.12, 0.0, 0.0, 0.3]
    T_tcp = pm.pose_to_matrix(flange, conv) @ pm.pose_to_matrix(tcp_off, conv)
    tcp_pose = list(T_tcp[:3, 3]) + list(_matrix_to_rpy(T_tcp[:3, :3]))
    assert np.allclose(pm.marker_point_in_base(tcp_pose, tcp_off, offset, conv), expected, atol=1e-9)


def test_marker_center_is_projective_centre():
    R = pm.rpy_to_matrix(0.6, -0.4, 0.2)
    c3 = np.array([0.03, -0.02, 0.5])
    sq = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]) * 0.035
    pts = (sq @ R.T) + c3
    px = np.stack([K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2], K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]], 1)
    true_c = np.array([K[0, 0] * c3[0] / c3[2] + K[0, 2], K[1, 1] * c3[1] / c3[2] + K[1, 2]])
    assert np.linalg.norm(core.marker_center(px) - true_c) < 1e-6


def _tilted_plane_xyz(center, normal_rpy, noise_m=0.0, seed=0, w=640, h=480):
    """XYZ image of a tilted plane (marker) seen by a pinhole camera, and the marker corner pixels."""
    R = pm.rpy_to_matrix(*normal_rpy)
    n = R[:, 2]
    us, vs = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    rays = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1)
    z = (n @ center) / (rays @ n)
    xyz = rays * z[..., None]
    if noise_m:
        rng = np.random.default_rng(seed)
        xyz[..., 2] += rng.normal(0, noise_m, size=z.shape)
    sq = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]) * 0.03
    pts = sq @ R.T + center
    corners = np.stack([K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2], K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]], 1)
    return xyz.astype(np.float32), corners


def test_point_from_xyz_on_tilted_plane():
    center = np.array([0.05, -0.03, 0.7])
    xyz, corners = _tilted_plane_xyz(center, (0.4, -0.3, 0.2))
    pp = core.point_from_xyz(xyz, corners, core.marker_center(corners))
    assert pp.point is not None and pp.points_used > 200
    assert np.linalg.norm(pp.point - center) * 1000 < 0.2


def test_marker_rotation_from_tilted_plane():
    R = pm.rpy_to_matrix(0.4, -0.3, 0.2)
    xyz, corners = _tilted_plane_xyz(np.array([0.05, -0.03, 0.7]), (0.4, -0.3, 0.2), noise_m=0.001)
    pp = core.point_from_xyz(xyz, corners, core.marker_center(corners))
    # ArUco axes: X = TL->TR, Y = BL->TL, Z = towards the camera
    expected = R @ np.diag([1.0, -1.0, -1.0])
    assert pm.rotation_angle_deg(pp.rotation, expected) < 0.5


def test_marker_rotation_from_2d_corners_only():
    """RGB (no depth): marker orientation from the corner pixels only. Same without knowing the size."""
    R = pm.rpy_to_matrix(0.4, -0.3, 0.2)
    _, corners = _tilted_plane_xyz(np.array([0.05, -0.03, 0.7]), (0.4, -0.3, 0.2))
    rot = core.marker_rotation_from_corners(corners, K, None)
    assert pm.rotation_angle_deg(rot, R @ np.diag([1.0, -1.0, -1.0])) < 0.5


def test_fit_pnp_recovers_camera_and_rejects_outlier():
    rng = np.random.default_rng(2)
    R_true = pm.rpy_to_matrix(3.1, 0.02, 1.55)             # base_T_cam
    t_true = np.array([-0.20, 0.42, 1.43])
    base = np.array(mf.generate_grid_poses([-0.25, 0.45, 0.70, 0, 0, 0], "+y", (0.05, 0.05, 0.05)))[:, :3]
    cam = (base - t_true) @ R_true                          # p_cam = R^T (p_base - t)
    Kc = np.array([[1550.0, 0, 1024.0], [0, 1550.0, 768.0], [0, 0, 1.0]])
    uv = np.stack([Kc[0, 0] * cam[:, 0] / cam[:, 2] + Kc[0, 2], Kc[1, 1] * cam[:, 1] / cam[:, 2] + Kc[1, 2]], 1)
    uv += rng.normal(0, 0.2, uv.shape)
    uv[7] += [25.0, -10.0]                                   # one badly measured sample
    fit = core.fit_pnp(base, uv, Kc)
    assert not fit.inliers[7] and fit.inliers.sum() == 26
    assert pm.rotation_angle_deg(fit.R, R_true) < 0.3
    assert np.linalg.norm(fit.t - t_true) * 1000 < 5.0
    assert fit.rms_px < 0.5


@pytest.mark.parametrize("sigma", [0.6, 1.5, 3.0])
def test_marker_edge_width_measures_blur(sigma):
    """The 10-90 % edge width of a Gaussian blur σ is about 2.56 σ (+ about 1 px from the drawing itself)."""
    import cv2 as cv
    from robot_camera_calibration.sim_camera import _marker_image
    img = np.full((600, 600), 60, np.uint8)
    img[210:390, 210:390] = 230                                # white board
    img[240:360, 240:360] = _marker_image("DICT_4X4_50", 0, 120)
    blurred = cv.GaussianBlur(img, (0, 0), sigma)
    det = core.MarkerDetector("DICT_4X4_50", 0)
    corners = det.detect(cv.cvtColor(blurred, cv.COLOR_GRAY2BGR))
    width = core.marker_edge_width(blurred, corners)
    assert width is not None
    assert abs(width - 2.563 * sigma) < 1.2
    assert core.marker_edge_width(blurred, corners * 0.1) is None   # a marker that is too small cannot be measured


def test_triton_bayer_conversion():
    from robot_camera_calibration.camera_triton import TritonCamera
    raw = np.zeros((4, 6, 1), np.uint8)
    raw[0::2, 0::2] = 200                                    # R positions of the RG/GB pattern
    bgr = TritonCamera.convert(raw, "BayerRG8")
    assert bgr.shape == (4, 6, 3) and bgr[..., 2].mean() > bgr[..., 0].mean()


def test_matrix_to_rpy_roundtrip():
    for rpy in [(0.3, -0.2, 1.1), (-2.9, 0.4, -0.7), (3.1, 0.0, 0.0)]:
        R = pm.rpy_to_matrix(*rpy)
        assert pm.rotation_angle_deg(pm.rpy_to_matrix(*pm.matrix_to_rpy(R)), R) < 1e-6


def test_average_rotation():
    R0 = pm.rpy_to_matrix(0.2, 0.1, -0.4)
    rots = [R0 @ pm.rpy_to_matrix(0.0, 0.0, d) for d in (-0.02, 0.0, 0.02)]
    assert pm.rotation_angle_deg(core.average_rotation(rots), R0) < 1e-6


def test_point_from_xyz_ignores_holes_and_flying_pixels():
    center = np.array([-0.02, 0.04, 0.8])
    xyz, corners = _tilted_plane_xyz(center, (0.2, 0.3, -0.1), noise_m=0.002, seed=3)
    c = core.marker_center(corners)
    u, v = int(round(c[0])), int(round(c[1]))
    xyz[v - 2:v + 3, u - 2:u + 3] = np.nan            # hole at the centre
    rng = np.random.default_rng(1)
    us, vs = core._quad_pixels(xyz.shape, corners, c, 0.8)
    pick = rng.choice(len(us), size=len(us) // 20, replace=False)
    xyz[vs[pick], us[pick], 2] += 0.05                 # 5 % jump by 5 cm
    pp = core.point_from_xyz(xyz, corners, c)
    assert pp.point is not None
    assert np.linalg.norm(pp.point - center) * 1000 < 1.0


def test_point_from_xyz_needs_enough_points():
    xyz, corners = _tilted_plane_xyz(np.array([0, 0, 0.7]), (0, 0, 0))
    xyz[:] = np.nan
    pp = core.point_from_xyz(xyz, corners, core.marker_center(corners))
    assert pp.point is None and pp.points_used == 0


def test_bright_marker_found_from_raw_intensity():
    """ToF head-on reflection: a small marker is much brighter than the scene, so p99.5 8-bit clips black and white together."""
    from robot_camera_calibration.camera_base import intensity_to_gray
    from robot_camera_calibration.sim_camera import _marker_image
    rng = np.random.default_rng(0)
    intensity = rng.integers(800, 1200, size=(480, 640)).astype(np.uint16)
    m = _marker_image("DICT_4X4_50", 0, 24)                      # marker 24 px, white board 38 px (7 px margin)
    patch = np.full((38, 38), 255, np.uint8)
    patch[7:31, 7:31] = m
    intensity[221:259, 301:339] = np.where(patch > 127, 60000, 30000)   # even the black cells are 30x the background
    clipped = intensity_to_gray(intensity, 99.5, 1.0)
    assert (clipped[221:259, 301:339] == 255).all()              # contrast completely gone
    det = core.MarkerDetector("DICT_4X4_50", 0)
    assert det.detect(clipped) is None
    corners = det.detect(clipped, intensity)
    assert corners is not None and det.last_variant != "display"
    assert np.linalg.norm(core.marker_center(corners) - [319.5, 239.5]) < 1.0


def test_robust_fit_recovers_transform_and_rejects_outlier():
    rng = np.random.default_rng(0)
    R_true = pm.rpy_to_matrix(3.1, 0.02, 1.55)
    t_true = np.array([0.48, -0.05, 0.84])
    base = np.array(mf.generate_grid_poses([0.45, -0.05, 0.25, 0, 0, 0], "+x"))[:, :3]
    cam = (base - t_true) @ R_true
    cam += rng.normal(0, 0.0007, cam.shape)
    cam[5] += [0.03, 0, 0]
    fit = core.robust_fit(cam, base)
    assert not fit.inliers[5] and fit.inliers.sum() == 26
    assert pm.rotation_angle_deg(fit.R, R_true) < 0.3
    assert np.linalg.norm(fit.t - t_true) < 0.002


def test_helios_convert_scale_offset_and_invalid():
    raw = np.zeros((2, 3, 4), np.uint16)
    raw[..., 0] = 1000
    raw[..., 1] = 2000
    raw[..., 2] = 4000
    raw[..., 3] = 500
    raw[1, 2, 2] = INVALID_RAW
    scale = np.array([0.25, 0.25, 0.25])
    offset = np.array([-100.0, -200.0, 0.0])
    image, xyz = HeliosCamera.convert(raw, scale, offset)
    assert image.shape == (2, 3, 3) and image.dtype == np.uint8
    assert np.allclose(xyz[0, 0], [(1000 * 0.25 - 100) / 1000, (2000 * 0.25 - 200) / 1000, 1.0])
    assert np.isnan(xyz[1, 2]).all()
