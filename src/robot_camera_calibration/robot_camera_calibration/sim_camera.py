"""
sim_camera.py - simulated Helios camera (calib_gui --sim and tests).

Draws the marker held by the simulated robot into an intensity image and an XYZ image. The true camera
pose is known, so the calibration result can be compared with it.

Simplifications: the marker board is always drawn parallel to the image plane, depth noise is Gaussian,
some pixels are invalid (NaN). The camera looks down almost vertically from above the camera_side
(piece-picking cell layout).
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Sequence, Tuple

import cv2 as cv
import numpy as np

from robot_camera_calibration import motion_frame as mf
from robot_camera_calibration import pose_math as pm
from robot_camera_calibration.camera_base import FrameBuffer
from robot_camera_calibration.sim_robot import SimRobot

# values close to the Helios2 (640x480, about 69 deg horizontal field of view)
SIM_K = np.array([[465.0, 0.0, 322.0],
                  [0.0, 465.0, 238.0],
                  [0.0, 0.0, 1.0]])
# values close to the Triton RGB (real 2048x1536, simulated at 1024x768 to save computation)
SIM_K_RGB = np.array([[780.0, 0.0, 515.0],
                      [0.0, 780.0, 382.0],
                      [0.0, 0.0, 1.0]])
# simulated Chroma: RGB camera pose relative to the ToF camera (60 mm to the side, slightly rotated)
SIM_RGB_FROM_TOF_R = pm.rpy_to_matrix(np.radians(0.5), np.radians(-0.8), np.radians(0.3))
SIM_RGB_FROM_TOF_T = np.array([0.060, 0.0, 0.005])


def _marker_image(dictionary: str, marker_id: int, size: int) -> np.ndarray:
    d = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, dictionary))
    if hasattr(cv.aruco, "generateImageMarker"):            # OpenCV >= 4.7
        return cv.aruco.generateImageMarker(d, int(marker_id), size, borderBits=1)
    return cv.aruco.drawMarker(d, int(marker_id), size, borderBits=1)


def true_camera_pose(side: str) -> Tuple[np.ndarray, np.ndarray]:
    """Camera looking down from 0.45 m on the camera_side and 0.85 m high (slightly tilted)."""
    axes = mf.view_axes(side)
    lateral, depth = axes[:, 0], axes[:, 1]
    # camera X = lateral, Y = depth, Z = down  (lateral x depth = -Z, so a right-handed frame)
    R0 = np.column_stack([lateral, depth, [0.0, 0.0, -1.0]])
    tilt = pm.rpy_to_matrix(np.radians(2.0), np.radians(-3.0), np.radians(4.0))
    R = tilt @ R0
    t = depth * 0.45 + lateral * 0.03 + np.array([0.0, 0.0, 0.85])
    return R, t


class SimCamera:
    def __init__(self, robot: SimRobot, R: np.ndarray, t: np.ndarray,
                 marker_offset_m: Sequence[float], convention: str,
                 dictionary: str = "DICT_4X4_50", marker_id: int = 0,
                 marker_size_m: float = 0.06, width: int = 640, height: int = 480,
                 fps: int = 30, depth_noise_m: float = 0.0008, invalid_ratio: float = 0.01,
                 background_depth_m: float = 1.2, marker_inplane_deg: float = 20.0, seed: int = 1,
                 K: Optional[np.ndarray] = None, with_depth: bool = True, name: str = "Helios"):
        self.robot = robot
        self.R, self.t = np.asarray(R, float), np.asarray(t, float)
        self.marker_offset_m = np.asarray(marker_offset_m, float)
        self.convention = convention
        self.width, self.height, self.fps = width, height, fps
        self.K = (SIM_K if K is None else np.asarray(K, float)).copy()
        self.dist = np.zeros(5)
        self.intrinsics_source = "simulated camera"
        self.with_depth = with_depth
        self.description = f"SIMULATED {name} {width}x{height}@{fps}"
        self.last_error = ""
        self.warnings = []
        self.marker_size_m = marker_size_m
        self.depth_noise_m = depth_noise_m
        self.invalid_ratio = invalid_ratio
        self.background_depth_m = background_depth_m
        # the marker is drawn parallel to the image plane, rotated in the image by marker_inplane_deg.
        # marker rotation in the camera frame (ArUco axes: X right, Y up, Z towards the camera)
        self._inplane = pm.rpy_to_matrix(0.0, 0.0, np.radians(marker_inplane_deg))
        self.marker_rotation_cam = self._inplane @ np.diag([1.0, -1.0, -1.0])
        self._marker_img = _marker_image(dictionary, marker_id, 240)
        self._rng = np.random.default_rng(seed)
        us, vs = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        self._ray_x = (us - self.K[0, 2]) / self.K[0, 0]
        self._ray_y = (vs - self.K[1, 2]) / self.K[1, 1]
        self._buf = FrameBuffer()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="sim-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_latest(self):
        return self._buf.latest()

    def wait_for_new_frame(self, after_seq: int, timeout: float = 2.0):
        return self._buf.wait_newer(after_seq, timeout)

    # ------------------------------------------------------------------
    def marker_in_camera(self) -> np.ndarray:
        p_base = pm.marker_point_in_base(self.robot.current_pose(), [0.0] * 6,
                                         self.marker_offset_m, self.convention)
        return self.R.T @ (p_base - self.t)

    def _project(self, pts_cam: np.ndarray) -> np.ndarray:
        K = self.K
        return np.stack([K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2],
                         K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]], axis=1)

    def render(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        w, h = self.width, self.height
        gray = np.full((h, w), 60, np.uint8)
        depth = np.full((h, w), self.background_depth_m, np.float32) if self.with_depth else None
        c = self.marker_in_camera()
        if c[2] > 0.05:
            half = self.marker_size_m / 2.0
            square = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float) @ self._inplane.T
            marker_px = self._project(c + square * half).astype(np.float32)
            plate_px = self._project(c + square * half * 1.8).astype(np.int32)
            cv.fillConvexPoly(gray, plate_px, 230)
            if depth is not None:
                plate = np.zeros((h, w), np.uint8)
                cv.fillConvexPoly(plate, plate_px, 1)
                depth[plate > 0] = c[2]

            s = float(self._marker_img.shape[0])
            src = np.array([[0, 0], [s, 0], [s, s], [0, s]], np.float32)
            H = cv.getPerspectiveTransform(src, marker_px)
            warped = cv.warpPerspective(self._marker_img, H, (w, h), flags=cv.INTER_LINEAR)
            mask = cv.warpPerspective(np.full_like(self._marker_img, 255), H, (w, h)) > 127
            gray[mask] = np.where(warped[mask] > 127, 230, 25).astype(np.uint8)

        xyz = None
        if depth is not None:
            depth = depth + self._rng.normal(0.0, self.depth_noise_m, size=(h, w)).astype(np.float32)
            xyz = np.stack([self._ray_x * depth, self._ray_y * depth, depth], axis=-1)
            xyz[self._rng.random((h, w)) < self.invalid_ratio] = np.nan
        noisy = np.clip(gray.astype(np.float32) + self._rng.normal(0.0, 2.0, size=gray.shape), 0, 255)
        return cv.cvtColor(noisy.astype(np.uint8), cv.COLOR_GRAY2BGR), xyz

    def _loop(self) -> None:
        period = 1.0 / float(self.fps)
        while not self._stop.is_set():
            t0 = time.time()
            image, xyz = self.render()
            self._buf.publish(image, xyz)
            time.sleep(max(0.0, period - (time.time() - t0)))


def create_simulation(cfg: dict, side: Optional[str] = None,
                      true_marker_offset_mm: Optional[Sequence[float]] = None, kind: str = "depth"):
    """Build a simulated robot + camera from the config.

    Without side, the camera is placed on the configured camera_side. The robot starts with the marker
    0.60 m along the ToF camera's optical axis and the flange facing up (towards the camera), rotated 25 deg.
    With kind = "rgb" it builds the RGB camera next to the ToF camera (no depth).
    Returns: (robot, camera, truth description, (R_true, t_true))  — the truth is that camera's base_T_cam
    """
    conv = cfg["robot"]["rotation_convention"]
    mk = cfg["marker"]
    side = mf.normalize_side(side or cfg["sequence"]["camera_side"])
    offset_m = np.asarray(true_marker_offset_mm if true_marker_offset_mm is not None
                          else mk["offset_from_flange_mm"], float) / 1000.0
    R_tof, t_tof = true_camera_pose(side)

    orient = [0.0, 0.0, np.radians(25.0)]
    marker_base = R_tof @ np.array([0.0, 0.0, 0.60]) + t_tof
    R_tcp = pm.rotation_from_pose([0, 0, 0] + orient, conv)
    start_pose = list(marker_base - R_tcp @ offset_m) + orient

    robot = SimRobot(start_pose)
    common = dict(dictionary=mk["dictionary"], marker_id=int(mk["id"]),
                  marker_size_m=float(mk["size_mm"]) / 1000.0)
    if kind == "rgb":
        R = R_tof @ SIM_RGB_FROM_TOF_R
        t = t_tof + R_tof @ SIM_RGB_FROM_TOF_T
        camera = SimCamera(robot, R, t, offset_m, conv, width=1024, height=768, K=SIM_K_RGB,
                           with_depth=False, name="Triton RGB", **common)
    else:
        R, t = R_tof, t_tof
        camera = SimCamera(robot, R, t, offset_m, conv, **common)
    # true marker rotation seen from the flange (the tool orientation stays the same during the grid, so it is unique)
    R_flange_marker = R_tcp.T @ R @ camera.marker_rotation_cam
    camera.true_marker_rotation_deg = np.degrees(pm.matrix_to_rpy(R_flange_marker))
    truth = (f"SIMULATION ({kind}): camera on the {side} side, true t = {pm.format_mm(t)}, "
             f"marker offset = {pm.format_mm(offset_m)}, "
             f"marker rotation = {pm.format_deg(camera.true_marker_rotation_deg)}")
    return robot, camera, truth, (R, t)
