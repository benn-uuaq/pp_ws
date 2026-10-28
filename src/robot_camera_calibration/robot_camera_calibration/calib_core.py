"""
calib_core.py - marker detection, marker-centre 3D measurement, rigid fit, result file.

Taken from the original calib_core.py; only the 3D measurement was changed for the Helios.

* The original (RealSense) back-projected the centre pixel with the depth image + intrinsics K.
* The Helios gives XYZ directly for every pixel (camera frame, X right, Y down,
  Z in front of the lens), so K is not needed.
* ToF is weak on the black cells and produces jumping points (flying pixels) on the marker
  border. Instead of taking the single centre value, a plane model is fitted to the points inside
  the marker square (sample_scale ratio) and read at the centre pixel.

  For a pinhole camera, points on a plane satisfy, for pixel (u, v),
      X/Z, Y/Z, 1/Z  are all linear in u, v.
  The three are fitted by least squares; points with a large depth residual are dropped once and refitted.
  Lens distortion is negligible within the marker size.
* The same plane model gives the 3D of the four corners, so the marker rotation is measured too. It is not
  used for the fit; it checks the marker rotation entered by the operator.

Result base_T_cam = (R, t):  p_base = R @ p_cam + t  (m)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2 as cv
import numpy as np
import yaml

from robot_camera_calibration.camera_base import intensity_to_gray


# ======================================================================
# Marker detection
# ======================================================================
class MarkerDetector:
    """Find one ArUco marker (dictionary, id) in a BGR / grey image."""

    def __init__(self, dictionary: str = "DICT_4X4_50", marker_id: int = 0):
        if not hasattr(cv, "aruco"):
            raise RuntimeError("This OpenCV has no ArUco module (opencv-contrib required).")
        if not hasattr(cv.aruco, dictionary):
            raise ValueError(f"Unknown ArUco dictionary '{dictionary}'")
        self.dictionary_name = dictionary
        self.marker_id = int(marker_id)
        self.last_variant = ""
        dictionary_obj = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, dictionary))

        if hasattr(cv.aruco, "ArucoDetector"):            # OpenCV >= 4.7
            params = cv.aruco.DetectorParameters()
            self._set_subpix(params)
            self._detector = cv.aruco.ArucoDetector(dictionary_obj, params)
            self._legacy = None
        else:                                               # older OpenCV (Ubuntu 22.04 = 4.5.4)
            params = cv.aruco.DetectorParameters_create()
            self._set_subpix(params)
            self._detector = None
            self._legacy = (dictionary_obj, params)

    @staticmethod
    def _set_subpix(params) -> None:
        method = getattr(cv.aruco, "CORNER_REFINE_SUBPIX", None)
        if method is not None:
            try:
                params.cornerRefinementMethod = method
            except Exception:
                pass

    def _detect_gray(self, gray: np.ndarray):
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            dictionary_obj, params = self._legacy
            corners, ids, _ = cv.aruco.detectMarkers(gray, dictionary_obj, parameters=params)
        if ids is None:
            return [], np.zeros(0, int)
        return [np.asarray(c, dtype=float).reshape(4, 2) for c in corners], np.asarray(ids).flatten()

    @staticmethod
    def _gray_variants(image: np.ndarray, intensity: Optional[np.ndarray]):
        """8-bit images to try detection on (name, image).

        On ToF the illumination is on the camera, so a marker in the middle of the image facing the camera
        reflects much brighter than the rest. Clipping to 8 bit on the whole-image range saturates both the white
        and black cells to 255, so the raw 16-bit intensity is viewed several times with different references and gamma.
        """
        if image is not None:
            yield "display", cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if intensity is None:
            return
        yield "p99.5", intensity_to_gray(intensity, 99.5, 1.0)
        yield "max-gamma", intensity_to_gray(intensity, 100.0, 0.5)
        yield "max", intensity_to_gray(intensity, 100.0, 1.0)
        log = np.log1p(intensity.astype(np.float32))
        log8 = (log * (255.0 / max(float(log.max()), 1e-6))).astype(np.uint8)
        yield "log-clahe", cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(log8)

    def detect_all(self, image: np.ndarray, intensity: Optional[np.ndarray] = None):
        """All markers found (list of corners, id array).

        Returns the result of the image where the target id was found. If none has it, returns the result with the
        most markers (to show other ids on screen). The image name used is in last_variant.
        """
        best = ([], np.zeros(0, int))
        self.last_variant = ""
        for name, gray in self._gray_variants(image, intensity):
            corners, ids = self._detect_gray(gray)
            if any(int(m) == self.marker_id for m in ids):
                self.last_variant = name
                return corners, ids
            if len(ids) > len(best[1]):
                best = (corners, ids)
        return best

    def detect(self, image: np.ndarray, intensity: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Four corners of the target marker (4x2, TL-TR-BR-BL) or None."""
        corners, ids = self.detect_all(image, intensity)
        for c, mid in zip(corners, ids):
            if int(mid) == self.marker_id:
                return c
        return None


def marker_center(corners: np.ndarray) -> np.ndarray:
    """Marker centre pixel = intersection of the two diagonals.

    Under perspective projection the mean of the four corners is not the projection of the square centre.
    The diagonal intersection is exact.
    """
    p0, p1, p2, p3 = np.asarray(corners, dtype=float).reshape(4, 2)
    d1 = p2 - p0
    d2 = p3 - p1
    A = np.column_stack((d1, -d2))
    if abs(np.linalg.det(A)) < 1e-9:
        return np.mean([p0, p1, p2, p3], axis=0)
    s, _ = np.linalg.solve(A, p1 - p0)
    return p0 + s * d1


# ======================================================================
# 3D measurement (XYZ image)
# ======================================================================
def _quad_pixels(shape, corners: np.ndarray, center: np.ndarray, scale: float):
    """Pixels (u, v) inside the marker square shrunk by scale around the centre."""
    pts = center + (np.asarray(corners, float) - center) * float(scale)
    mask = np.zeros(shape[:2], np.uint8)
    cv.fillConvexPoly(mask, np.round(pts).astype(np.int32), 1)
    vs, us = np.nonzero(mask)
    return us, vs


@dataclass
class PlanePoint:
    point: Optional[np.ndarray]     # marker centre 3D [m], None on failure
    points_used: int                # points used for the plane fit
    plane_rms_mm: float             # depth residual RMS
    rotation: Optional[np.ndarray] = None   # marker rotation in the camera frame (3x3), see marker_rotation below


def marker_rotation(corner_pts: np.ndarray) -> np.ndarray:
    """Marker frame rotation from the 3D coordinates of the four corners (TL, TR, BR, BL).

    Follows the ArUco (OpenCV) marker frame:
      X = top-left -> top-right corner, Y = bottom -> top, Z = out of the marker surface (towards the viewer).
    """
    c0, c1, c2, c3 = np.asarray(corner_pts, dtype=float).reshape(4, 3)
    x = (c1 - c0) + (c2 - c3)
    y = (c0 - c3) + (c1 - c2)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def average_rotation(rotations) -> np.ndarray:
    """Mean of several rotation matrices (sum of matrices projected back to the nearest rotation with SVD)."""
    M = np.sum(np.asarray(rotations, dtype=float).reshape(-1, 3, 3), axis=0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def point_from_xyz(xyz: np.ndarray, corners: np.ndarray, pixel: np.ndarray,
                   sample_scale: float = 0.8, min_points: int = 30) -> PlanePoint:
    """3D coordinates of the marker centre from an XYZ image (HxWx3, m, invalid = NaN)."""
    us, vs = _quad_pixels(xyz.shape, corners, np.asarray(pixel, float), sample_scale)
    P = xyz[vs, us].astype(float)
    ok = np.isfinite(P).all(axis=1) & (P[:, 2] > 0)
    us, vs, P = us[ok], vs[ok], P[ok]
    if len(P) < min_points:
        return PlanePoint(None, len(P), float("nan"))

    A = np.column_stack([us, vs, np.ones(len(us))]).astype(float)
    Y = np.column_stack([P[:, 0] / P[:, 2], P[:, 1] / P[:, 2], 1.0 / P[:, 2]])

    def solve(mask):
        coef = np.linalg.lstsq(A[mask], Y[mask], rcond=None)[0]
        z_fit = 1.0 / (A @ coef[:, 2])
        return coef, P[:, 2] - z_fit

    keep = np.ones(len(P), bool)
    coef, res = solve(keep)
    mad = float(np.median(np.abs(res - np.median(res))))
    threshold = max(3.0 * 1.4826 * mad, 0.002)       # never filter below 2 mm
    inliers = np.abs(res) <= threshold
    if min_points <= inliers.sum() < len(P):
        keep = inliers
        coef, res = solve(keep)

    def at(uv):
        a = np.array([uv[0], uv[1], 1.0]) @ coef          # [X/Z, Y/Z, 1/Z]
        return None if a[2] <= 0 else np.array([a[0], a[1], 1.0]) / a[2]

    center = at(pixel)
    if center is None:
        return PlanePoint(None, int(keep.sum()), float("nan"))
    rms = float(np.sqrt(np.mean(res[keep] ** 2)) * 1000.0)
    corner_pts = [at(c) for c in np.asarray(corners, float)]
    rotation = None if any(c is None for c in corner_pts) else marker_rotation(corner_pts)
    return PlanePoint(center, int(keep.sum()), rms, rotation)


@dataclass
class MarkerObservation:
    corners: np.ndarray                 # 4x2 pixels
    pixel: np.ndarray                   # marker centre (u, v)
    point_cam: Optional[np.ndarray]     # 3D centre in the camera frame [m], None if the measurement failed
    points_used: int = 0
    plane_rms_mm: float = float("nan")
    rotation_cam: Optional[np.ndarray] = None   # marker rotation in the camera frame (3x3)

    def tilt_deg(self) -> Optional[float]:
        """Angle by which the marker face is tilted from facing the camera head-on."""
        if self.rotation_cam is None:
            return None
        return math.degrees(math.acos(max(-1.0, min(1.0, -float(self.rotation_cam[2, 2])))))


def observe_marker(image: np.ndarray, xyz: Optional[np.ndarray], detector: MarkerDetector,
                   sample_scale: float = 0.8, min_points: int = 30,
                   intensity: Optional[np.ndarray] = None) -> Optional[MarkerObservation]:
    """Find the marker and measure its centre in 3D. None if there is no marker."""
    corners = detector.detect(image, intensity)
    if corners is None:
        return None
    return measure_corners(corners, xyz, sample_scale, min_points)


def measure_corners(corners: np.ndarray, xyz: Optional[np.ndarray],
                    sample_scale: float = 0.8, min_points: int = 30,
                    K: Optional[np.ndarray] = None, dist: Optional[np.ndarray] = None) -> MarkerObservation:
    """Measure the centre pixel and 3D from already-found marker corners.

    Without XYZ (RGB camera) there is no 3D; with intrinsics K only the marker orientation is measured.
    """
    c = marker_center(corners)
    if xyz is None:
        rotation = marker_rotation_from_corners(corners, K, dist) if K is not None else None
        return MarkerObservation(corners, c, None, rotation_cam=rotation)
    pp = point_from_xyz(xyz, corners, c, sample_scale, min_points)
    return MarkerObservation(corners, c, pp.point, pp.points_used, pp.plane_rms_mm, pp.rotation)


def marker_edge_width(image: np.ndarray, corners: np.ndarray) -> Optional[float]:
    """Sharpness: width [px] over which the brightness of the marker outer edge (black border -> white margin) rises from 10 % to 90 %.

    An in-focus camera usually gives 2 ~ 3 px; blur makes it larger. Only one marker cell (1/6 of a side) is examined, so
    if the marker is too small (cell under 4 px) it cannot be measured -> None.
    """
    gray = image if image.ndim == 2 else cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    P = np.asarray(corners, dtype=float).reshape(4, 2)
    centre = P.mean(axis=0)
    module = float(np.mean(np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1))) / 6.0
    if module < 4.0:
        return None
    # inside only within the black border (one cell), outside only within the white margin. Using the whole cell would
    # reach the white inner cells and spoil the "black side" reference.
    half = 0.7 * module
    s = np.linspace(-half, half, max(80, int(half * 20)))
    quarter = len(s) // 5
    widths = []
    for i in range(4):
        p0, p1 = P[i], P[(i + 1) % 4]
        n = np.array([-(p1 - p0)[1], (p1 - p0)[0]])
        n /= np.linalg.norm(n)
        if np.dot((p0 + p1) / 2.0 - centre, n) < 0:
            n = -n                                         # outwards (white margin)
        for f in np.linspace(0.3, 0.7, 9):                 # avoid the corners
            m = p0 + f * (p1 - p0)
            pts = (m[None, :] + s[:, None] * n[None, :]).astype(np.float32)
            prof = cv.remap(gray, pts[:, :1], pts[:, 1:], cv.INTER_LINEAR).ravel()
            lo, hi = float(np.median(prof[:quarter])), float(np.median(prof[-quarter:]))
            if hi - lo < 20.0:                             # skip if there is no contrast (occluded / saturated)
                continue
            x = (prof - lo) / (hi - lo)
            above10, above90 = np.nonzero(x > 0.1)[0], np.nonzero(x > 0.9)[0]
            if len(above10) and len(above90):
                widths.append(float(s[above90[0]] - s[above10[0]]))
    return float(np.median(widths)) if len(widths) >= 6 else None


# Corners of the ArUco marker frame (TL, TR, BR, BL), side 2. This is the order solvePnP IPPE_SQUARE requires.
_MARKER_SQUARE = np.array([[-1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]])


def marker_rotation_from_corners(corners: np.ndarray, K: np.ndarray,
                                 dist: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Marker rotation in the camera frame from the 2D corners only. The rotation does not depend on the marker size."""
    try:
        ok, rvec, _ = cv.solvePnP(_MARKER_SQUARE, np.asarray(corners, float).reshape(4, 1, 2),
                                  np.asarray(K, float), None if dist is None else np.asarray(dist, float),
                                  flags=cv.SOLVEPNP_IPPE_SQUARE)
    except cv.error:
        return None
    return cv.Rodrigues(rvec)[0] if ok else None


# ======================================================================
# PnP fit (RGB camera: base 3D points <-> image pixels)
# ======================================================================
@dataclass
class PnpFitResult:
    R: np.ndarray               # base_T_cam rotation
    t: np.ndarray               # base_T_cam translation [m]
    reproj_px: np.ndarray       # reprojection error per sample [px]
    residuals_mm: np.ndarray    # reprojection error converted to mm at that point's distance (lateral error)
    inliers: np.ndarray
    rms_px: float
    max_px: float
    rms_mm: float
    max_mm: float
    median_mm: float


def fit_pnp(base_pts, pixels, K: np.ndarray, dist: Optional[np.ndarray] = None,
            ransac_px: float = 3.0, min_samples: int = 6) -> PnpFitResult:
    """Camera pose from the marker base coordinates known to the robot and the image pixels (RANSAC, then LM refinement).

    p_cam = R_cb p_base + t_cb  ->  base_T_cam = (R_cb^T, -R_cb^T t_cb)
    """
    # OpenCV only accepts contiguous arrays (sliced arrays fail the npoints check).
    P = np.ascontiguousarray(np.asarray(base_pts, dtype=np.float64).reshape(-1, 3))
    uv = np.ascontiguousarray(np.asarray(pixels, dtype=np.float64).reshape(-1, 2))
    K = np.asarray(K, dtype=np.float64)
    D = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1)
    if len(P) < max(6, min_samples):
        raise ValueError(f"PnP needs at least {max(6, min_samples)} points ({len(P)} given)")
    ok, rvec, tvec, inl = cv.solvePnPRansac(P, uv, K, D, reprojectionError=float(ransac_px),
                                            iterationsCount=500, confidence=0.999,
                                            flags=cv.SOLVEPNP_EPNP)
    inliers = np.zeros(len(P), bool)
    if ok and inl is not None:
        inliers[np.asarray(inl).reshape(-1)] = True
    if not ok or inliers.sum() < max(6, min_samples):
        inliers[:] = True
        ok, rvec, tvec = cv.solvePnP(P, uv, K, D, flags=cv.SOLVEPNP_EPNP)
        if not ok:
            raise RuntimeError("PnP could not be solved")
    rvec, tvec = cv.solvePnPRefineLM(P[inliers], uv[inliers], K, D, rvec, tvec)

    proj, _ = cv.projectPoints(P, rvec, tvec, K, D)
    err_px = np.linalg.norm(proj.reshape(-1, 2) - uv, axis=1)
    R_cb = cv.Rodrigues(rvec)[0]
    z = (P @ R_cb.T + tvec.reshape(3))[:, 2]
    err_mm = err_px * z / float(K[0, 0]) * 1000.0
    used_px, used_mm = err_px[inliers], err_mm[inliers]
    return PnpFitResult(R=R_cb.T, t=-R_cb.T @ tvec.reshape(3), reproj_px=err_px, residuals_mm=err_mm,
                        inliers=inliers,
                        rms_px=float(np.sqrt(np.mean(used_px ** 2))), max_px=float(np.max(used_px)),
                        rms_mm=float(np.sqrt(np.mean(used_mm ** 2))), max_mm=float(np.max(used_mm)),
                        median_mm=float(np.median(used_mm)))


# ======================================================================
# Rigid fit (camera points -> base points)
# ======================================================================
def fit_rigid_transform(cam_pts, base_pts) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares R, t with base = R @ cam + t (Kabsch / SVD, no scale)."""
    P = np.asarray(cam_pts, dtype=float).reshape(-1, 3)
    Q = np.asarray(base_pts, dtype=float).reshape(-1, 3)
    if len(P) != len(Q) or len(P) < 3:
        raise ValueError("Need at least 3 matching point pairs")
    mp, mq = P.mean(axis=0), Q.mean(axis=0)
    H = (Q - mq).T @ (P - mp)
    U, _, Vt = np.linalg.svd(H)
    d = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    t = mq - R @ mp
    return R, t


def residuals_mm(R: np.ndarray, t: np.ndarray, cam_pts, base_pts) -> np.ndarray:
    P = np.asarray(cam_pts, dtype=float).reshape(-1, 3)
    Q = np.asarray(base_pts, dtype=float).reshape(-1, 3)
    return np.linalg.norm(P @ R.T + t - Q, axis=1) * 1000.0


def points_well_spread(points, min_extent_m: float = 0.01) -> bool:
    """True if the points are not bunched on one line and span at least a plane."""
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(P) < 3:
        return False
    s = np.linalg.svd(P - P.mean(axis=0), compute_uv=False) / math.sqrt(len(P))
    return bool(s[1] >= min_extent_m / 4.0)


@dataclass
class FitResult:
    R: np.ndarray
    t: np.ndarray
    residuals_mm: np.ndarray    # per sample (against the final fit)
    inliers: np.ndarray         # samples used for the final fit
    rms_mm: float
    max_mm: float
    median_mm: float


def robust_fit(cam_pts, base_pts, outlier_factor: float = 3.0,
               outlier_min_mm: float = 2.0, min_samples: int = 6) -> FitResult:
    """Fit, drop large outliers (> max(factor*median, min_mm)) once, and fit again."""
    P = np.asarray(cam_pts, dtype=float).reshape(-1, 3)
    Q = np.asarray(base_pts, dtype=float).reshape(-1, 3)
    R, t = fit_rigid_transform(P, Q)
    res = residuals_mm(R, t, P, Q)
    threshold = max(outlier_factor * float(np.median(res)), outlier_min_mm)
    inliers = res <= threshold
    if inliers.all() or inliers.sum() < max(3, min_samples):
        inliers = np.ones(len(P), dtype=bool)
    else:
        R, t = fit_rigid_transform(P[inliers], Q[inliers])
        res = residuals_mm(R, t, P, Q)
    used = res[inliers]
    return FitResult(R=R, t=t, residuals_mm=res, inliers=inliers,
                     rms_mm=float(np.sqrt(np.mean(used ** 2))),
                     max_mm=float(np.max(used)),
                     median_mm=float(np.median(used)))


# ======================================================================
# File
# ======================================================================
def to_builtin(obj):
    """Convert numpy values to plain Python types that YAML can write."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_builtin(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def build_result(R: np.ndarray, t: np.ndarray, meta: dict) -> dict:
    """Result dict. base_T_cam uses the same R / t format as the original tool, so existing readers still work."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return to_builtin({
        "base_T_cam": {"R": R, "t": t},
        "base_T_cam_matrix": T,
        "meta": meta,
    })


def save_yaml(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_builtin(data), f, sort_keys=False, allow_unicode=True)
