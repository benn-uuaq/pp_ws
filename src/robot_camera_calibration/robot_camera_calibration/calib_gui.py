"""
calib_gui.py - Robot-camera calibration GUI (Helios + Elite CS, eye-to-hand).

Setup: a Helios camera fixed in the cell, one ArUco marker attached to the robot flange.

Flow
  1. Enter the robot IP -> [Connect robot]
  2. Tick where the camera is as seen from the robot base (+X / -X / +Y / -Y)
  3. [Connect camera] -> the live image shows the ArUco marker detection right away
  4. Jog the robot so the marker sits inside the circle in the middle of the image -> [Start calibration]
  5. The robot visits a 3x3x3 grid in camera axes, measures the marker centre in 3D at every pose,
     pairs it with the marker position reported by the robot, fits base_T_cam and saves it

Run:
  ros2 run robot_camera_calibration calib_gui
  ros2 run robot_camera_calibration calib_gui --sim     # simulated robot + simulated camera
"""

from __future__ import annotations

import argparse
import datetime
import ipaddress
import math
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np
from PyQt5.QtCore import QPointF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QTextEdit, QVBoxLayout, QWidget,
)

from robot_camera_calibration import calib_core as core
from robot_camera_calibration import motion_frame as mf
from robot_camera_calibration import pose_math as pm
from robot_camera_calibration import settings

APP_TITLE = "Robot-Camera Calibration (Helios2 Chroma, ArUco, eye-to-hand)"
VIEW_W, VIEW_H = 800, 600
LIVE_DETECT_WIDTH = 1024        # the live view detects on an image scaled to this width (RGB 3.2 MP is slow). Measurement uses full size
SHARP_OK_PX, SHARP_WARN_PX = 3.5, 6.0   # marker edge blur width [px]: below = sharp / slightly blurred, above = blurred
SIDE_LABELS = {"+x": "+X", "-x": "−X", "+y": "+Y", "-y": "−Y"}
OFFSET_REFERENCES = {"flange": "Flange (flange face centre, independent of the pendant TCP)",
                     "tcp": "Active TCP (the TCP set on the pendant)"}


def line_scale(img: np.ndarray) -> int:
    """Drawing line-width scale. Based on an 800 px wide image (3x for the 2048 px RGB image)."""
    return max(1, int(round(img.shape[1] / 800.0)))


# ======================================================================
# Calibration job (separate thread)
# ======================================================================
@dataclass
class Measurement:
    point_cam: Optional[np.ndarray]  # marker centre 3D (median over frames), camera frame [m]. None for RGB
    pixel: np.ndarray           # centre pixel (median)
    std_mm: float               # (depth) spread of the per-frame 3D points
    frames: int                 # number of frames used
    image: np.ndarray           # last image (for saving)
    corners: np.ndarray
    rotation_cam: Optional[np.ndarray]  # marker rotation in the camera frame (frame average), None if not measured
    std_px: float = float("nan")        # spread of the per-frame centre pixel


class CalibrationWorker(QThread):
    """kind = "depth": rigid fit of marker 3D (ToF) <-> base 3D.
    kind = "rgb"  : PnP of marker pixel (RGB) <-> base 3D (uses the camera intrinsics)."""
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished_run = pyqtSignal(bool, str)

    def __init__(self, robot, camera, detector, cfg, kind: str = "depth", parent=None,
                 start_target=None):
        super().__init__(parent)
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.cfg = cfg
        self.kind = kind
        self.start_target = None if start_target is None else list(start_target)   # saved start pose
        self.K = getattr(camera, "K", None)
        self.dist = getattr(camera, "dist", None)
        self.run_dir: Optional[Path] = None
        self.result_path: Optional[Path] = None
        self.result: Optional[dict] = None
        self._cancel = threading.Event()
        self._lines = []

    # ---- control ------------------------------------------------------------
    def request_stop(self) -> None:
        self._cancel.set()
        try:
            self.robot.stop_motion()
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        line = f"{datetime.datetime.now():%H:%M:%S} {msg}"
        self._lines.append(line)
        self.log.emit(line)

    def _log_alarms(self) -> None:
        for a in self.robot.pop_alarms():
            self._log(f"  [ROBOT ALARM] {a}")

    def run(self) -> None:
        ok, msg = False, ""
        try:
            ok, msg = self._run()
        except Exception as e:
            ok, msg = False, f"Unexpected error: {e}"
            self._log(traceback.format_exc())
        finally:
            self._log(("Done: " if ok else "Failed: ") + msg)
            if self.run_dir is not None:
                try:
                    (self.run_dir / "run.log").write_text("\n".join(self._lines) + "\n", encoding="utf-8")
                except Exception:
                    pass
            self.finished_run.emit(ok, msg)

    # ---- measurement ------------------------------------------------------------
    def _measure(self):
        sc, mc = self.cfg["sequence"], self.cfg["marker"]
        rgb = self.kind == "rgb"
        n = int(sc["frames_per_pose"])
        latest = self.camera.get_latest()
        seq = latest.seq if latest is not None else 0
        pts, pix, rots = [], [], []
        image = corners = None
        tries = 0
        while len(pix) < n and tries < 3 * n:
            if self._cancel.is_set():
                return None, "cancelled"
            fr = self.camera.wait_for_new_frame(seq, timeout=3.0)
            if fr is None:
                return None, "No camera frames are arriving"
            seq = fr.seq
            tries += 1
            found = self.detector.detect(fr.image, fr.intensity)
            if found is None:
                continue
            obs = core.measure_corners(found, fr.xyz, float(mc["sample_scale"]), int(mc["min_points"]),
                                       self.K, self.dist)
            if not rgb:
                if obs.point_cam is None:
                    continue
                pts.append(obs.point_cam)
            pix.append(obs.pixel)
            if obs.rotation_cam is not None:
                rots.append(obs.rotation_cam)
            image, corners = fr.image, obs.corners
        if len(pix) < max(3, n // 2):
            what = "marker detection" if rgb else "marker 3D measurement"
            return None, f"{what} succeeded in only {len(pix)} of {tries} frames"
        uv = np.array(pix)
        std_px = float(np.linalg.norm(uv.std(axis=0)))
        rotation = core.average_rotation(rots) if rots else None
        if rgb:
            if std_px > float(sc["max_pixel_std_px"]):
                return None, f"Measurement is unstable (std {std_px:.2f} px > {sc['max_pixel_std_px']} px)"
            return Measurement(None, np.median(uv, axis=0), float("nan"), len(pix), image, corners,
                               rotation, std_px), ""
        P = np.array(pts)
        std_mm = float(np.linalg.norm(P.std(axis=0)) * 1000.0)
        if std_mm > float(sc["max_point_std_mm"]):
            return None, f"Measurement is unstable (std {std_mm:.1f} mm > {sc['max_point_std_mm']} mm)"
        return Measurement(np.median(P, axis=0), np.median(uv, axis=0), std_mm, len(pts), image, corners,
                           rotation, std_px), ""

    def _save_image(self, path: Path, m: Measurement, label: str) -> None:
        img = m.image.copy()
        s = line_scale(img)
        cv.polylines(img, [m.corners.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2 * s)
        cv.circle(img, (int(round(m.pixel[0])), int(round(m.pixel[1]))), 4 * s, (0, 0, 255), -1)
        cv.putText(img, label, (10 * s, 25 * s), cv.FONT_HERSHEY_SIMPLEX, 0.6 * s, (0, 255, 255), 2 * s)
        cv.imwrite(str(path), img)

    def _move(self, target, tol_m: float):
        rc = self.cfg["robot"]
        if not self.robot.move_linear(target, float(rc["acceleration"]), float(rc["velocity"])):
            return False, f"Failed to send the move command: {self.robot.last_error}"
        return self.robot.wait_until_reached(target, tol_m, float(rc["move_timeout_s"]),
                                             cancel=self._cancel.is_set)

    # ---- whole sequence ---------------------------------------------------------
    def _run(self):
        rc, mc, sc = self.cfg["robot"], self.cfg["marker"], self.cfg["sequence"]
        fc, oc = self.cfg["fit"], self.cfg["output"]
        side = sc["camera_side"]
        conv = rc["rotation_convention"]
        offset_m = np.asarray(mc["offset_from_flange_mm"], dtype=float) / 1000.0
        from_tcp = mc["offset_reference"] == "tcp"
        tol_m = float(rc["position_tolerance_mm"]) / 1000.0

        out_dir = settings.output_dir(self.cfg)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = out_dir / "runs" / stamp
        img_dir = self.run_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        rgb = self.kind == "rgb"
        if rgb and self.K is None:
            return False, "The RGB camera intrinsics are missing."
        self._log(f"Run folder: {self.run_dir}")
        self._log(f"Camera: {settings.CAMERA_LABELS[self.kind]} - {getattr(self.camera, 'description', '')}")
        if rgb:
            self._log(f"Intrinsics: {getattr(self.camera, 'intrinsics_source', '')}, "
                      f"fx {self.K[0, 0]:.1f} fy {self.K[1, 1]:.1f} cx {self.K[0, 2]:.1f} cy {self.K[1, 2]:.1f}")
        self._log(f"Camera position: {SIDE_LABELS[side]} side of the base ({mf.describe(side)})")
        self._log(f"Rotation convention: {conv} | marker centre: {pm.format_mm(offset_m)} "
                  f"(relative to {'active TCP' if from_tcp else 'flange'})")

        # 0) if a start pose is saved, go there first
        if self.start_target is not None:
            st = self.robot.get_state()
            if st is None:
                return False, "No robot state received. Check the robot connection."
            d = float(np.linalg.norm(np.subtract(st.tcp_pose[:3], self.start_target[:3])))
            if d > tol_m:
                self._log(f"Moving to the saved start pose ({d * 1000:.0f} mm): {pm.format_pose(self.start_target)}")
                ok, why = self._move(self.start_target, tol_m)
                self._log_alarms()
                if not ok:
                    if self._cancel.is_set():
                        return False, "Stopped by the user."
                    return False, f"Could not reach the saved start pose: {why}"
                if self._cancel.wait(float(rc["settle_time_s"])):
                    return False, "Stopped by the user."

        # 1) start pose = where the operator left the robot (or the saved pose just reached)
        st = self.robot.get_state()
        if st is None:
            return False, "No robot state received. Check the robot connection."
        if st.emergency_stopped or st.protective_stopped:
            return False, "The robot is in emergency / protective stop."
        start_pose = list(st.tcp_pose)
        self._log(f"Start pose (actual TCP): {pm.format_pose(start_pose)}")
        if from_tcp:
            self._log(f"Active TCP offset {pm.format_pose(st.tcp_offset)} - the marker position is relative to this TCP. "
                      "Do not change the pendant TCP during calibration.")
        elif any(abs(v) > 1e-6 for v in st.tcp_offset):
            self._log(f"Note: the active TCP offset is not zero {pm.format_pose(st.tcp_offset)}. "
                      "It is compensated, so the marker position is used relative to the flange.")

        m0, why = self._measure()
        if m0 is None:
            return False, f"Cannot measure the marker at the start pose: {why}"
        if rgb:
            self._log(f"Start measurement: pixel ({m0.pixel[0]:.1f}, {m0.pixel[1]:.1f}), std {m0.std_px:.2f} px")
        else:
            self._log(f"Start measurement: pixel ({m0.pixel[0]:.1f}, {m0.pixel[1]:.1f}), "
                      f"camera {pm.format_mm(m0.point_cam)}, std {m0.std_mm:.2f} mm")

        # 2) visit the grid
        offsets = mf.grid_offsets_view(settings.step_m(self.cfg), int(sc["points_per_axis"]))
        total = len(offsets)
        self._log(f"Sequence: {total} poses, spacing lateral/depth/vertical = "
                  f"{sc['step_mm']['lateral']:g}/{sc['step_mm']['depth']:g}/{sc['step_mm']['vertical']:g} mm, "
                  f"speed {float(rc['velocity']) * 1000:.0f} mm/s")
        self.progress.emit(0, total)
        samples = []
        for i, o in enumerate(offsets, start=1):
            if self._cancel.is_set():
                return False, "Stopped by the user."
            target = mf.offset_pose(start_pose, side, o)
            self._log(f"[{i}/{total}] move - {mf.format_view_mm(o)}")
            reached, why = self._move(target, tol_m)
            self._log_alarms()
            if not reached:
                if self._cancel.is_set():
                    return False, "Stopped by the user."
                if "STOP" in why:
                    return False, f"Robot stopped: {why}"
                self._log(f"  [WARN] {why} -> skipped")
                self.progress.emit(i, total)
                continue
            if self._cancel.wait(float(rc["settle_time_s"])):
                return False, "Stopped by the user."

            m, why = self._measure()
            st = self.robot.get_state()
            if m is None or st is None:
                self._log(f"  [WARN] {why or 'no robot state'} -> skipped")
                self.progress.emit(i, total)
                continue
            # if relative to the TCP, treat the TCP as the flange (offset 0) and add as is.
            tcp_offset = [0.0] * 6 if from_tcp else st.tcp_offset
            marker_base = pm.marker_point_in_base(st.tcp_pose, tcp_offset, offset_m, conv)
            samples.append({
                "index": i,
                "offset_view_mm": [float(v) * 1000.0 for v in o],
                "target_pose": list(target),
                "actual_tcp_pose": list(st.tcp_pose),
                "tcp_offset": list(st.tcp_offset),
                "marker_base_m": marker_base,
                "marker_cam_m": m.point_cam,
                "pixel": m.pixel,
                "cam_std_mm": m.std_mm,
                "pixel_std_px": m.std_px,
                "frames": m.frames,
                "marker_rot_cam": m.rotation_cam,
                "reference_rot_base": pm.reference_rotation(st.tcp_pose, st.tcp_offset, conv, from_tcp),
            })
            if oc["save_images"]:
                self._save_image(img_dir / f"pose_{i:02d}.png", m, f"pose {i}/{total}")
            if rgb:
                self._log(f"  OK pixel ({m.pixel[0]:.1f}, {m.pixel[1]:.1f})  base {pm.format_mm(marker_base)}"
                          f"  std {m.std_px:.2f} px ({m.frames} frames)")
            else:
                self._log(f"  OK camera {pm.format_mm(m.point_cam)}  base {pm.format_mm(marker_base)}"
                          f"  std {m.std_mm:.2f} mm ({m.frames} frames)")
            self.progress.emit(i, total)

        # 3) return to the start pose
        if sc["return_to_start"] and not self._cancel.is_set():
            self._log("Returning to the start pose ...")
            ok, why = self._move(start_pose, tol_m)
            if not ok:
                self._log(f"  [WARN] return to start pose: {why}")

        # 4) fit
        core.save_yaml(self.run_dir / "samples.yaml", {"samples": samples})
        n_ok = len(samples)
        if n_ok < int(sc["min_valid_samples"]):
            return False, (f"Only {n_ok} valid samples (minimum {sc['min_valid_samples']}). "
                           "Check run.log: is the marker visible" + ("" if rgb else ", are 3D points measured") + "?")
        base = np.array([s["marker_base_m"] for s in samples])
        if not core.points_well_spread(base):
            return False, "The measured points are almost on one line, so the rotation cannot be solved."
        if rgb:
            pix = np.array([s["pixel"] for s in samples])
            fit = core.fit_pnp(base, pix, self.K, self.dist, float(fc["ransac_px"]),
                               int(sc["min_valid_samples"]))
            for s, r_px in zip(samples, fit.reproj_px):
                s["reprojection_px"] = float(r_px)
        else:
            cam = np.array([s["marker_cam_m"] for s in samples])
            fit = core.robust_fit(cam, base, float(fc["outlier_factor"]),
                                  float(fc["outlier_min_mm"]), int(sc["min_valid_samples"]))
        for s, r, inl in zip(samples, fit.residuals_mm, fit.inliers):
            s["residual_mm"] = float(r)
            s["used_in_fit"] = bool(inl)
            if not inl:
                self._log(f"  outlier removed: pose {s['index']} (residual {r:.1f} mm)")
        core.save_yaml(self.run_dir / "samples.yaml", {"samples": samples})

        cam_z_in_base = fit.R[:, 2]
        self._log(f"Fit: {int(fit.inliers.sum())}/{n_ok} samples, RMS {fit.rms_mm:.2f} mm, "
                  f"median {fit.median_mm:.2f} mm, max {fit.max_mm:.2f} mm"
                  + (f" (reprojection RMS {fit.rms_px:.2f} px, max {fit.max_px:.2f} px)" if rgb else ""))
        self._log(f"Camera position in base frame: {pm.format_mm(fit.t)}")
        self._log("Camera viewing direction in base frame: "
                  f"[{cam_z_in_base[0]:+.3f}, {cam_z_in_base[1]:+.3f}, {cam_z_in_base[2]:+.3f}]")
        self._log(f"R =\n{np.array2string(fit.R, precision=6, suppress_small=True)}")
        rotation_check = self._check_marker_rotation(samples, fit)

        # 5) compare the result with the selected camera side (the result itself is valid; this checks the checkbox)
        side_warning = ""
        if np.linalg.norm(fit.t[:2]) > 0.05:
            found = mf.nearest_side(fit.t[:2])
            if found != side:
                side_warning = (f"The result puts the camera on the {SIDE_LABELS[found]} side of the base, but "
                                f"{SIDE_LABELS[side]} is selected. Check the camera position checkbox "
                                "(the fit result is valid).")
                self._log(f"[WARN] {side_warning}")

        # 6) save
        camera_meta = {"kind": self.kind, "description": getattr(self.camera, "description", "")}
        if rgb:
            camera_meta.update({"intrinsics_source": getattr(self.camera, "intrinsics_source", ""),
                                "camera_matrix": self.K,
                                "dist_coeffs": np.zeros(5) if self.dist is None else self.dist})
        meta = {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "tool": "robot_camera_calibration/calib_gui",
            "camera_kind": self.kind,
            "method": ("PnP (robot base points <-> RGB marker pixels, RANSAC + LM), translation-only grid"
                       if rgb else "rigid point registration (SVD), translation-only grid"),
            "samples_total": total,
            "samples_valid": n_ok,
            "samples_used": int(fit.inliers.sum()),
            "rms_error_mm": round(fit.rms_mm, 3),
            "median_error_mm": round(fit.median_mm, 3),
            "max_error_mm": round(fit.max_mm, 3),
            "rotation_convention": conv,
            "marker": {
                "dictionary": mc["dictionary"],
                "id": int(mc["id"]),
                "offset_from_flange_mm": [float(v) for v in mc["offset_from_flange_mm"]],
                "offset_reference": mc["offset_reference"],
                "rotation_deg": [float(v) for v in mc["rotation_deg"]],
                **rotation_check,
            },
            "start_tcp_pose": start_pose,
            "grid": {"points_per_axis": int(sc["points_per_axis"]),
                     "camera_side": side,
                     "axes": mf.describe(side),
                     "step_mm": dict(sc["step_mm"])},
            "camera": camera_meta,
            "run_folder": str(self.run_dir),
        }
        if rgb:
            meta["rms_reprojection_px"] = round(fit.rms_px, 3)
            meta["max_reprojection_px"] = round(fit.max_px, 3)
        if side_warning:
            meta["warning"] = side_warning
        result = core.build_result(fit.R, fit.t, meta)
        self.result = result
        core.save_yaml(self.run_dir / "result.yaml", result)

        if fit.rms_mm > float(fc["max_rms_mm"]):
            return False, (f"Quality check failed: RMS {fit.rms_mm:.2f} mm > {fc['max_rms_mm']} mm. "
                           f"The main result file was not updated. Details: {self.run_dir}")
        self.result_path = settings.result_path(self.cfg, self.kind)
        core.save_yaml(self.result_path, result)
        msg = (f"[{settings.CAMERA_LABELS[self.kind]}] RMS {fit.rms_mm:.2f} mm, max {fit.max_mm:.2f} mm "
               + (f"(reprojection {fit.rms_px:.2f} px) " if rgb else "")
               + f"({int(fit.inliers.sum())}/{n_ok} samples).\nCamera position {pm.format_mm(fit.t)}\n"
               f"Saved: {self.result_path}")
        if "rotation_measured_deg" in rotation_check:
            msg += (f"\nMeasured marker rotation {pm.format_deg(rotation_check['rotation_measured_deg'])} "
                    f"({rotation_check['rotation_diff_deg']:.1f}° from the input)")
        for warning in (side_warning, rotation_check.get("rotation_warning", "")):
            if warning:
                msg += f"\n\nWarning: {warning}"
        return True, msg

    def _check_marker_rotation(self, samples, fit) -> dict:
        """Check the entered marker rotation against the marker orientation measured by the camera.

        For every pose compute  R_ref_marker = R_base_ref^T @ R_base_cam @ R_cam_marker,
        average them and compare with the input. Does not affect the fit (R, t).
        """
        mc = self.cfg["marker"]
        est = [np.asarray(s["reference_rot_base"]).T @ fit.R @ np.asarray(s["marker_rot_cam"])
               for s, inl in zip(samples, fit.inliers) if inl and s["marker_rot_cam"] is not None]
        if not est:
            self._log("[WARN] No sample measured the marker orientation, so the marker rotation could not be checked.")
            return {}
        R_meas = core.average_rotation(est)
        spread = max(pm.rotation_angle_deg(R_meas, R) for R in est)
        meas_deg = np.degrees(pm.matrix_to_rpy(R_meas))
        diff = pm.rotation_angle_deg(pm.rpy_deg_to_matrix(mc["rotation_deg"]), R_meas)
        self._log(f"Marker rotation: input {pm.format_deg(mc['rotation_deg'])}, measured {pm.format_deg(meas_deg)} "
                  f"(difference {diff:.1f}°, max spread over poses {spread:.1f}°)")
        info = {"rotation_measured_deg": [round(float(v), 2) for v in meas_deg],
                "rotation_diff_deg": round(float(diff), 2)}
        if diff > float(mc["rotation_tolerance_deg"]):
            info["rotation_warning"] = (
                f"The entered marker rotation {pm.format_deg(mc['rotation_deg'])} and the value measured by the camera "
                f"{pm.format_deg(meas_deg)} differ by {diff:.1f}°. Check the marker rotation input "
                "(the camera position result R, t is not affected).")
            self._log(f"[WARN] {info['rotation_warning']}")
        return info


class CameraStarter(QThread):
    """Camera discovery / connection can take a few seconds, so it runs separately to keep the UI responsive."""
    done = pyqtSignal(object, str)

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera

    def run(self) -> None:
        try:
            self.camera.start()
            self.done.emit(self.camera, "")
        except Exception as e:
            self.done.emit(None, str(e))


# ======================================================================
# Camera position drawing (cell seen from above)
# ======================================================================
class TopView(QWidget):
    """Robot base and camera position seen from above. X to the right, Y up."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.side = "+x"
        self.setFixedSize(150, 150)

    def set_side(self, side: str) -> None:
        self.side = side
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        p.fillRect(self.rect(), QColor(250, 250, 250))
        p.setPen(QPen(QColor(200, 200, 200), 1))
        p.drawRect(0, 0, w - 1, h - 1)

        # base axes
        axis_len = 34
        for (dx, dy), name, color in (((1, 0), "X", QColor(200, 40, 40)), ((0, -1), "Y", QColor(40, 150, 40))):
            p.setPen(QPen(color, 2))
            end = QPointF(cx + dx * axis_len, cy + dy * axis_len)
            p.drawLine(QPointF(cx, cy), end)
            p.drawText(end + QPointF(3 if dx else -12, -3 if dy else 4), name)
        p.setPen(QPen(QColor(60, 60, 60), 1))
        p.setBrush(QColor(120, 120, 120))
        p.drawEllipse(QPointF(cx, cy), 10, 10)

        # camera
        d = mf.CAMERA_SIDES[self.side]
        cam = QPointF(cx + d[0] * 55, cy - d[1] * 55)
        p.setPen(QPen(QColor(0, 90, 170), 1, Qt.DashLine))
        p.drawLine(QPointF(cx, cy), cam)
        p.setPen(QPen(QColor(0, 60, 130), 1))
        p.setBrush(QColor(0, 119, 196))
        p.drawRect(int(cam.x()) - 9, int(cam.y()) - 7, 18, 14)
        # lens: triangle pointing from the camera towards the robot (screen y grows downwards)
        fx, fy = -d[0], d[1]
        lens = QPolygonF([QPointF(cam.x() + fx * 18, cam.y() + fy * 18),
                          QPointF(cam.x() + fx * 9 - fy * 6, cam.y() + fy * 9 + fx * 6),
                          QPointF(cam.x() + fx * 9 + fy * 6, cam.y() + fy * 9 - fx * 6)])
        p.drawPolygon(lens)
        p.setPen(QColor(90, 90, 90))
        p.setFont(QFont(self.font().family(), 8))
        p.drawText(4, 13, "Cell seen from above")
        p.end()


# ======================================================================
# Main window
# ======================================================================
BUTTON_STYLE = """
QPushButton { color: white; background-color: rgb(0, 119, 196);
              font: bold 13pt; border: 1px solid black; padding: 6px; }
QPushButton:pressed { background-color: rgb(0, 80, 150); }
QPushButton:disabled { background-color: rgb(150, 150, 150); color: rgb(225, 225, 225); }
"""
STOP_STYLE = BUTTON_STYLE.replace("rgb(0, 119, 196)", "rgb(200, 40, 40)").replace(
    "rgb(0, 80, 150)", "rgb(150, 20, 20)")
SMALL_BUTTON_STYLE = BUTTON_STYLE.replace("font: bold 13pt", "font: bold 10pt")


def _valid_ip(text: str) -> bool:
    try:
        ipaddress.ip_address(text.strip())
        return True
    except ValueError:
        return False


class CalibrationWindow(QWidget):
    def __init__(self, cfg: dict, sim: bool = False, info_lines=()):
        super().__init__()
        self.cfg = cfg
        self.sim = sim
        mc = cfg["marker"]
        self.detector = core.MarkerDetector(mc["dictionary"], int(mc["id"]))
        self.robot = None
        self.camera = None
        self._sim_robot = None
        self._sim_cameras = {}
        self._starter: Optional[CameraStarter] = None
        self.worker: Optional[CalibrationWorker] = None
        self._last_obs: Optional[core.MarkerObservation] = None
        self._view_seq = -1
        self._fps_ref = (0, time.time())
        self._fps = 0.0
        self._jog_active = False
        self.start_pose = settings.load_start_pose(cfg)

        self.setWindowTitle(APP_TITLE + ("  [SIMULATION]" if sim else ""))
        self._build_ui()
        self._show_start_pose()
        for line in info_lines:
            self._append_log(line)

        self.jog_timer = QTimer(self)
        self.jog_timer.timeout.connect(self._jog_watchdog)
        self.jog_timer.start(100)

        self.view_timer = QTimer(self)
        self.view_timer.timeout.connect(self._update_view)
        self.view_timer.start(33)
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(250)

    # ---- UI ----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        self.view = QLabel("Camera not connected")
        self.view.setFixedSize(VIEW_W, VIEW_H)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet("background: black; color: white; font-size: 14pt;")
        left.addWidget(self.view, 0)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(160)
        self.log_box.setStyleSheet("QTextEdit { background: white; color: black; "
                                   "font-family: monospace; font-size: 9pt; }")
        left.addWidget(self.log_box, 1)
        root.addLayout(left, 0)

        # the settings panel on the right scrolls so it is not cut off on small screens.
        right_panel = QWidget()
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(0, 0, 4, 0)
        mono = "font-family: monospace;"

        # robot
        box = QGroupBox("Robot")
        g = QGridLayout(box)
        self.edit_robot_ip = QLineEdit(str(self.cfg["robot"]["ip"]))
        self.edit_robot_ip.setPlaceholderText("e.g. 192.168.1.101")
        self.btn_robot = QPushButton("Connect robot")
        self.btn_robot.setStyleSheet(SMALL_BUTTON_STYLE)
        self.btn_robot.clicked.connect(self._on_connect_robot)
        self.lbl_robot = QLabel("Not connected")
        self.lbl_tcp = QLabel("-")
        self.lbl_tcp.setStyleSheet(mono)
        g.addWidget(QLabel("IP"), 0, 0)
        g.addWidget(self.edit_robot_ip, 0, 1)
        g.addWidget(self.btn_robot, 0, 2)
        g.addWidget(self.lbl_robot, 1, 0, 1, 3)
        g.addWidget(self.lbl_tcp, 2, 0, 1, 3)
        right.addWidget(box)

        # robot jog / calibration start pose
        right.addWidget(self._build_jog_box())

        # camera position
        box = QGroupBox("Camera position (relative to the robot base)")
        hl = QHBoxLayout(box)
        self.top_view = TopView()
        hl.addWidget(self.top_view)
        col = QVBoxLayout()
        checks = QGridLayout()
        self.side_group = QButtonGroup(self)
        self.side_group.setExclusive(True)
        self.side_checks = {}
        for i, side in enumerate(("+x", "-x", "+y", "-y")):
            cb = QCheckBox(f"{SIDE_LABELS[side]} direction")
            self.side_group.addButton(cb)
            self.side_checks[side] = cb
            checks.addWidget(cb, i // 2, i % 2)
        col.addLayout(checks)
        self.lbl_axes = QLabel("-")
        self.lbl_axes.setWordWrap(True)
        col.addWidget(self.lbl_axes)
        col.addStretch(1)
        hl.addLayout(col, 1)
        self.side_checks[self.cfg["sequence"]["camera_side"]].setChecked(True)
        self.side_group.buttonToggled.connect(self._on_side_changed)
        self._on_side_changed()
        right.addWidget(box)

        # camera (RGB / depth selection)
        box = QGroupBox("Camera (Helios2 Chroma)")
        g = QGridLayout(box)
        self.kind_group = QButtonGroup(self)
        self.kind_group.setExclusive(True)
        self.kind_checks = {}
        for col, kind in enumerate(settings.CAMERA_KINDS):
            cb = QCheckBox(settings.CAMERA_LABELS[kind])
            self.kind_group.addButton(cb)
            self.kind_checks[kind] = cb
            g.addWidget(cb, 0, col, 1, 1 if col == 0 else 2)
        self.kind = self.cfg["active_camera"]
        self.kind_checks[self.kind].setChecked(True)
        self.kind_group.buttonToggled.connect(self._on_kind_changed)
        self.edit_cam_ip = QLineEdit(self._camera_cfg(self.kind)["ip"])
        self.edit_cam_ip.setPlaceholderText("leave empty to auto-discover")
        self.btn_camera = QPushButton("Connect camera")
        self.btn_camera.setStyleSheet(SMALL_BUTTON_STYLE)
        self.btn_camera.clicked.connect(self._on_connect_camera)
        self.lbl_camera = QLabel("Not connected")
        self.lbl_camera.setWordWrap(True)
        self.lbl_result_file = QLabel("-")
        self.lbl_result_file.setWordWrap(True)
        self.lbl_result_file.setStyleSheet("color: #555;")
        self.lbl_marker = QLabel("-")
        self.lbl_marker.setWordWrap(True)
        self.lbl_marker.setStyleSheet(mono)
        g.addWidget(QLabel("IP"), 1, 0)
        g.addWidget(self.edit_cam_ip, 1, 1)
        g.addWidget(self.btn_camera, 1, 2)
        g.addWidget(self.lbl_camera, 2, 0, 1, 3)
        g.addWidget(self.lbl_result_file, 3, 0, 1, 3)
        g.addWidget(QLabel("<b>Marker</b>"), 4, 0, Qt.AlignTop)
        g.addWidget(self.lbl_marker, 4, 1, 1, 2)
        right.addWidget(box)
        self._show_result_file()

        # marker position (reference -> marker centre)
        box = QGroupBox("Marker position (reference → marker centre)")
        g = QGridLayout(box)
        self.combo_ref = QComboBox()
        for ref, text in OFFSET_REFERENCES.items():
            self.combo_ref.addItem(text, ref)
        self.combo_ref.setCurrentIndex(max(0, self.combo_ref.findData(self.cfg["marker"]["offset_reference"])))
        g.addWidget(QLabel("Reference"), 0, 0)
        g.addWidget(self.combo_ref, 0, 1, 1, 5)
        self.spin_offset = []
        for col, name in enumerate("XYZ"):
            sp = QDoubleSpinBox()
            sp.setRange(-500.0, 500.0)
            sp.setDecimals(1)
            sp.setSuffix(" mm")
            sp.setValue(float(self.cfg["marker"]["offset_from_flange_mm"][col]))
            self.spin_offset.append(sp)
            g.addWidget(QLabel(name), 1, 2 * col)
            g.addWidget(sp, 1, 2 * col + 1)
        self.spin_rot = []
        for col, name in enumerate(("RX", "RY", "RZ")):
            sp = QDoubleSpinBox()
            sp.setRange(-180.0, 180.0)
            sp.setDecimals(1)
            sp.setSuffix(" °")
            sp.setValue(float(self.cfg["marker"]["rotation_deg"][col]))
            self.spin_rot.append(sp)
            g.addWidget(QLabel(name), 2, 2 * col)
            g.addWidget(sp, 2, 2 * col + 1)
        hint = QLabel("X/Y/Z = marker <b>centre</b> (not a corner), on the surface; rotation = for checking (hover for details)")
        hint.setStyleSheet("color: #555;")
        g.addWidget(hint, 3, 0, 1, 6)
        box.setToolTip(
            "X/Y/Z: from the reference to the centre of the black marker square (where the diagonals cross), on the marker surface.\n"
            "Not a corner. The axes are the reference (flange / TCP) frame, not the base.\n\n"
            "RX/RY/RZ: marker rotation seen from the reference frame [deg], R = Rz·Ry·Rx.\n"
            "Marker axes follow ArUco: X = top-left → top-right corner, Y = bottom → top, Z = out of the surface (towards the camera).\n"
            "Not used for the camera pose (R, t); after calibration it is compared with the marker orientation\n"
            "measured by the camera to check the input, and the measured value is stored in the result.")
        right.addWidget(box)

        # grid spacing
        box = QGroupBox("Grid spacing (camera axes)")
        g = QGridLayout(box)
        self.spin_step = {}
        names = {"lateral": "Lateral", "depth": "Depth", "vertical": "Vertical"}
        for col, axis in enumerate(mf.VIEW_AXES):
            sp = QDoubleSpinBox()
            sp.setRange(5.0, 150.0)
            sp.setDecimals(0)
            sp.setSingleStep(5.0)
            sp.setSuffix(" mm")
            sp.setValue(float(self.cfg["sequence"]["step_mm"][axis]))
            self.spin_step[axis] = sp
            g.addWidget(QLabel(f"±{names[axis]}"), 0, 2 * col)
            g.addWidget(sp, 0, 2 * col + 1)
        right.addWidget(box)

        # start / stop
        self.btn_start = QPushButton("Start calibration")
        self.btn_stop = QPushButton("Stop")
        self.btn_start.setStyleSheet(BUTTON_STYLE)
        self.btn_stop.setStyleSheet(STOP_STYLE)
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        buttons = QHBoxLayout()
        for b in (self.btn_start, self.btn_stop):
            b.setMinimumHeight(46)
            buttons.addWidget(b)
        right.addLayout(buttons)

        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m poses")
        self.progress.setMaximum(int(self.cfg["sequence"]["points_per_axis"]) ** 3)
        self.progress.setValue(0)
        right.addWidget(self.progress)

        self.lbl_result = QLabel("")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right.addWidget(self.lbl_result)
        right.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(right_panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # no horizontal scrolling, so make it wide enough for the content (including the vertical scroll bar).
        scroll.setMinimumWidth(right_panel.minimumSizeHint().width()
                               + scroll.verticalScrollBar().sizeHint().width() + 4)
        root.addWidget(scroll, 1)
        self.resize(VIEW_W + scroll.minimumWidth() + 30, VIEW_H + 220)

    def _build_jog_box(self) -> QGroupBox:
        """Jog buttons (move in camera axes / rotate about base axes) and save / go to the start pose."""
        jc = self.cfg["jog"]
        box = QGroupBox("Robot jog / calibration start pose")
        v = QVBoxLayout(box)

        top = QHBoxLayout()
        self.combo_jog = QComboBox()
        self.combo_jog.addItem("Continuous (while pressed)", None)
        for mm, deg in jc["steps"]:
            self.combo_jog.addItem(f"Step {float(mm):g} mm / {float(deg):g}°", (float(mm), float(deg)))
        self.spin_jog_speed = QDoubleSpinBox()
        self.spin_jog_speed.setRange(1.0, 100.0)
        self.spin_jog_speed.setDecimals(0)
        self.spin_jog_speed.setSuffix(" mm/s")
        self.spin_jog_speed.setValue(float(jc["speed_mm_s"]))
        self.spin_jog_rot = QDoubleSpinBox()
        self.spin_jog_rot.setRange(0.5, 30.0)
        self.spin_jog_rot.setDecimals(1)
        self.spin_jog_rot.setSuffix(" °/s")
        self.spin_jog_rot.setValue(float(jc["rot_speed_deg_s"]))
        top.addWidget(QLabel("Mode"))
        top.addWidget(self.combo_jog, 1)
        top.addWidget(QLabel("Speed"))
        top.addWidget(self.spin_jog_speed)
        top.addWidget(self.spin_jog_rot)
        v.addLayout(top)

        g = QGridLayout()
        self.jog_buttons = []
        self.jog_lin_buttons = {}
        names = {"lateral": "Lateral", "depth": "Depth", "vertical": "Vertical"}
        for row, axis in enumerate(mf.VIEW_AXES):
            g.addWidget(QLabel(names[axis]), row, 0)
            for col, sign in enumerate((-1, +1)):
                b = self._jog_button(("lin", row, sign))
                self.jog_lin_buttons[(axis, sign)] = b
                g.addWidget(b, row, 1 + col)
        rot_row = QHBoxLayout()
        for idx, name in enumerate(("RX", "RY", "RZ")):
            for sign in (-1, +1):
                b = self._jog_button(("rot", idx, sign))
                b.setText(f"{name}{'−' if sign < 0 else '+'}")
                rot_row.addWidget(b)
        g.addWidget(QLabel("Rotate"), 3, 0)
        g.addLayout(rot_row, 3, 1, 1, 2)
        v.addLayout(g)
        hint = QLabel("Moves use camera axes (actual base axis in brackets); rotations turn about base axes around the TCP. "
                      "Continuous mode stops when the button is released.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        v.addWidget(hint)

        h = QHBoxLayout()
        self.btn_save_start = QPushButton("Save current pose")
        self.btn_goto_start = QPushButton("Go to saved pose")
        self.btn_jog_stop = QPushButton("Stop")
        for b in (self.btn_save_start, self.btn_goto_start, self.btn_jog_stop):
            b.setStyleSheet(SMALL_BUTTON_STYLE if b is not self.btn_jog_stop
                            else STOP_STYLE.replace("font: bold 13pt", "font: bold 10pt"))
            b.setMinimumHeight(34)
            h.addWidget(b, 0 if b is self.btn_jog_stop else 1)
        self.btn_save_start.clicked.connect(self._on_save_start)
        self.btn_goto_start.clicked.connect(self._on_goto_start)
        self.btn_jog_stop.clicked.connect(self._on_stop)          # stops the go-to-saved-pose move / jog immediately
        v.addLayout(h)
        self.lbl_start = QLabel("-")
        self.lbl_start.setWordWrap(True)
        self.lbl_start.setStyleSheet("font-family: monospace;")
        v.addWidget(self.lbl_start)
        self.chk_start_saved = QCheckBox("Start calibration from the saved pose (move there first)")
        self.chk_start_saved.setChecked(bool(self.cfg["sequence"]["start_from_saved"]))
        v.addWidget(self.chk_start_saved)
        return box

    def _jog_button(self, spec) -> QPushButton:
        b = QPushButton()
        b.setAutoRepeat(False)
        b.setMinimumHeight(30)
        b.pressed.connect(lambda: self._on_jog_pressed(*spec))
        b.released.connect(self._on_jog_released)
        self.jog_buttons.append(b)
        return b

    def _update_jog_labels(self) -> None:
        axes = mf.view_axes(self._selected_side())
        texts = {("lateral", -1): "◀ Left", ("lateral", 1): "Right ▶",
                 ("depth", -1): "Away from camera", ("depth", 1): "Towards camera",
                 ("vertical", -1): "▼ Down", ("vertical", 1): "▲ Up"}
        for i, axis in enumerate(mf.VIEW_AXES):
            for sign in (-1, 1):
                base = mf.base_axis_label(axes[:, i] * sign)
                self.jog_lin_buttons[(axis, sign)].setText(f"{texts[(axis, sign)]} ({base})")

    def _append_log(self, text: str) -> None:
        self.log_box.append(text)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    # ---- jog ----------------------------------------------------------------
    def _jog_state(self):
        """Robot state if jogging is allowed, otherwise None (a button is held, so log instead of a dialog)."""
        if self._running() or self.robot is None:
            if self.robot is None:
                self._append_log("[Jog] Connect the robot first.")
            return None
        st = self.robot.get_state()
        if st is None:
            self._append_log("[Jog] No robot data.")
            return None
        if st.emergency_stopped or st.protective_stopped:
            self._append_log("[Jog] The robot is in emergency / protective stop.")
            return None
        return st

    def _on_jog_pressed(self, kind: str, idx: int, sign: int) -> None:
        st = self._jog_state()
        if st is None:
            return
        jc = self.cfg["jog"]
        side = self._selected_side()
        lin_speed = float(self.spin_jog_speed.value()) / 1000.0
        rot_speed = math.radians(float(self.spin_jog_rot.value()))
        step = self.combo_jog.currentData()
        xd = [0.0] * 6
        if kind == "lin":
            xd[:3] = list(mf.view_axes(side)[:, idx] * sign * lin_speed)
            acc = float(jc["acceleration"])
        else:
            xd[3 + idx] = sign * rot_speed
            acc = float(jc["rot_acceleration"])

        if step is None:                               # continuous: stops on release, or after hold_s if nothing arrives
            if self.robot.speed_linear(xd, acc, float(jc["hold_s"])):
                self._jog_active = True
            else:
                self._append_log(f"[Jog] Failed to send the command: {self.robot.last_error}")
            return
        if st.is_moving():
            self._append_log("[Jog] Still moving. Press again after it stops.")
            return
        mm, deg = step
        if kind == "lin":                              # exact distance: movel
            target = mf.step_pose(st.tcp_pose, side, mf.VIEW_AXES[idx], sign * mm / 1000.0)
            self.robot.move_linear(target, acc, lin_speed)
        else:                                          # rotation: speedl for a fixed time (approx. deg)
            self.robot.speed_linear(xd, acc, math.radians(deg) / rot_speed)

    def _on_jog_released(self) -> None:
        if self._jog_active:
            self._jog_stop()

    def _jog_stop(self) -> None:
        self._jog_active = False
        if self.robot is not None:
            self.robot.stop_motion()

    def _jog_watchdog(self) -> None:
        """Stop a continuous jog when the button release was missed (window lost focus etc.)."""
        if self._jog_active and not any(b.isDown() for b in self.jog_buttons):
            self._jog_stop()

    # ---- start pose -------------------------------------------------------------
    def _show_start_pose(self, current=None) -> None:
        sp = self.start_pose
        if not sp:
            self.lbl_start.setText("No saved start pose — jog the robot, then [Save current pose]")
            self.btn_goto_start.setEnabled(False)
            return
        text = f"Saved {sp.get('saved', '')}<br>{pm.format_pose(sp['tcp_pose']).replace(' | ', '<br>')}"
        if current is not None:
            d = float(np.linalg.norm(np.subtract(current[:3], sp["tcp_pose"][:3]))) * 1000.0
            text += f"<br>{d:.1f} mm from the current TCP" + (" (<span style='color:#080'>reached</span>)" if d < 2.0 else "")
        self.lbl_start.setText(text)
        self.btn_goto_start.setEnabled(self.robot is not None and not self._running())

    def _on_save_start(self) -> None:
        if self.robot is None:
            QMessageBox.warning(self, "Start pose", "Connect the robot first.")
            return
        st = self.robot.get_state()
        if st is None:
            QMessageBox.warning(self, "Start pose", "No robot data.")
            return
        if st.is_moving():
            QMessageBox.warning(self, "Start pose", "The robot is moving. Save after it stops.")
            return
        if self.start_pose and not self._ask("Start pose", "Replace the saved start pose with the current pose?"):
            return
        path = settings.save_start_pose(self.cfg, st.tcp_pose, st.tcp_offset, st.joint_positions)
        self.start_pose = settings.load_start_pose(self.cfg)
        self._append_log(f"Start pose saved: {pm.format_pose(st.tcp_pose)} -> {path}")
        self._show_start_pose(st.tcp_pose)

    def _on_goto_start(self) -> None:
        if self._running() or not self.start_pose or self.robot is None:
            return
        st = self._jog_state()
        if st is None:
            return
        target = self.start_pose["tcp_pose"]
        rc = self.cfg["robot"]
        d = float(np.linalg.norm(np.subtract(st.tcp_pose[:3], target[:3]))) * 1000.0
        warn = ""
        saved_offset = self.start_pose.get("tcp_offset")
        if saved_offset and not np.allclose(saved_offset, st.tcp_offset, atol=1e-4):
            warn = ("\n\nWarning: the active TCP on the pendant differs from when the pose was saved. "
                    "Switch to the same TCP before moving.")
        if not self._ask("Go to saved pose",
                         f"Linear move (movel) to the saved start pose.\n\n{pm.format_pose(target)}\n"
                         f"Distance {d:.0f} mm, speed {float(rc['velocity']) * 1000:.0f} mm/s{warn}\n\n"
                         "Make sure the path is clear. Move now?"):
            return
        if self.robot.move_linear(target, float(rc["acceleration"]), float(rc["velocity"])):
            self._append_log(f"Going to the saved pose: {pm.format_pose(target)} ({d:.0f} mm) — [Stop] stops it")
        else:
            self._append_log(f"[WARN] Failed to send the move command: {self.robot.last_error}")

    def _set_editing_enabled(self, enabled: bool) -> None:
        for w in ([self.edit_robot_ip, self.btn_robot, self.edit_cam_ip, self.btn_camera, self.btn_start,
                   self.combo_ref]
                  + list(self.kind_checks.values())
                  + list(self.side_checks.values()) + list(self.spin_step.values())
                  + self.spin_offset + self.spin_rot + self.jog_buttons
                  + [self.combo_jog, self.spin_jog_speed, self.spin_jog_rot, self.btn_save_start,
                     self.chk_start_saved]):
            w.setEnabled(enabled)
        self.btn_goto_start.setEnabled(enabled and bool(self.start_pose) and self.robot is not None)

    # ---- inputs -------------------------------------------------------------
    def _selected_side(self) -> str:
        for side, cb in self.side_checks.items():
            if cb.isChecked():
                return side
        return self.cfg["sequence"]["camera_side"]

    def _on_side_changed(self, *_args) -> None:
        side = self._selected_side()
        self.top_view.set_side(side)
        axes = mf.view_axes(side)
        lbl = [mf.base_axis_label(axes[:, i]) for i in range(3)]
        self.lbl_axes.setText(f"Towards camera (depth) = base <b>{lbl[1]}</b><br>"
                              f"Right as seen by the camera (lateral) = <b>{lbl[0]}</b><br>"
                              f"Up (vertical) = <b>{lbl[2]}</b>")
        self._update_jog_labels()

    def _camera_cfg(self, kind: str) -> dict:
        return self.cfg["camera_rgb" if kind == "rgb" else "camera"]

    def _show_result_file(self) -> None:
        path = settings.result_path(self.cfg, self.kind)
        if path.exists():
            when = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            state = f"<span style='color:#080'>exists</span> ({when})"
        else:
            state = "not yet"
        self.lbl_result_file.setText(f"Result file: {path.name} — {state}")
        self.lbl_result_file.setToolTip(str(path))

    def _on_kind_changed(self, *_args) -> None:
        new = next((k for k, cb in self.kind_checks.items() if cb.isChecked()), self.kind)
        if new == self.kind:
            return
        self._camera_cfg(self.kind)["ip"] = self.edit_cam_ip.text().strip()
        if self.camera is not None:
            self._disconnect_camera()
        self.kind = new
        self.cfg["active_camera"] = new
        self.edit_cam_ip.setText(self._camera_cfg(new)["ip"])
        self._show_result_file()
        self._append_log(f"Calibration camera: {settings.CAMERA_LABELS[new]}")
        self._collect_ui()

    def _collect_ui(self) -> None:
        """Copy the on-screen inputs into the settings and keep them in ui_state.yaml."""
        self.cfg["robot"]["ip"] = self.edit_robot_ip.text().strip()
        self.cfg["active_camera"] = self.kind
        self._camera_cfg(self.kind)["ip"] = self.edit_cam_ip.text().strip()
        self.cfg["sequence"]["camera_side"] = self._selected_side()
        self.cfg["sequence"]["step_mm"] = {a: float(sp.value()) for a, sp in self.spin_step.items()}
        self.cfg["marker"]["offset_from_flange_mm"] = [float(sp.value()) for sp in self.spin_offset]
        self.cfg["marker"]["rotation_deg"] = [float(sp.value()) for sp in self.spin_rot]
        self.cfg["marker"]["offset_reference"] = self.combo_ref.currentData()
        self.cfg["sequence"]["start_from_saved"] = self.chk_start_saved.isChecked()
        self.cfg["jog"]["speed_mm_s"] = float(self.spin_jog_speed.value())
        self.cfg["jog"]["rot_speed_deg_s"] = float(self.spin_jog_rot.value())
        try:
            settings.save_ui_state(self.cfg)
        except Exception as e:
            self._append_log(f"[WARN] Failed to save inputs: {e}")

    # ---- connection ---------------------------------------------------------------
    def _simulation(self):
        """Attach one simulated camera per camera kind to a single simulated robot."""
        from robot_camera_calibration.sim_camera import create_simulation
        if self._sim_robot is None:
            self._sim_robot = create_simulation(self.cfg)[0]
        if self.kind not in self._sim_cameras:
            _, camera, truth, _ = create_simulation(self.cfg, kind=self.kind)
            camera.robot = self._sim_robot
            self._sim_cameras[self.kind] = camera
            self._append_log(truth)
        return self._sim_robot, self._sim_cameras[self.kind]

    def _on_connect_robot(self) -> None:
        if self._running():
            return
        self._collect_ui()
        ip = self.cfg["robot"]["ip"]
        if self.robot is not None:
            self.robot.close()
            self.robot = None
        if self.sim:
            self.robot = self._simulation()[0]
        else:
            if not _valid_ip(ip):
                QMessageBox.warning(self, "Robot", f"Not a valid IP address: '{ip}'")
                return
            from robot_camera_calibration.robot_interface import RobotInterface
            rc = self.cfg["robot"]
            self.robot = RobotInterface(ip, int(rc["port"]), rc["move_command"], rc["stop_command"])
        ok = self.robot.connect()
        self._append_log(f"Robot connection {'OK' if ok else 'waiting (retrying in the background)'}: {self.robot.description}"
                         + ("" if ok else f" - {self.robot.last_error}"))

    def _disconnect_camera(self) -> None:
        self.camera.stop()
        self.camera = None
        self._last_obs = None
        self._view_seq = -1
        self.btn_camera.setText("Connect camera")
        self.view.setText("Camera not connected")
        self._append_log("Camera disconnected.")

    def _on_connect_camera(self) -> None:
        if self._running() or self._starter is not None:
            return
        if self.camera is not None:
            self._disconnect_camera()
            return
        self._collect_ui()
        cc = self._camera_cfg(self.kind)
        if cc["ip"] and not _valid_ip(cc["ip"]):
            QMessageBox.warning(self, "Camera", f"Not a valid IP address: '{cc['ip']}'")
            return
        if self.sim:
            camera = self._simulation()[1]
        elif self.kind == "rgb":
            from robot_camera_calibration.camera_triton import TritonCamera
            camera = TritonCamera(cc["ip"], cc["pixel_format"], cc["intrinsics_file"], int(cc["timeout_ms"]),
                                  int(cc["packet_size"]), int(cc["packet_delay_ns"]))
        else:
            from robot_camera_calibration.camera_helios import HeliosCamera
            camera = HeliosCamera(cc["ip"], cc["operating_mode"], cc["exposure_time"],
                                  int(cc["image_accumulation"]), bool(cc["spatial_filter"]),
                                  bool(cc["confidence_threshold"]), int(cc["timeout_ms"]),
                                  int(cc["packet_size"]), int(cc["packet_delay_ns"]))
        for cb in self.kind_checks.values():
            cb.setEnabled(False)
        self.btn_camera.setEnabled(False)
        self.btn_camera.setText("Connecting ...")
        self.view.setText("Connecting camera ...")
        self._starter = CameraStarter(camera, self)
        self._starter.done.connect(self._on_camera_started)
        self._starter.start()

    def _on_camera_started(self, camera, error: str) -> None:
        self._starter = None
        self.btn_camera.setEnabled(True)
        for cb in self.kind_checks.values():
            cb.setEnabled(True)
        if camera is None:
            self.btn_camera.setText("Connect camera")
            self.view.setText("Camera connection failed")
            self._append_log(f"Camera connection failed: {error}")
            QMessageBox.warning(self, "Camera", f"Cannot open the camera.\n\n{error}")
            return
        self.camera = camera
        self.btn_camera.setText("Disconnect camera")
        self._fps_ref = (0, time.time())
        self._append_log(f"Camera connected: {camera.description}")
        K = getattr(camera, "K", None)
        if self.kind == "rgb" and K is not None:
            self._append_log(f"  Intrinsics: {getattr(camera, 'intrinsics_source', '')} — fx {K[0, 0]:.1f}, "
                             f"fy {K[1, 1]:.1f}, cx {K[0, 2]:.1f}, cy {K[1, 2]:.1f}")
        for w in getattr(camera, "warnings", []):
            self._append_log(f"  [WARN] {w}")

    # ---- live image ----------------------------------------------------------
    def _update_view(self) -> None:
        if self.camera is None:
            return
        fr = self.camera.get_latest()
        if fr is None or fr.seq == self._view_seq:
            return
        self._view_seq = fr.seq
        img = fr.image.copy()
        h, w = img.shape[:2]
        s = line_scale(img)
        mc = self.cfg["marker"]
        K, dist = getattr(self.camera, "K", None), getattr(self.camera, "dist", None)
        # big images (RGB 3.2 MP) are detected scaled down so the view stays fast. Measurement (calibration) uses full size.
        shrink = min(1.0, LIVE_DETECT_WIDTH / float(w))
        if shrink < 1.0:
            small = cv.resize(fr.image, None, fx=shrink, fy=shrink, interpolation=cv.INTER_AREA)
            corners_all, ids = self.detector.detect_all(small, None)
            corners_all = [c / shrink for c in corners_all]
        else:
            corners_all, ids = self.detector.detect_all(fr.image, fr.intensity)
        variant = self.detector.last_variant
        target = next((c for c, m in zip(corners_all, ids) if int(m) == self.detector.marker_id), None)
        obs = None if target is None else core.measure_corners(
            target, fr.xyz, float(mc["sample_scale"]), int(mc["min_points"]), K, dist)
        self._last_obs = obs

        rgb_only = fr.xyz is None
        tol = self._centering_tolerance(w)
        cx, cy = self._optical_centre(w, h)
        centered = False
        if obs is not None:
            dx, dy = obs.pixel[0] - cx, obs.pixel[1] - cy
            centered = math.hypot(dx, dy) <= tol
            tilt = obs.tilt_deg()
            tilt_text = f", tilt vs. facing the camera {tilt:.0f}°" if tilt is not None else ""
            sharp_text = self._sharpness_text(fr.image, obs.corners)
            if rgb_only:
                edge = float(np.mean(np.linalg.norm(np.roll(obs.corners, -1, axis=0) - obs.corners, axis=1)))
                text = f"ID {self.detector.marker_id} OK  px {obs.pixel[0]:.0f}, {obs.pixel[1]:.0f}"
                self.lbl_marker.setText(
                    f"<span style='color:#080'><b>Detected</b></span> id {self.detector.marker_id}<br>"
                    f"pixel ({obs.pixel[0]:.1f}, {obs.pixel[1]:.1f}), marker side {edge:.0f} px{tilt_text}<br>"
                    f"{sharp_text}<br>"
                    f"from centre dx {dx:+.0f}, dy {dy:+.0f} px "
                    + ("(<span style='color:#080'>centred</span>)" if centered else "(move it to the centre)"))
            elif obs.point_cam is not None:
                p = obs.point_cam * 1000.0
                text = f"ID {self.detector.marker_id} OK  XYZ {p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f} mm"
                self.lbl_marker.setText(
                    f"<span style='color:#080'><b>Detected</b></span> id {self.detector.marker_id}"
                    + (f" (detection image {variant})" if variant and variant != "display" else "") + "<br>"
                    f"camera XYZ [{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}] mm<br>"
                    f"{obs.points_used} plane points, depth residual {obs.plane_rms_mm:.2f} mm{tilt_text}<br>"
                    f"{sharp_text}<br>"
                    f"from centre dx {dx:+.0f}, dy {dy:+.0f} px "
                    + ("(<span style='color:#080'>centred</span>)" if centered else "(move it to the centre)"))
            else:
                text = f"ID {self.detector.marker_id} found, NO 3D POINTS"
                self.lbl_marker.setText(
                    f"<span style='color:#c60'><b>Detected, no 3D</b></span> - valid points inside the marker "
                    f"{obs.points_used} (minimum {mc['min_points']}). Check distance / reflections.")
        else:
            others = ", ".join(str(int(i)) for i in ids)
            text = "MARKER NOT FOUND" + (f" (other ids: {others})" if others else "")
            self.lbl_marker.setText("<span style='color:#c00'><b>Not detected</b></span>"
                                    + (f" - only other ids {others} visible (configured id {self.detector.marker_id})"
                                       if others else ""))

        measured = obs is not None and (rgb_only or obs.point_cam is not None)
        ok_col = (0, 200, 0) if centered else (0, 165, 255)
        icx, icy = int(round(cx)), int(round(cy))
        cv.line(img, (icx, 0), (icx, h - 1), (200, 200, 200), s)
        cv.line(img, (0, icy), (w - 1, icy), (200, 200, 200), s)
        cv.circle(img, (icx, icy), int(tol), ok_col, 2 * s)
        for c, mid in zip(corners_all, ids):
            if int(mid) != self.detector.marker_id:
                cv.polylines(img, [c.astype(np.int32).reshape(-1, 1, 2)], True, (160, 160, 160), s)
                cv.putText(img, str(int(mid)), tuple(c[0].astype(int)), cv.FONT_HERSHEY_SIMPLEX,
                           0.5 * s, (160, 160, 160), s, cv.LINE_AA)
        if obs is not None:
            col = (0, 255, 0) if measured else (0, 165, 255)
            cv.polylines(img, [obs.corners.astype(np.int32).reshape(-1, 1, 2)], True, col, 2 * s)
            p = (int(round(obs.pixel[0])), int(round(obs.pixel[1])))
            cv.line(img, (icx, icy), p, ok_col, s)
            cv.circle(img, p, 4 * s, (0, 0, 255), -1)
        cv.rectangle(img, (0, 0), (w, 24 * s), (0, 0, 0), -1)
        cv.putText(img, text, (8 * s, 17 * s), cv.FONT_HERSHEY_SIMPLEX, 0.5 * s,
                   (0, 255, 0) if measured else (0, 140, 255), s, cv.LINE_AA)
        if self._running():
            cv.putText(img, "CALIBRATION RUNNING", (8 * s, h - 12 * s), cv.FONT_HERSHEY_SIMPLEX,
                       0.7 * s, (0, 0, 255), 2 * s, cv.LINE_AA)

        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        self.view.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @staticmethod
    def _sharpness_text(image: np.ndarray, corners: np.ndarray) -> str:
        """One line of sharpness from the marker edge blur width (measured on the full-resolution image)."""
        width = core.marker_edge_width(image, corners)
        if width is None:
            return "Sharpness: - (marker too small or no contrast to measure)"
        if width <= SHARP_OK_PX:
            color, word = "#080", "sharp"
        elif width <= SHARP_WARN_PX:
            color, word = "#c60", "slightly blurred"
        else:
            color, word = "#c00", "blurred"
        return (f"Sharpness: edge blur <span style='color:{color}'><b>{width:.1f} px ({word})</b></span> "
                f"— 2~3 px when in focus")

    def _optical_centre(self, w: int, h: int):
        """Where the optical axis meets the image: (cx, cy) if the intrinsics are known, else the image centre."""
        K = getattr(self.camera, "K", None)
        if K is not None:
            return float(K[0, 2]), float(K[1, 2])
        return (w - 1) / 2.0, (h - 1) / 2.0

    def _centering_tolerance(self, w: int) -> float:
        """Radius of the centre circle. The setting is for a 640 px wide image, so larger images scale it up."""
        return float(self.cfg["sequence"]["centering_tolerance_px"]) * max(1.0, w / 640.0)

    # ---- status ---------------------------------------------------------------
    def _update_status(self) -> None:
        if self.robot is None:
            self.lbl_robot.setText("Not connected")
            self.lbl_tcp.setText("-")
        else:
            st = self.robot.get_state()
            if st is None:
                err = getattr(self.robot, "last_error", "") or "waiting for data"
                self.lbl_robot.setText(f"<span style='color:#c00'><b>No data</b></span> "
                                       f"{self.robot.description} - {err}")
                self.lbl_tcp.setText("-")
            else:
                if st.emergency_stopped:
                    flag = "<span style='color:#c00'><b>Emergency stop</b></span>"
                elif st.protective_stopped:
                    flag = "<span style='color:#c00'><b>Protective stop</b></span>"
                elif not st.power_on:
                    flag = "<span style='color:#c60'><b>Power off</b></span>"
                else:
                    flag = "<span style='color:#080'><b>Normal</b></span>"
                self.lbl_robot.setText(f"{flag} {self.robot.description}")
                self.lbl_tcp.setText(pm.format_pose(st.tcp_pose).replace(" | ", "<br>"))
            self._show_start_pose(None if st is None else st.tcp_pose)
            if not self._running():
                self.btn_stop.setEnabled(True)          # jog / go-to-saved-pose are also stopped by [Stop]
                for a in self.robot.pop_alarms():
                    self._append_log(f"[ROBOT ALARM] {a}")

        if self.camera is not None:
            fr = self.camera.get_latest()
            seq0, t0 = self._fps_ref
            now = time.time()
            if fr is not None and now - t0 >= 1.0:
                self._fps = (fr.seq - seq0) / (now - t0) if seq0 else 0.0
                self._fps_ref = (fr.seq, now)
            err = getattr(self.camera, "last_error", "")
            self.lbl_camera.setText(f"<span style='color:#080'><b>Connected</b></span> "
                                    f"{self.camera.description}, {self._fps:.0f} fps"
                                    + (f" - <span style='color:#c00'>{err}</span>" if err else ""))
        elif self._starter is None:
            self.lbl_camera.setText("Not connected")
            self.lbl_marker.setText("-")

    # ---- buttons ---------------------------------------------------------------
    def _ask(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No) == QMessageBox.Yes

    def _on_start(self) -> None:
        if self._running():
            return
        if self.robot is None:
            QMessageBox.warning(self, "Robot", "Connect the robot first.")
            return
        st = self.robot.get_state()
        if st is None:
            QMessageBox.warning(self, "Robot", "No data from the robot. Check the connection.")
            return
        if st.emergency_stopped or st.protective_stopped:
            QMessageBox.warning(self, "Robot", "The robot is in emergency / protective stop.")
            return
        if self.camera is None:
            QMessageBox.warning(self, "Camera", "Connect the camera first.")
            return
        if self.kind == "rgb" and getattr(self.camera, "K", None) is None:
            QMessageBox.warning(self, "Camera", "The RGB camera intrinsics are missing.")
            return

        # When starting from the saved pose, go there first. If the robot is not there now, the marker
        # check happens at the job's first measurement after the move.
        start_target, start_dist = None, 0.0
        if self.chk_start_saved.isChecked():
            if not self.start_pose:
                QMessageBox.warning(self, "Start pose",
                                    "No saved start pose.\nJog the robot and press [Save current pose], or "
                                    "untick 'Start from the saved pose'.")
                return
            start_target = list(self.start_pose["tcp_pose"])
            start_dist = float(np.linalg.norm(np.subtract(st.tcp_pose[:3], start_target[:3]))) * 1000.0
        away = start_dist > 2.0

        obs = self._last_obs
        if not away:
            if obs is None:
                QMessageBox.warning(self, "Marker", "The marker is not detected in the image.")
                return
            if self.kind == "depth" and obs.point_cam is None:
                QMessageBox.warning(self, "Marker", "The marker is visible but has no 3D points. Check distance / reflections.")
                return

        self._collect_ui()
        if not away:
            fr = self.camera.get_latest()
            h, w = fr.image.shape[:2]
            cx, cy = self._optical_centre(w, h)
            dist = math.hypot(obs.pixel[0] - cx, obs.pixel[1] - cy)
            tol = self._centering_tolerance(w)
            if dist > tol and not self._ask(
                    "Marker position",
                    f"The marker centre is {dist:.0f} px from the image centre (allowed {tol:.0f} px).\n"
                    "It may leave the image while the grid is visited.\n\nStart anyway?"):
                return
        ref_text = OFFSET_REFERENCES[self.cfg["marker"]["offset_reference"]]
        if all(abs(float(v)) < 1e-9 for v in self.cfg["marker"]["offset_from_flange_mm"]) and not self._ask(
                "Marker position",
                f"The marker position is [0, 0, 0] mm ({ref_text}).\n"
                "That is only correct if the marker centre is exactly at the reference.\n"
                "If wrong, the camera position t is off by the same amount.\n\nContinue?"):
            return
        if (self.cfg["marker"]["offset_reference"] == "tcp" and any(abs(v) > 1e-6 for v in st.tcp_offset[:3])
                and not self._ask(
                    "Active TCP",
                    "The marker position was entered relative to the active TCP.\n"
                    f"Active TCP offset on the pendant now: {pm.format_mm(st.tcp_offset[:3])}\n"
                    "Was the value measured relative to this TCP?")):
            return

        sc, rc = self.cfg["sequence"], self.cfg["robot"]
        side = sc["camera_side"]
        axes = mf.view_axes(side)
        n = int(sc["points_per_axis"]) ** 3
        ranges = "\n".join(
            f"  {name} ±{sc['step_mm'][name]:g} mm  (base {mf.base_axis_label(axes[:, i])})"
            for i, name in enumerate(mf.VIEW_AXES))
        if not self._ask(
                "Start calibration",
                f"Camera: {settings.CAMERA_LABELS[self.kind]}\n"
                f"Result file: {settings.result_path(self.cfg, self.kind)}\n"
                f"Camera position: {SIDE_LABELS[side]} side of the base\n"
                + ("Start pose: current pose\n" if start_target is None else
                   f"Start pose: saved pose — first a {start_dist:.0f} mm linear move\n" if away else
                   "Start pose: saved pose (current pose)\n")
                + f"\nThe robot visits {n} poses around the start pose "
                f"at {float(rc['velocity']) * 1000:.0f} mm/s.\n{ranges}\n\n"
                "Clear the area around the tool and keep a hand on the emergency stop.\n\nStart?"):
            return

        self.log_box.clear()
        self.lbl_result.setText("")
        self.progress.setMaximum(n)
        self.progress.setValue(0)
        self.worker = CalibrationWorker(self.robot, self.camera, self.detector, self.cfg, self.kind, self,
                                        start_target=start_target)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_run.connect(self._on_finished)
        self._set_editing_enabled(False)
        self.btn_stop.setEnabled(True)
        self.worker.start()

    def _on_stop(self) -> None:
        if self._running():
            self._append_log("Stop requested - stopping the robot ...")
            self.worker.request_stop()
        elif self.robot is not None:
            self._jog_active = False
            self.robot.stop_motion()
            self._append_log("Stop command sent")

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(done)

    def _on_finished(self, ok: bool, msg: str) -> None:
        self._set_editing_enabled(True)
        self.btn_stop.setEnabled(False)
        self._show_result_file()
        color = "#080" if ok else "#c00"
        self.lbl_result.setText(f"<span style='color:{color}'><b>{'Done' if ok else 'Not completed'}</b></span> "
                                + msg.replace("\n", "<br>"))
        if ok:
            QMessageBox.information(self, "Calibration done", msg)
        else:
            QMessageBox.warning(self, "Calibration not completed", msg)

    def closeEvent(self, event) -> None:
        if self._running():
            if not self._ask("Close", "Calibration is running. Stop the robot and close?"):
                event.ignore()
                return
            self.worker.request_stop()
            self.worker.wait(10000)
        if self._starter is not None:
            self._starter.wait(10000)
        if self._jog_active:
            self._jog_stop()
        self.jog_timer.stop()
        self.view_timer.stop()
        self.status_timer.stop()
        self._collect_ui()
        try:
            if self.camera is not None:
                self.camera.stop()
        finally:
            if self.robot is not None:
                self.robot.close()
        event.accept()


# ======================================================================
# Entry point
# ======================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=APP_TITLE)
    ap.add_argument("--config", default=None, help="config file (default: the package's config/calibration.yaml)")
    ap.add_argument("--sim", action="store_true", help="run with a simulated robot + simulated camera (no hardware needed)")
    args = ap.parse_args(argv)

    app = QApplication(sys.argv[:1])
    config_path = Path(args.config) if args.config else settings.default_config_path()
    try:
        cfg = settings.load_config(config_path)
        restored = settings.apply_ui_state(cfg)
    except Exception as e:
        QMessageBox.critical(None, APP_TITLE, f"Config error:\n{e}")
        return 1

    info = [f"Config: {config_path}"]
    if restored:
        info.append(f"Loaded the last inputs: {settings.output_dir(cfg) / settings.UI_STATE_FILE}")
    info.append(f"Results folder: {settings.output_dir(cfg)}")
    info.append(f"Marker: {cfg['marker']['dictionary']} id {cfg['marker']['id']}")
    win = CalibrationWindow(cfg, sim=args.sim, info_lines=info)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
