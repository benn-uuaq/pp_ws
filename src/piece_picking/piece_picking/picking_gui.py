"""
picking_gui.py - GUI that moves the robot pick → place → home from a vision result (product position + surface normal).

Separate from the calibration program. It only reads the calibration result files (base_T_cam).
All robot motion is servoj (servo_motion.py — script loop on the robot side, same approach as dusan_v5).

Flow
  1. Robot IP -> [Connect robot]
  2. Both calibration results (RGB / depth) are loaded automatically at start; tick the one to use
  3. Jog to the home / pick standby / place poses with the pendant and press [Save Home] / [Save Pick standby] / [Save Place]
  4. Tick the vision input mode: manual entry / vision node trigger (trigger not implemented yet, TODO)
  5. Enter the product position (X, Y, Z m in the same camera frame as the selected result) + normal (NX, NY, NZ) — or fill them with [Paste vision JSON],
     with [Apply rotation] the tool is tilted along the normal, otherwise it picks vertically with the tool orientation unchanged
     -> tool orientation, suction tip and TCP targets are computed and shown immediately
  6. [Pick → Place → Home]: (home) -> (pick standby) -> approach -> product -> grip -> retreat along the tool
     -> lift along base +Z (tool still tilted, clears the box) -> place approach -> place
     -> release -> retreat -> home.  Approach and retreat run along the tool axis (suction direction).
     Suction outputs DO4~7 together, normally ON / OFF while gripping (config gripper).
     [Stop] is stopl.

TODO: the products are inside a box, so an approach path that avoids the box walls (now a straight approach along the tool axis).

Run:
  ros2 run piece_picking picking_gui
  ros2 run piece_picking picking_gui --sim     # simulated robot
"""

from __future__ import annotations

import argparse
import datetime
import ipaddress
import sys
import threading
import traceback
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QTextEdit,
    QVBoxLayout, QWidget,
)

from robot_camera_calibration import pose_math as pm
from piece_picking import picking_core as pc
from piece_picking import picking_settings as settings
from piece_picking.servo_motion import ServoMover
from piece_picking.vision_client import (
    MAX_POSITION_M, NORMAL_DECIMALS, POSITION_DECIMALS, VisionClient, VisionResult, parse_vision_candidates,
)

APP_TITLE = "Piece Picking - pick & place from vision"
NEAR_MM = 5.0       # within this distance the robot counts as already at that pose

BUTTON_STYLE = """
QPushButton { color: white; background-color: rgb(0, 119, 196);
              font: bold 12pt; border: 1px solid black; padding: 6px; }
QPushButton:pressed { background-color: rgb(0, 80, 150); }
QPushButton:disabled { background-color: rgb(150, 150, 150); color: rgb(225, 225, 225); }
"""
STOP_STYLE = BUTTON_STYLE.replace("rgb(0, 119, 196)", "rgb(200, 40, 40)").replace(
    "rgb(0, 80, 150)", "rgb(150, 20, 20)")
CYCLE_STYLE = BUTTON_STYLE.replace("rgb(0, 119, 196)", "rgb(20, 140, 70)").replace(
    "rgb(0, 80, 150)", "rgb(10, 100, 50)")
SMALL_BUTTON_STYLE = BUTTON_STYLE.replace("font: bold 12pt", "font: bold 10pt")

# One motion step: (kind, name, value)
#   ("line", name, (TCP pose, speed m/s))      servoj linear
#   ("joint", name, (joints, TCP pose to check)) servoj joint interpolation (linear if there are no joints)
#   ("io", name, grip/release)                 suction digital outputs (ON/OFF from gripper.grip_level) + wait
Step = Tuple[str, str, object]


def _do_names(outs: Sequence[int]) -> str:
    return ", ".join(f"DO{i}" for i in outs)


# ======================================================================
# Motion job (separate thread)
# ======================================================================
class MotionWorker(QThread):
    log = pyqtSignal(str)
    finished_run = pyqtSignal(bool, str)

    def __init__(self, robot, steps: List[Step], cfg: dict, parent=None):
        super().__init__(parent)
        self.robot = robot
        self.steps = steps
        self.cfg = cfg
        self._cancel = threading.Event()

    def request_stop(self) -> None:
        self._cancel.set()
        try:
            self.robot.stop_motion()
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        self.log.emit(f"{datetime.datetime.now():%H:%M:%S} {msg}")

    def run(self) -> None:
        try:
            ok, msg = self._run()
        except Exception as e:
            ok, msg = False, f"Unexpected error: {e}"
            self._log(traceback.format_exc())
        self.finished_run.emit(ok, msg)

    def _run(self):
        rc, gc = self.cfg["robot"], self.cfg["gripper"]
        mover = ServoMover(self.robot, self.cfg["servo"], rc["rotation_convention"],
                           float(rc["position_tolerance_mm"]), self._log)
        for kind, name, value in self.steps:
            if self._cancel.is_set():
                return False, "Stopped by the user."
            if kind == "io":
                outs = settings.suction_outputs(self.cfg)
                if not outs:
                    self._log(f"{name}: suction outputs not configured, skipped (gripper.suction_do)")
                    continue
                level = bool(gc["grip_level"]) == bool(value)        # grip = grip_level, release = the opposite
                self._log(f"{name}: {_do_names(outs)} = {'ON' if level else 'OFF'}")
                if not mover.digital_out(outs, level):
                    return False, f"{name} failed: {self.robot.last_error}"
                if self._cancel.wait(float(gc["grip_wait_s" if value else "release_wait_s"])):
                    return False, "Stopped by the user."
                continue
            if kind == "line":
                pose, speed = value
                self._log(f"Moving to {name} (servoj linear {speed * 1000:.0f} mm/s): {pc.format_pose(pose)}")
                ok, why = mover.line(pose, speed, cancel=self._cancel.is_set)
            else:
                joints, pose = value
                if joints is None:
                    self._log(f"Moving to {name} (no joints -> servoj linear): {pc.format_pose(pose)}")
                    ok, why = mover.line(pose, float(rc["velocity"]), cancel=self._cancel.is_set)
                else:
                    self._log(f"Moving to {name} (servoj joint): {pc.format_pose(pose)}")
                    ok, why = mover.joint(joints, pose, cancel=self._cancel.is_set)
            for a in self.robot.pop_alarms():
                self._log(f"  [ROBOT ALARM] {a}")
            if not ok:
                return False, f"{name} not reached: {why}"
            self._log(f"  {name} reached")
        return True, " → ".join(name for _, name, _ in self.steps) + " done"


def _valid_ip(text: str) -> bool:
    try:
        ipaddress.ip_address(text.strip())
        return True
    except ValueError:
        return False


def _spin(lo: float, hi: float, value: float, decimals: int = 1, suffix: str = " mm",
          step: float = 1.0) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setDecimals(decimals)
    sp.setSingleStep(step)
    sp.setSuffix(suffix)
    sp.setValue(float(value))
    return sp


def _retreat(pose: Sequence[float], distance_m: float, convention: str) -> List[float]:
    """Pose retreated by distance against the tool axis (suction direction +Z)."""
    z = pm.rotation_from_pose(pose, convention)[:, 2]
    return list(np.subtract(pose[:3], z * distance_m)) + [float(v) for v in pose[3:6]]


# ======================================================================
# Main window
# ======================================================================
class PickingWindow(QWidget):
    def __init__(self, cfg: dict, sim: bool = False, info_lines=()):
        super().__init__()
        self.cfg = cfg
        self.sim = sim
        self.robot = None
        self.calibs = {kind: None for kind in pc.CALIB_KINDS}     # calibration result per camera
        self.calib_errors = {kind: "" for kind in pc.CALIB_KINDS}
        self.vision = VisionClient(cfg["vision"]["trigger_topic"], cfg["vision"]["result_topic"],
                                   float(cfg["vision"]["timeout_s"]))
        self.worker: Optional[MotionWorker] = None
        self.plan: Optional[pc.PickPlan] = None
        self.plan_error = ""
        self.return_pose: Optional[List[float]] = None     # pose to return to when no home is saved
        self.poses = settings.load_poses(cfg)                # home / place

        self.setWindowTitle(APP_TITLE + ("  [SIMULATION]" if sim else ""))
        self._build_ui()
        for line in info_lines:
            self._append_log(line)
        for kind in pc.CALIB_KINDS:
            self._load_calibration(kind, self.cfg["calibration"]["result_files"][kind])
        self._on_calib_choice_changed()
        self._on_mode_changed()
        self._show_poses()

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(250)

    @property
    def conv(self) -> str:
        return self.cfg["robot"]["rotation_convention"]

    # ---- UI ----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        left_panel = QWidget()
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 4, 0)
        right = QVBoxLayout()
        mono = "font-family: monospace;"

        # robot
        box = QGroupBox("Robot")
        g = QGridLayout(box)
        self.edit_robot_ip = QLineEdit(str(self.cfg["robot"]["ip"]))
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
        left.addWidget(box)

        # calibration results (use one of RGB / depth)
        box = QGroupBox("Calibration result (select the one to use)")
        v = QVBoxLayout(box)
        self.calib_group = QButtonGroup(self)
        self.calib_group.setExclusive(True)
        self.calib_checks, self.lbl_calibs = {}, {}
        for kind in pc.CALIB_KINDS:
            cb = QCheckBox(pc.CALIB_LABELS[kind])
            self.calib_group.addButton(cb)
            self.calib_checks[kind] = cb
            lbl = QLabel("-")
            lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lbl.setContentsMargins(22, 0, 0, 6)
            self.lbl_calibs[kind] = lbl
            v.addWidget(cb)
            v.addWidget(lbl)
        self.calib_checks[self.cfg["calibration"]["use"]].setChecked(True)
        self.calib_group.buttonToggled.connect(self._on_calib_choice_changed)
        h = QHBoxLayout()
        self.btn_calib_reload = QPushButton("Reload")
        self.btn_calib_pick = QPushButton("Change the selected result file ...")
        self.btn_calib_reload.clicked.connect(self._reload_calibrations)
        self.btn_calib_pick.clicked.connect(self._on_pick_calibration)
        h.addWidget(self.btn_calib_reload)
        h.addWidget(self.btn_calib_pick)
        v.addLayout(h)
        left.addWidget(box)

        # vision result (product position + surface normal)
        box = QGroupBox("Vision result (product position + normal)")
        g = QGridLayout(box)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.chk_manual = QCheckBox("Manual entry")
        self.chk_trigger = QCheckBox("Vision node trigger (not implemented)")
        for cb in (self.chk_manual, self.chk_trigger):
            self.mode_group.addButton(cb)
        (self.chk_trigger if self.cfg["vision"]["mode"] == "trigger" else self.chk_manual).setChecked(True)
        self.mode_group.buttonToggled.connect(self._on_mode_changed)
        g.addWidget(self.chk_manual, 0, 0, 1, 2)
        g.addWidget(self.chk_trigger, 0, 2, 1, 4)

        self.lbl_frame = QLabel("-")
        self.lbl_frame.setWordWrap(True)
        g.addWidget(QLabel("Frame"), 1, 0)
        g.addWidget(self.lbl_frame, 1, 1, 1, 5)

        self.spin_point, self.spin_normal = [], []
        for col, name in enumerate("XYZ"):
            sp = _spin(-MAX_POSITION_M, MAX_POSITION_M, self.cfg["vision"]["manual_point_m"][col],
                       POSITION_DECIMALS, " m", 0.001)
            sp.valueChanged.connect(self._recompute)
            self.spin_point.append(sp)
            g.addWidget(QLabel(name), 2, 2 * col)
            g.addWidget(sp, 2, 2 * col + 1)
        for col, name in enumerate(("NX", "NY", "NZ")):
            sp = _spin(-1.0, 1.0, self.cfg["vision"]["manual_normal"][col], NORMAL_DECIMALS, "", 0.01)
            sp.valueChanged.connect(self._recompute)
            self.spin_normal.append(sp)
            g.addWidget(QLabel(name), 3, 2 * col)
            g.addWidget(sp, 3, 2 * col + 1)
        self.chk_apply_normal = QCheckBox("Apply rotation — tilt the tool along the normal when picking")
        self.chk_apply_normal.setChecked(bool(self.cfg["pick"]["apply_normal"]))
        self.chk_apply_normal.toggled.connect(self._recompute)
        g.addWidget(self.chk_apply_normal, 4, 0, 1, 6)
        hint = QLabel("X/Y/Z = vision position_m [m], NX/NY/NZ = vision normal as is (direction in the camera frame, sign / length "
                      "handled automatically). [Paste vision JSON] = fill from a copied vision result. If Apply rotation is off "
                      "or the normal is 0, it picks vertically with the current tool orientation.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        g.addWidget(hint, 5, 0, 1, 6)

        self.btn_paste = QPushButton("Paste vision JSON")
        self.btn_paste.setStyleSheet(SMALL_BUTTON_STYLE)
        self.btn_paste.clicked.connect(self._on_paste_json)
        self.btn_capture = QPushButton("Capture with camera")
        self.btn_capture.setStyleSheet(SMALL_BUTTON_STYLE)
        self.btn_capture.clicked.connect(self._on_capture)
        g.addWidget(self.btn_paste, 6, 0, 1, 3)
        g.addWidget(self.btn_capture, 6, 3, 1, 3)
        self.lbl_vision = QLabel(self.vision.describe())
        self.lbl_vision.setWordWrap(True)
        self.lbl_vision.setStyleSheet("color: #555;")
        g.addWidget(self.lbl_vision, 7, 0, 1, 6)
        left.addWidget(box)

        # gripper (suction) — the TCP target is pulled back along tool +Z by this much so the suction tip lands on the product
        box = QGroupBox("Gripper (suction)")
        g = QGridLayout(box)
        self.spin_gripper = _spin(0.0, 500.0, self.cfg["pick"]["gripper_length_mm"], 1)
        self.spin_gripper.valueChanged.connect(self._recompute)
        g.addWidget(QLabel("Length"), 0, 0)
        g.addWidget(self.spin_gripper, 0, 1)
        hint = QLabel("Distance from the pendant's active TCP to the suction pad tip along tool +Z. If the active TCP is the flange (0), "
                      "it is flange face → pad tip. If the active TCP already includes the tool length, use 0.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        g.addWidget(hint, 1, 0, 1, 2)
        self.lbl_gripper = QLabel("-")
        self.lbl_gripper.setWordWrap(True)
        g.addWidget(self.lbl_gripper, 2, 0, 1, 2)
        outs = settings.suction_outputs(self.cfg)
        grip = "ON" if self.cfg["gripper"]["grip_level"] else "OFF"
        idle = "OFF" if self.cfg["gripper"]["grip_level"] else "ON"
        self.lbl_suction = QLabel(f"Suction outputs: {_do_names(outs)} together — grip = {grip}, idle / release = {idle}" if outs else
                                  "Suction outputs: <span style='color:#c60'>not configured</span> (config gripper.suction_do) — "
                                  "grip / release are skipped in the pick & place sequence")
        self.lbl_suction.setWordWrap(True)
        g.addWidget(self.lbl_suction, 3, 0, 1, 2)
        row = QHBoxLayout()
        self.btn_grip = QPushButton(f"Grip ({grip})")
        self.btn_release = QPushButton(f"Release ({idle})")
        for b, grip_on in ((self.btn_grip, True), (self.btn_release, False)):
            b.setStyleSheet(SMALL_BUTTON_STYLE)
            b.clicked.connect(lambda _=False, v=grip_on: self._on_suction(v))
            row.addWidget(b)
        g.addLayout(row, 4, 0, 1, 2)
        g.setColumnStretch(1, 1)
        left.addWidget(box)
        left.addStretch(1)

        # target (right side)
        box = QGroupBox("Picking target (robot base frame)")
        g = QGridLayout(box)
        self.lbl_target = QLabel("-")
        self.lbl_target.setWordWrap(True)
        self.lbl_target.setStyleSheet(mono)
        self.lbl_target.setTextInteractionFlags(Qt.TextSelectableByMouse)
        g.addWidget(self.lbl_target, 0, 0, 1, 6)
        self.spin_approach = _spin(0.0, 500.0, self.cfg["pick"]["approach_mm"], 0, step=10.0)
        self.spin_offset = _spin(-50.0, 300.0, self.cfg["pick"]["final_offset_mm"], 1)
        self.spin_speed = _spin(5.0, 250.0, float(self.cfg["robot"]["velocity"]) * 1000.0, 0, " mm/s", 5.0)
        self.spin_retreat = _spin(0.0, 300.0, self.cfg["pick"]["retreat_mm"], 0, step=10.0)
        self.spin_lift = _spin(0.0, 500.0, self.cfg["pick"]["lift_mm"], 0, step=10.0)
        for sp in (self.spin_approach, self.spin_offset, self.spin_retreat, self.spin_lift):
            sp.valueChanged.connect(self._recompute)
        g.addWidget(QLabel("Approach"), 2, 0)
        g.addWidget(self.spin_approach, 2, 1)
        g.addWidget(QLabel("Final offset"), 2, 2)
        g.addWidget(self.spin_offset, 2, 3)
        g.addWidget(QLabel("Speed"), 2, 4)
        g.addWidget(self.spin_speed, 2, 5)
        g.addWidget(QLabel("Retreat (tool)"), 3, 0)
        g.addWidget(self.spin_retreat, 3, 1)
        g.addWidget(QLabel("Lift (base +Z)"), 3, 2)
        g.addWidget(self.spin_lift, 3, 3)
        note = QLabel(f"All motion is servoj (pendant speed ratio does not apply). The last segment before the product / placing runs at "
                      f"{float(self.cfg['pick']['descent_speed_ratio']) * 100:.0f} % of the speed. Approach follows the tool axis; after the pick "
                      "the robot backs off along the tool axis, then lifts straight up with the tool still tilted, and only then turns back. "
                      "TODO: approach path that avoids the box walls.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #a60;")
        g.addWidget(note, 4, 0, 1, 6)
        right.addWidget(box)

        # work poses (home / pick standby / place)
        box = QGroupBox("Work poses (home / pick standby / place)")
        g = QGridLayout(box)
        self.lbl_poses, self.btn_save_pose, self.btn_goto_pose = {}, {}, {}
        for row, name in enumerate(settings.POSE_NAMES):
            label = settings.POSE_NAMES[name]
            g.addWidget(QLabel(f"<b>{label}</b>"), row, 0)
            lbl = QLabel("-")
            lbl.setStyleSheet(mono)
            lbl.setWordWrap(True)
            self.lbl_poses[name] = lbl
            g.addWidget(lbl, row, 1)
            save = QPushButton(f"Save {label}")
            goto = QPushButton(f"Go to {label}")
            save.clicked.connect(lambda _=False, n=name: self._on_save_pose(n))
            goto.clicked.connect(lambda _=False, n=name: self._on_goto_pose(n))
            self.btn_save_pose[name], self.btn_goto_pose[name] = save, goto
            g.addWidget(save, row, 2)
            g.addWidget(goto, row, 3)
        self.chk_start_home = QCheckBox("Start picking from home (go home first if not there)")
        self.chk_start_home.setChecked(bool(self.cfg["pick"]["start_from_home"]))
        self.chk_start_home.toggled.connect(self._recompute)          # changes the tool orientation reference (home / current)
        g.addWidget(self.chk_start_home, len(settings.POSE_NAMES), 0, 1, 4)
        hint = QLabel("Pick standby (optional): visited right before the approach point. The tool turns towards the "
                      "product only on the way from the standby pose to the approach point.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        g.addWidget(hint, len(settings.POSE_NAMES) + 1, 0, 1, 4)
        g.setColumnStretch(1, 1)
        right.addWidget(box)

        grid = QGridLayout()
        self.btn_approach = QPushButton("Move to approach")
        self.btn_target = QPushButton("Move to product")
        self.btn_cycle = QPushButton("Pick → Place → Home")
        self.btn_return = QPushButton("Retreat up, then home")
        self.btn_stop = QPushButton("Stop")
        for b in (self.btn_approach, self.btn_target, self.btn_return):
            b.setStyleSheet(BUTTON_STYLE)
        self.btn_cycle.setStyleSheet(CYCLE_STYLE)
        self.btn_stop.setStyleSheet(STOP_STYLE)
        self.btn_approach.clicked.connect(lambda: self._on_move("approach"))
        self.btn_target.clicked.connect(lambda: self._on_move("target"))
        self.btn_cycle.clicked.connect(self._on_cycle)
        self.btn_return.clicked.connect(self._on_return)
        self.btn_stop.clicked.connect(self._on_stop)
        for b in (self.btn_approach, self.btn_target, self.btn_cycle, self.btn_return, self.btn_stop):
            b.setMinimumHeight(44)
        grid.addWidget(self.btn_approach, 0, 0)
        grid.addWidget(self.btn_target, 0, 1)
        grid.addWidget(self.btn_cycle, 1, 0, 1, 2)
        grid.addWidget(self.btn_return, 2, 0)
        grid.addWidget(self.btn_stop, 2, 1)
        right.addLayout(grid)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("QTextEdit { background: white; color: black; "
                                   "font-family: monospace; font-size: 9pt; }")
        right.addWidget(self.log_box, 1)

        scroll = QScrollArea()
        scroll.setWidget(left_panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(left_panel.minimumSizeHint().width()
                               + scroll.verticalScrollBar().sizeHint().width() + 4)
        root.addWidget(scroll, 1)
        root.addLayout(right, 1)
        self.resize(1300, 860)

    def _append_log(self, text: str) -> None:
        self.log_box.append(text)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    # ---- settings / inputs -----------------------------------------------------------
    def _mode(self) -> str:
        return "trigger" if self.chk_trigger.isChecked() else "manual"

    def _use(self) -> str:
        return next((k for k, cb in self.calib_checks.items() if cb.isChecked()), self.cfg["calibration"]["use"])

    def _frame(self) -> str:
        """Vision frame = camera of the selected calibration result."""
        return pc.KIND_TO_FRAME[self._use()]

    def _point_m(self) -> List[float]:
        return [float(sp.value()) for sp in self.spin_point]

    def _point_mm(self) -> List[float]:
        return [v * 1000.0 for v in self._point_m()]

    def _normal(self) -> List[float]:
        return [float(sp.value()) for sp in self.spin_normal]

    def _reference_pose(self, current: Sequence[float]) -> Tuple[List[float], str]:
        """Tool orientation reference = the pose right before the approach point:
        the pick standby pose if saved, else home when starting from home, else the current pose."""
        if "standby" in self.poses:
            return list(self.poses["standby"]["tcp_pose"]), "pick standby"
        if self.chk_start_home.isChecked() and "home" in self.poses:
            return list(self.poses["home"]["tcp_pose"]), "home"
        return list(current), "current"

    def _collect_ui(self) -> None:
        self.cfg["robot"]["ip"] = self.edit_robot_ip.text().strip()
        self.cfg["robot"]["velocity"] = float(self.spin_speed.value()) / 1000.0
        self.cfg["calibration"]["use"] = self._use()
        self.cfg["vision"]["mode"] = self._mode()
        self.cfg["vision"]["manual_point_m"] = self._point_m()
        self.cfg["vision"]["manual_normal"] = self._normal()
        self.cfg["pick"]["approach_mm"] = float(self.spin_approach.value())
        self.cfg["pick"]["final_offset_mm"] = float(self.spin_offset.value())
        self.cfg["pick"]["retreat_mm"] = float(self.spin_retreat.value())
        self.cfg["pick"]["lift_mm"] = float(self.spin_lift.value())
        self.cfg["pick"]["gripper_length_mm"] = float(self.spin_gripper.value())
        self.cfg["pick"]["apply_normal"] = self.chk_apply_normal.isChecked()
        self.cfg["pick"]["start_from_home"] = self.chk_start_home.isChecked()
        try:
            settings.save_ui_state(self.cfg)
        except Exception as e:
            self._append_log(f"[WARN] Failed to save inputs: {e}")

    def _on_mode_changed(self, *_args) -> None:
        manual = self._mode() == "manual"
        for sp in self.spin_point + self.spin_normal:
            sp.setReadOnly(not manual)
            sp.setButtonSymbols(QDoubleSpinBox.UpDownArrows if manual else QDoubleSpinBox.NoButtons)
        self.btn_capture.setEnabled(not manual and not self._running())
        self.btn_paste.setEnabled(manual and not self._running())

    # ---- calibration results ------------------------------------------------------
    def _load_calibration(self, kind: str, path, quiet: bool = True) -> None:
        """Read the result file for a camera kind. Both are loaded automatically at start."""
        try:
            self.calibs[kind] = pc.load_calibration(path, expected=kind)
            self.calib_errors[kind] = ""
            self.cfg["calibration"]["result_files"][kind] = str(self.calibs[kind].path)
            self._append_log(f"{pc.CALIB_LABELS[kind]} loaded: {self.calibs[kind].path}")
        except Exception as e:
            if self.sim:
                self.calibs[kind] = self._sim_calibration(kind)
                self.calib_errors[kind] = ""
                self._append_log(f"[SIM] using simulated values instead of the {pc.CALIB_LABELS[kind]} file ({e})")
            else:
                self.calibs[kind] = None
                self.calib_errors[kind] = f"{path}<br>{e}"
                self._append_log(f"[WARN] Could not load the {pc.CALIB_LABELS[kind]}: {e}")
                if not quiet:
                    QMessageBox.warning(self, "Calibration", f"Could not load the result file.\n\n{path}\n{e}")
        self._show_calibrations()
        self._recompute()

    def _reload_calibrations(self) -> None:
        for kind in pc.CALIB_KINDS:
            self._load_calibration(kind, self.cfg["calibration"]["result_files"][kind])

    def _show_calibrations(self) -> None:
        for kind, lbl in self.lbl_calibs.items():
            calib = self.calibs[kind]
            if calib is not None:
                lbl.setText("<span style='color:#080'>loaded</span><br>" + "<br>".join(calib.summary_lines()))
            else:
                lbl.setText("<span style='color:#c00'>missing</span> " + (self.calib_errors[kind] or "-"))

    def _on_calib_choice_changed(self, *_args) -> None:
        use = self._use()
        self.lbl_frame.setText(f"<b>{pc.FRAME_LABELS[pc.KIND_TO_FRAME[use]]}</b> "
                               f"(enter in the same camera frame as the selected {pc.CALIB_LABELS[use]})")
        self._recompute()

    def _sim_calibration(self, kind: str) -> pc.CalibrationResult:
        from robot_camera_calibration import sim_camera
        R, t = sim_camera.true_camera_pose("+y")
        if kind == "rgb":
            R, t = R @ sim_camera.SIM_RGB_FROM_TOF_R, t + R @ sim_camera.SIM_RGB_FROM_TOF_T
        return pc.CalibrationResult(Path(f"(simulation {kind})"), R, t, {"created": "simulation"}, kind)

    def _on_pick_calibration(self) -> None:
        kind = self._use()
        start = str(Path(self.cfg["calibration"]["result_files"][kind]).expanduser().parent)
        path, _ = QFileDialog.getOpenFileName(self, pc.CALIB_LABELS[kind], start, "YAML (*.yaml *.yml)")
        if path:
            self._load_calibration(kind, path, quiet=False)
            self._collect_ui()

    # ---- target computation ---------------------------------------------------------------
    def _state(self):
        return None if self.robot is None else self.robot.get_state()

    def _current_pose(self) -> Optional[List[float]]:
        st = self._state()
        return None if st is None else list(st.tcp_pose)

    def _gripper_warning(self) -> str:
        """Warn if the gripper length and the pendant's active TCP disagree (applied twice / both 0)."""
        st = self._state()
        return "" if st is None else pc.gripper_tcp_warning(st.tcp_offset, float(self.spin_gripper.value()))

    def _update_gripper_info(self) -> None:
        st = self._state()
        text = ("Active TCP offset: - (shown after connecting the robot)" if st is None else
                "Pendant active TCP offset (relative to the flange): ["
                + ", ".join(f"{v * 1000:.1f}" for v in st.tcp_offset[:3]) + "] mm")
        warning = self._gripper_warning()
        if warning:
            text += f"<br><span style='color:#c60'><b>Warning:</b> {warning}</span>"
        self.lbl_gripper.setText(text)

    def _recompute(self, *_args) -> None:
        self.plan, self.plan_error = None, ""
        lines = []
        calib = self.calibs.get(self._use())
        if calib is None:
            self.plan_error = f"{pc.CALIB_LABELS[self._use()]} is missing. Create it with the calibration program first."
        else:
            try:
                product, how = pc.camera_point_to_base(self._point_mm(), self._frame(), calib)
                lines.append(f"Product  {pc.format_xyz_mm(product)}  ({how})")
                d_base, flipped = pc.camera_normal_to_base(self._normal(), self._point_mm(), self._frame(), calib)
                if d_base is None:
                    lines.append("normal: none (0, 0, 0)")
                else:
                    lines.append("normal → approach direction [" + ", ".join(f"{v:.3f}" for v in d_base) + "] "
                                 + pc.describe_direction(d_base)
                                 + (" — it pointed towards the camera (outwards), so it was flipped" if flipped else ""))
                pose = self._current_pose()
                if pose is None:
                    self.plan_error = "Robot TCP not known yet (the current TCP is needed for the tool orientation)."
                else:
                    ref, ref_name = self._reference_pose(pose)
                    tilt = d_base if self.chk_apply_normal.isChecked() else None
                    self.plan = pc.plan_pick(product, ref, self.spin_approach.value(), self.spin_offset.value(),
                                             self.spin_gripper.value(), self.conv, tilt,
                                             self.spin_retreat.value(), self.spin_lift.value())
                    if self.plan.mode == "tilt":
                        lines.append(f"Tool: {pc.ORIENTATION_LABELS['tilt']} — {self.plan.tilt_deg:.1f}° from vertical, "
                                     f"{self.plan.rotation_change_deg:.1f}° rotation from the {ref_name} pose")
                    else:
                        why = ("Apply rotation is off" if not self.chk_apply_normal.isChecked() else "no normal")
                        lines.append(f"Tool: {ref_name} tool orientation unchanged — vertical pick ({why})")
                    lines.append(f"Suction tip  approach {pc.format_xyz_mm(self.plan.tip_approach)} → "
                                 f"target {pc.format_xyz_mm(self.plan.tip_target)}")
                    lines.append(f"Approach (TCP)  {pc.format_pose(self.plan.approach_pose)}")
                    lines.append(f"Target (TCP)    {pc.format_pose(self.plan.target_pose)}")
                    lines.append(f"After pick: retreat {self.spin_retreat.value():.0f} mm along the tool → lift "
                                 f"{self.spin_lift.value():.0f} mm (base +Z) to {pc.format_xyz_mm(self.plan.lift_pose)} (TCP)")
                    dist = np.linalg.norm(np.subtract(self.plan.approach_pose[:3], pose[:3])) * 1000.0
                    lines.append(f"Gripper {self.plan.gripper_mm:.1f} mm applied, current TCP → approach {dist:.0f} mm")
                    problems = pc.check_plan(self.plan, pose, self.cfg["safety"])
                    st = self._state()
                    if st is not None and pc.gripper_missing(st.tcp_offset, self.plan.gripper_mm):
                        problems.insert(0, "Gripper length is 0. The pendant's active TCP is the flange (0), so enter the gripper "
                                           "length for the suction tip to land on the product (with 0 the gripper collides).")
                    if problems:
                        self.plan_error = "<br>".join(problems)
            except Exception as e:
                self.plan_error = str(e)
        text = "<br>".join(lines)
        if self.plan_error:
            text += ("<br>" if text else "") + f"<span style='color:#c00'>{self.plan_error}</span>"
        self.lbl_target.setText(text or "-")
        self._update_gripper_info()
        self._update_buttons()

    def _update_buttons(self) -> None:
        running = self._running()
        connected = self.robot is not None
        can_move = self.plan is not None and not self.plan_error and not running
        self.btn_approach.setEnabled(can_move)
        self.btn_target.setEnabled(can_move)
        self.btn_cycle.setEnabled(can_move and "place" in self.poses)
        self.btn_return.setEnabled(connected and not running)
        self.btn_stop.setEnabled(connected)
        for b in (self.btn_grip, self.btn_release):
            b.setEnabled(connected and not running and bool(settings.suction_outputs(self.cfg)))
        for name in settings.POSE_NAMES:
            self.btn_save_pose[name].setEnabled(connected and not running)
            self.btn_goto_pose[name].setEnabled(connected and not running and name in self.poses)
        for w in [self.edit_robot_ip, self.btn_robot, self.btn_calib_reload, self.btn_calib_pick,
                  self.chk_manual, self.chk_trigger, self.chk_apply_normal, self.chk_start_home] \
                + list(self.calib_checks.values()):
            w.setEnabled(not running)
        self.btn_capture.setEnabled(self._mode() == "trigger" and not running)
        self.btn_paste.setEnabled(self._mode() == "manual" and not running)

    # ---- robot ----------------------------------------------------------------
    def _on_connect_robot(self) -> None:
        if self._running():
            return
        self._collect_ui()
        ip = self.cfg["robot"]["ip"]
        if self.robot is not None:
            self.robot.close()
            self.robot = None
        if self.sim:
            from robot_camera_calibration.sim_robot import SimRobot
            calib = self.calibs.get(self._use())
            start = (self.poses.get("home", {}).get("tcp_pose")
                     or (calib.meta.get("start_tcp_pose") if calib else None)
                     or [-0.30, 0.40, 0.75, np.pi, 0.0, 0.0])
            self.robot = SimRobot(start)
        else:
            if not _valid_ip(ip):
                QMessageBox.warning(self, "Robot", f"Not a valid IP address: '{ip}'")
                return
            from robot_camera_calibration.robot_interface import RobotInterface
            rc = self.cfg["robot"]
            self.robot = RobotInterface(ip, int(rc["port"]), rc["move_command"], rc["stop_command"])
        ok = self.robot.connect()
        self.return_pose = None
        self._append_log(f"Robot connection {'OK' if ok else 'waiting (retrying in the background)'}: {self.robot.description}"
                         + ("" if ok else f" - {self.robot.last_error}"))

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
                self.lbl_tcp.setText(pc.format_pose(st.tcp_pose).replace(" | ", "<br>"))
            if not self._running():
                for a in self.robot.pop_alarms():
                    self._append_log(f"[ROBOT ALARM] {a}")
        self._show_poses()
        self._recompute()

    # ---- home / place -------------------------------------------------------------
    def _show_poses(self) -> None:
        cur = self._current_pose()
        for name, lbl in self.lbl_poses.items():
            p = self.poses.get(name)
            if not p:
                lbl.setText("not saved")
                continue
            x, y, z = (v * 1000.0 for v in p["tcp_pose"][:3])
            text = f"X {x:.0f} Y {y:.0f} Z {z:.0f} mm ({p.get('saved', '')[5:16].replace('T', ' ')})"
            if cur is not None:
                d = float(np.linalg.norm(np.subtract(cur[:3], p["tcp_pose"][:3]))) * 1000.0
                text += " — reached" if d < NEAR_MM else f" — {d:.0f} mm"
            lbl.setText(text)

    def _on_save_pose(self, name: str) -> None:
        st = self._state()
        if st is None:
            QMessageBox.warning(self, "Save pose", "No robot data.")
            return
        if st.is_moving():
            QMessageBox.warning(self, "Save pose", "The robot is moving. Save after it stops.")
            return
        label = settings.POSE_NAMES[name]
        if name in self.poses and not self._ask(f"Save {label}", f"Replace the saved {label} pose with the current pose?"):
            return
        self.poses = settings.save_pose(self.cfg, name, st.tcp_pose, st.joint_positions, st.tcp_offset)
        self._append_log(f"{label} pose saved: {pc.format_pose(st.tcp_pose)} -> {settings.poses_path(self.cfg)}")
        self._show_poses()
        self._recompute()                              # the tool orientation reference may have changed

    def _pose_step(self, name: str) -> Step:
        """servoj joint move to a saved pose (home / pick standby)."""
        p = self.poses[name]
        return ("joint", settings.POSE_NAMES[name], (p.get("joints"), p["tcp_pose"]))

    def _place_steps(self) -> List[Step]:
        place = self.poses["place"]["tcp_pose"]
        speed = float(self.spin_speed.value()) / 1000.0
        slow = speed * float(self.cfg["pick"]["descent_speed_ratio"])
        above = _retreat(place, float(self.spin_approach.value()) / 1000.0, self.conv)
        return [("line", "Place approach", (above, speed)), ("line", "Place", (place, slow))]

    def _on_goto_pose(self, name: str) -> None:
        if self._running() or name not in self.poses:
            return
        pose = self._robot_ready()
        if pose is None:
            return
        steps = self._place_steps() if name == "place" else [self._pose_step(name)]
        self._start_motion(steps)

    # ---- vision ----------------------------------------------------------------
    def _on_capture(self) -> None:
        try:
            result = self.vision.trigger()
        except NotImplementedError as e:
            self._append_log(f"[TODO] {e}")
            QMessageBox.information(self, "Vision trigger", str(e))
            return
        except Exception as e:
            self._append_log(f"[WARN] Vision trigger failed: {e}")
            QMessageBox.warning(self, "Vision trigger", f"No vision result received.\n\n{e}")
            return
        if result is not None:
            self._fill_vision(result, "Vision trigger")

    def _on_paste_json(self) -> None:
        """Fill the position / normal from a copied vision result (JSON). If the clipboard has none, open a paste dialog."""
        text = QApplication.clipboard().text()
        try:
            results = parse_vision_candidates(text)
        except ValueError:
            results = []
        if not results:
            text = self._ask_json_text(text)
            if text is None:
                return
            try:
                results = parse_vision_candidates(text)
            except ValueError as e:
                QMessageBox.warning(self, "Vision JSON", str(e))
                return
            if not results:
                QMessageBox.warning(self, "Vision JSON", "\"position_m\" not found.")
                return
        result = self._choose_candidate(results)
        if result is not None:
            self._fill_vision(result, "Pasted")

    def _ask_json_text(self, text: str) -> Optional[str]:
        text, ok = QInputDialog.getMultiLineText(
            self, "Paste vision JSON",
            "Paste the vision result (one entry of candidates.json, several entries or the whole file):", text)
        return text if ok else None

    def _choose_candidate(self, results: List[VisionResult]) -> Optional[VisionResult]:
        """If there are several candidates, let the user choose (by rank, rank 1 is the default)."""
        if len(results) == 1:
            return results[0]
        results = sorted(results, key=lambda r: (r.rank is None, r.rank if r.rank is not None else 0))
        items = [f"{r.info or f'#{i + 1}'} | position {np.round(r.position_m, 4).tolist()} m"
                 for i, r in enumerate(results)]
        item, ok = QInputDialog.getItem(self, "Vision candidates", f"{len(results)} candidates — choose one:", items, 0, False)
        return results[items.index(item)] if ok else None

    def _fill_vision(self, result: VisionResult, source: str) -> None:
        if result.frame is not None and result.frame != self._frame():
            self._append_log(f"[WARN] The vision result is in the {pc.FRAME_LABELS.get(result.frame, result.frame)} frame but "
                             f"the {pc.CALIB_LABELS[self._use()]} is selected. Select the result of the same camera.")
        for sp, v in zip(self.spin_point, result.position_m):
            sp.setValue(float(v))
        normal = result.normal if result.normal is not None else [0.0] * 3
        for sp, v in zip(self.spin_normal, normal):
            sp.setValue(float(v))
        self._append_log(f"{source}: position [{', '.join(f'{v:.{POSITION_DECIMALS}f}' for v in result.position_m)}] m, "
                         f"normal [{', '.join(f'{v:.{NORMAL_DECIMALS}f}' for v in normal)}]"
                         + (f" ({result.info})" if result.info else "")
                         + ("" if result.normal is not None else " — no normal (vertical pick)"))
        self._recompute()

    # ---- motion ----------------------------------------------------------------
    def _ask(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No) == QMessageBox.Yes

    def _robot_ready(self) -> Optional[List[float]]:
        if self.robot is None:
            QMessageBox.warning(self, "Robot", "Connect the robot first.")
            return None
        st = self.robot.get_state()
        if st is None:
            QMessageBox.warning(self, "Robot", "No data from the robot.")
            return None
        if st.emergency_stopped or st.protective_stopped:
            QMessageBox.warning(self, "Robot", "The robot is in emergency / protective stop.")
            return None
        return list(st.tcp_pose)

    @staticmethod
    def _near(a: Sequence[float], b: Sequence[float]) -> bool:
        return float(np.linalg.norm(np.subtract(a[:3], b[:3]))) * 1000.0 <= NEAR_MM

    def _checked_plan(self) -> Optional[pc.PickPlan]:
        self._recompute()
        if self.plan is None or self.plan_error:
            QMessageBox.warning(self, "Target", f"Cannot move.\n\n{self.plan_error.replace('<br>', chr(10))}")
            return None
        return self.plan

    def _pick_steps(self, plan: pc.PickPlan, pose: Sequence[float], to_target: bool) -> List[Step]:
        """(home) -> (pick standby) -> approach -> (product).

        Home comes first if "start from home" is on and the robot is not near the pick or the standby pose.
        The pick standby pose (if saved) comes right before the approach point, so the tool turns towards
        the product only on the standby -> approach move.
        """
        speed = float(self.spin_speed.value()) / 1000.0
        slow = speed * float(self.cfg["pick"]["descent_speed_ratio"])
        steps: List[Step] = []
        near_pick = self._near(pose, plan.approach_pose) or self._near(pose, plan.target_pose)
        near_standby = "standby" in self.poses and self._near(pose, self.poses["standby"]["tcp_pose"])
        if (self.chk_start_home.isChecked() and "home" in self.poses and not near_pick and not near_standby
                and not self._near(pose, self.poses["home"]["tcp_pose"])):
            steps.append(self._pose_step("home"))
        if "standby" in self.poses and not near_pick and not near_standby:
            steps.append(self._pose_step("standby"))
        if not (to_target and self._near(pose, plan.target_pose)) and not self._near(pose, plan.approach_pose):
            steps.append(("line", "Approach", (plan.approach_pose, speed)))
        if to_target:
            steps.append(("line", "Product", (plan.target_pose, slow)))
        return steps

    def _pick_note(self, plan: pc.PickPlan) -> str:
        note = (f"Tool: {pc.ORIENTATION_LABELS[plan.mode]} ({plan.tilt_deg:.1f}° from vertical, "
                f"rotation {plan.rotation_change_deg:.1f}°) — straight along tool +Z from the approach point to the product\n"
                f"Gripper length {plan.gripper_mm:.1f} mm applied — suction tip target {pc.format_xyz_mm(plan.tip_target)}")
        warning = self._gripper_warning()
        return note + (f"\n\nWarning: {warning}" if warning else "")

    def _on_move(self, kind: str) -> None:
        if self._running():
            return
        pose = self._robot_ready()
        plan = None if pose is None else self._checked_plan()
        if plan is None:
            return
        steps = self._pick_steps(plan, pose, to_target=(kind == "target"))
        if "home" not in self.poses and not (self._near(pose, plan.approach_pose) or self._near(pose, plan.target_pose)):
            self.return_pose = list(pose)              # without a home, come back to this pose
        self._start_motion(steps, self._pick_note(plan))

    def _on_cycle(self) -> None:
        """One cycle: pick -> place -> home."""
        if self._running():
            return
        if "place" not in self.poses:
            QMessageBox.warning(self, "Pick & place", "No place pose saved.")
            return
        pose = self._robot_ready()
        plan = None if pose is None else self._checked_plan()
        if plan is None:
            return
        speed = float(self.spin_speed.value()) / 1000.0
        slow = speed * float(self.cfg["pick"]["descent_speed_ratio"])
        steps = self._pick_steps(plan, pose, to_target=True)
        steps += [("io", "Grip", True),
                  ("line", "Retreat from product", (plan.retreat_pose, slow))]   # along the tool axis
        if self.spin_lift.value() > 0:
            steps.append(("line", "Lift", (plan.lift_pose, speed)))   # base +Z, tool still tilted: clear the box first
        place_steps = self._place_steps()
        steps += place_steps
        steps += [("io", "Release", False),
                  ("line", "Retreat from place", (place_steps[0][2][0], slow))]
        if "home" in self.poses:
            steps.append(self._pose_step("home"))
        else:
            self._append_log("[WARN] No home saved, so the robot stops at the place retreat pose.")
        self._start_motion(steps, self._pick_note(plan))

    def _on_return(self) -> None:
        """Back off along the tool axis, lift along base +Z, then go home (without a home, to the pose before the pick)."""
        if self._running():
            return
        pose = self._robot_ready()
        if pose is None:
            return
        speed = float(self.spin_speed.value()) / 1000.0
        back = _retreat(pose, float(self.spin_retreat.value()) / 1000.0, self.conv)
        steps: List[Step] = [("line", "Retreat along tool", (back, speed))]
        if self.spin_lift.value() > 0:
            up = list(back)
            up[2] += float(self.spin_lift.value()) / 1000.0
            steps.append(("line", "Lift", (up, speed)))
        if "home" in self.poses:
            steps.append(self._pose_step("home"))
        elif self.return_pose is not None:
            steps.append(("line", "Original pose", (list(self.return_pose), speed)))
        self._start_motion(steps)

    def _on_suction(self, grip: bool) -> None:
        """Switch the grip / release outputs only (for testing, or to drop a product after stopping mid-cycle)."""
        if self._running() or self._robot_ready() is None:
            return
        self._start_motion([("io", "Grip" if grip else "Release", grip)])

    def _start_motion(self, steps: List[Step], note: str = "") -> None:
        self._collect_ui()
        lines = []
        outs = settings.suction_outputs(self.cfg)
        for i, (kind, name, value) in enumerate(steps, start=1):
            if kind == "io":
                level = bool(self.cfg["gripper"]["grip_level"]) == bool(value)
                lines.append(f"  {i}. {name} ({_do_names(outs)} = {'ON' if level else 'OFF'})" if outs
                             else f"  {i}. {name} (outputs not configured — skipped)")
            elif kind == "line":
                lines.append(f"  {i}. {name} (linear {value[1] * 1000:.0f} mm/s): {pc.format_pose(value[0])}")
            else:
                lines.append(f"  {i}. {name} (joint): {pc.format_pose(value[1])}")
        if not self._ask("Robot motion", "The robot will move in this order (all servoj, TCP poses).\n\n" + "\n".join(lines)
                         + "\n\n" + (f"{note}\n\n" if note else "")
                         + "Clear the area and keep a hand on the emergency stop. Proceed?"):
            return
        self.worker = MotionWorker(self.robot, steps, self.cfg, self)
        self.worker.log.connect(self._append_log)
        self.worker.finished_run.connect(self._on_motion_finished)
        self.worker.start()
        self._update_buttons()

    def _on_motion_finished(self, ok: bool, msg: str) -> None:
        self._append_log(("Done: " if ok else "Failed: ") + msg)
        if not ok:
            QMessageBox.warning(self, "Robot motion", msg)
        self._update_buttons()

    def _on_stop(self) -> None:
        if self._running():
            self._append_log("Stop requested - stopping the robot ...")
            self.worker.request_stop()
        elif self.robot is not None:
            self.robot.stop_motion()
            self._append_log("Stop command sent")

    def closeEvent(self, event) -> None:
        if self._running():
            if not self._ask("Close", "The robot is moving. Stop it and close?"):
                event.ignore()
                return
            self.worker.request_stop()
            self.worker.wait(10000)
        self.status_timer.stop()
        self._collect_ui()
        if self.robot is not None:
            self.robot.close()
        event.accept()


# ======================================================================
# Entry point
# ======================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=APP_TITLE)
    ap.add_argument("--config", default=None, help="config file (default: the package's config/picking.yaml)")
    ap.add_argument("--sim", action="store_true", help="run with a simulated robot")
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
    win = PickingWindow(cfg, sim=args.sim, info_lines=info)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
