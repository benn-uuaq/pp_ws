"""
robot_interface.py - Thread-safe robot access for the calibration tool.

Uses the primary port (30001) through
elite_robot_controller.robot.robot_driver.Robot_30001 (same driver as the
robot control node, so there is only one copy in this workspace):
  * a background thread decodes the robot state (actual TCP pose, active TCP
    offset, joint speeds, e-stop / protective stop flags) and robot alarms;
  * motions are sent as one-line scripts (movel) on the same port;
  * "motion finished" is decided from the ACTUAL TCP position and joint speeds,
    not from a running flag, so it also works for scripts sent outside a task.
"""

from __future__ import annotations

import collections
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RobotState:
    tcp_pose: List[float]          # actual TCP pose [m, rad], base frame
    tcp_offset: List[float]        # active TCP offset relative to the flange [m, rad]
    joint_velocity: List[float]    # actual joint speeds [rad/s]
    power_on: bool
    emergency_stopped: bool
    protective_stopped: bool
    timestamp: float               # time.time() when this state was received
    joint_positions: Optional[List[float]] = None   # actual joint angles [rad] (pp_ws)
    program_running: Optional[bool] = None          # robot mode "task running" flag (None if unknown)

    def is_moving(self, velocity_tol: float = 0.01) -> bool:
        return max((abs(v) for v in self.joint_velocity), default=0.0) > velocity_tol


def format_pose_for_script(pose: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.6f}" for v in pose) + "]"


class RobotBase:
    """Common motion-wait logic. Subclasses implement the hardware calls."""

    description = "robot"
    last_error = ""

    def connect(self) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def get_state(self, max_age_s: float = 1.0) -> Optional[RobotState]:
        raise NotImplementedError

    def move_linear(self, pose: Sequence[float], acceleration: float, velocity: float) -> bool:
        raise NotImplementedError

    def stop_motion(self) -> bool:
        raise NotImplementedError

    def speed_linear(self, xd: Sequence[float], acceleration: float, hold_s: float) -> bool:
        """TCP speed jog (pp_ws): xd = [vx, vy, vz (m/s), wx, wy, wz (rad/s)] in the base frame.
        The robot keeps moving for hold_s seconds unless stop_motion() is called first."""
        raise NotImplementedError

    def pop_alarms(self) -> List[str]:
        return []

    def wait_until_reached(self, target: Sequence[float],
                           position_tol_m: float = 0.001,
                           timeout_s: float = 30.0,
                           cancel: Optional[Callable[[], bool]] = None,
                           velocity_tol: float = 0.01,
                           hold_s: float = 0.2) -> Tuple[bool, str]:
        """Wait until the actual TCP is within tolerance of ``target`` and the
        robot is standing still for ``hold_s`` seconds.

        Returns (True, "") on success, otherwise (False, reason). A reason that
        contains "STOP" means the robot is in emergency/protective stop.
        """
        target_xyz = np.asarray(target[:3], dtype=float)
        start = time.time()
        reached_since = None
        error_m = None
        while True:
            now = time.time()
            if now - start > timeout_s:
                if error_m is None:
                    return False, f"timeout after {timeout_s:.0f} s (no robot state)"
                return False, (f"timeout after {timeout_s:.0f} s "
                               f"(still {error_m * 1000:.1f} mm from target)")
            if cancel is not None and cancel():
                return False, "cancelled by user"
            st = self.get_state()
            if st is None:
                time.sleep(0.05)
                continue
            if st.emergency_stopped:
                return False, "robot is in EMERGENCY STOP"
            if st.protective_stopped:
                return False, "robot is in PROTECTIVE STOP"
            error_m = float(np.linalg.norm(np.asarray(st.tcp_pose[:3]) - target_xyz))
            moving = max((abs(v) for v in st.joint_velocity), default=0.0) > velocity_tol
            if error_m <= position_tol_m and not moving:
                if reached_since is None:
                    reached_since = now
                elif now - reached_since >= hold_s:
                    return True, ""
            else:
                reached_since = None
            time.sleep(0.02)


class RobotInterface(RobotBase):
    """Real robot on the primary port (default 30001)."""

    RECONNECT_AFTER_S = 3.0      # no state for this long -> reconnect
    RECONNECT_INTERVAL_S = 3.0   # minimum time between connection attempts

    def __init__(self, ip: str, port: int = 30001,
                 move_command: str = "movel({pose}, a={a}, v={v})",
                 stop_command: str = "stopl(1.0)"):
        # imported here so --sim needs no driver deps
        from elite_robot_controller.robot.robot_driver import Robot_30001

        self.ip, self.port = ip, int(port)
        self.description = f"{ip}:{self.port}"
        self.move_command = move_command
        self.stop_command = stop_command
        self.last_error = ""
        self._primary = Robot_30001(ip, self.port)
        self._io_lock = threading.Lock()       # one socket, shared by poll thread and commands
        self._state_lock = threading.Lock()
        self._state: Optional[RobotState] = None
        self._alarms = collections.deque(maxlen=50)
        self._connected = False
        self._connected_since = 0.0
        self._last_attempt = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def connect(self) -> bool:
        """Connect once and start the polling thread (which keeps reconnecting)."""
        ok = self._try_connect()
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._poll_loop, name="robot-poll", daemon=True)
            self._thread.start()
        return ok

    def _try_connect(self) -> bool:
        self._last_attempt = time.time()
        with self._io_lock:
            try:
                self._primary.disconnect_30001()
            except Exception:
                pass
            sock = self._primary.connect_30001()
        self._connected = sock is not None
        if self._connected:
            self._connected_since = time.time()
        else:
            self.last_error = f"cannot connect to {self.ip}:{self.port}"
        return self._connected

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            data = None
            if self._connected:
                with self._io_lock:
                    try:
                        data = self._primary.get_data()
                    except Exception as e:
                        self.last_error = f"read error: {e}"
                self._drain_alarms()

            if data is not None:
                st = self._to_state(data)
                if st is not None:
                    with self._state_lock:
                        self._state = st
                    self.last_error = ""
            else:
                now = time.time()
                with self._state_lock:
                    last = self._state.timestamp if self._state else self._connected_since
                stale = (not self._connected) or (now - last > self.RECONNECT_AFTER_S)
                if self._connected and self._state is None and now - last > self.RECONNECT_AFTER_S:
                    self.last_error = ("connected, but no robot state could be decoded "
                                       "(packet format in robot_driver.py vs. controller version?)")
                if stale and now - self._last_attempt > self.RECONNECT_INTERVAL_S:
                    self._try_connect()
            time.sleep(0.01)

    @staticmethod
    def _to_state(d) -> Optional[RobotState]:
        try:
            return RobotState(
                tcp_pose=[float(d.tcp_x), float(d.tcp_y), float(d.tcp_z),
                          float(d.rot_x), float(d.rot_y), float(d.rot_z)],
                tcp_offset=[float(d.offset_px), float(d.offset_py), float(d.offset_pz),
                            float(d.offset_rotx), float(d.offset_roty), float(d.offset_rotz)],
                joint_velocity=[float(v) for v in d.actual_velocity],
                power_on=bool(d.is_robot_power_on),
                emergency_stopped=bool(d.is_emergency_stopped),
                protective_stopped=bool(d.is_robot_protective_stopped),
                timestamp=time.time(),
                joint_positions=[float(v) for v in d.actual_joint],
                program_running=bool(d.is_task_running) if hasattr(d, "is_task_running") else None,
            )
        except (AttributeError, TypeError, ValueError):
            return None

    def _drain_alarms(self) -> None:
        q = self._primary.alarm_queue
        while True:
            try:
                a = q.get_nowait()
            except queue.Empty:
                break
            text = a.msg if a.msg else f"E{a.code} S{a.sub} (level {a.level})"
            with self._state_lock:
                self._alarms.append(text)

    # ------------------------------------------------------------------
    def get_state(self, max_age_s: float = 1.0) -> Optional[RobotState]:
        with self._state_lock:
            st = self._state
        if st is None or time.time() - st.timestamp > max_age_s:
            return None
        return st

    def pop_alarms(self) -> List[str]:
        with self._state_lock:
            items = list(self._alarms)
            self._alarms.clear()
        return items

    def send_script(self, line: str) -> bool:
        with self._io_lock:
            ok = self._primary.send_script(line)
        if not ok:
            self.last_error = f"failed to send script: {line}"
        return ok

    def move_linear(self, pose: Sequence[float], acceleration: float, velocity: float) -> bool:
        cmd = self.move_command.format(pose=format_pose_for_script(pose),
                                       a=f"{float(acceleration):.4f}",
                                       v=f"{float(velocity):.4f}")
        return self.send_script(cmd)

    def stop_motion(self) -> bool:
        return self.send_script(self.stop_command)

    def speed_linear(self, xd: Sequence[float], acceleration: float, hold_s: float) -> bool:
        # Same as the jog of elite_robot_controller: one speedl per button press, the
        # robot stops by itself after hold_s if the stop on release never arrives.
        return self.send_script(f"speedl({format_pose_for_script(xd)}, "
                                f"{float(acceleration):.4f}, {float(hold_s):.3f})")

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._io_lock:
            try:
                self._primary.disconnect_30001()
            except Exception:
                pass
        self._connected = False
