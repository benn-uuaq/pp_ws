"""
sim_robot.py - Simulated robot (motion_tool --sim and the tests).

Taken from simulator.SimRobot of the original calibration tool. The robot
moves in a straight line at constant speed; the simulated camera stays out
until the Helios camera is integrated.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from robot_camera_calibration.robot_interface import RobotBase, RobotState


class SimRobot(RobotBase):
    def __init__(self, start_pose: Sequence[float]):
        self.description = "SIMULATED robot"
        self.last_error = ""
        self._lock = threading.Lock()
        self._from = list(map(float, start_pose))
        self._to = list(self._from)
        self._t0 = 0.0
        self._duration = 0.0

    def connect(self) -> bool:
        return True

    def _current(self, now: float) -> Tuple[List[float], bool]:
        if self._duration <= 0 or now >= self._t0 + self._duration:
            return list(self._to), False
        f = (now - self._t0) / self._duration
        pose = [a + f * (b - a) for a, b in zip(self._from, self._to)]
        return pose, True

    def current_pose(self) -> List[float]:
        with self._lock:
            return self._current(time.time())[0]

    def get_state(self, max_age_s: float = 1.0) -> Optional[RobotState]:
        with self._lock:
            pose, moving = self._current(time.time())
        return RobotState(tcp_pose=pose, tcp_offset=[0.0] * 6,
                          joint_velocity=[0.2 if moving else 0.0] * 6,
                          power_on=True, emergency_stopped=False,
                          protective_stopped=False, timestamp=time.time())

    def move_linear(self, pose: Sequence[float], acceleration: float, velocity: float) -> bool:
        with self._lock:
            now = time.time()
            current, _ = self._current(now)
            dist = float(np.linalg.norm(np.subtract(pose[:3], current[:3])))
            self._from = current
            self._to = list(map(float, pose))
            self._t0 = now
            self._duration = max(dist / max(float(velocity), 1e-3), 0.05)
        return True

    def stop_motion(self) -> bool:
        with self._lock:
            current, _ = self._current(time.time())
            self._from = self._to = current
            self._duration = 0.0
        return True

    def speed_linear(self, xd: Sequence[float], acceleration: float, hold_s: float) -> bool:
        """Moves at constant speed for hold_s (rotation is simplified to increasing the RX/RY/RZ values directly)."""
        with self._lock:
            now = time.time()
            current, _ = self._current(now)
            self._from = current
            self._to = [c + float(v) * float(hold_s) for c, v in zip(current, xd)]
            self._t0 = now
            self._duration = max(float(hold_s), 0.01)
        return True
