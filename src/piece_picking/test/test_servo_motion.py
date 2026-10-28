"""servo_motion: servoj script generation, velocity profile, sending (without a robot)."""

import math
import re
import time

import numpy as np
import pytest

from piece_picking import servo_motion as sm
from robot_camera_calibration.robot_interface import RobotState
from robot_camera_calibration.sim_robot import SimRobot

SERVO = dict(sm.DEFAULT_SERVO)


def _blocks_balanced(script: str) -> bool:
    """Elite script: one end per def / if / while (else does not add an end)."""
    opens = ends = 0
    for line in script.splitlines():
        s = line.strip()
        if re.match(r"^(def |if |if\(|while |while\()", s):
            opens += 1
        elif s == "end":
            ends += 1
    return opens == ends and opens > 0


def test_smooth_profile_duration_matches_physics():
    # 100 mm at 50 mm/s, peak 0.3 m/s^2: sine acceleration time ta = pi/2 v/a, total = d/v + ta
    vmax, acc, T = sm.line_profile([0, 0, 0.5, math.pi, 0, 0], [0.1, 0, 0.5, math.pi, 0, 0], 0.05, SERVO)
    assert T == pytest.approx(2.0 + math.pi / 2 * 0.05 / 0.3, rel=1e-6)
    # very short: max speed not reached: sqrt(2 pi d / a)
    _, _, T2 = sm.line_profile([0, 0, 0.5, math.pi, 0, 0], [0.002, 0, 0.5, math.pi, 0, 0], 0.05, SERVO)
    assert T2 == pytest.approx(math.sqrt(2 * math.pi * 0.002 / 0.3), rel=1e-6)


def test_rotation_can_dominate_duration():
    # no translation, 30 deg rotation: 30 deg/s, peak 60 deg/s^2 -> 1 + pi/4 s
    _, _, T = sm.line_profile([0, 0, 0.5, math.pi, 0, 0], [0, 0, 0.5, math.pi, 0, math.radians(30)], 0.05, SERVO)
    assert T == pytest.approx(1.0 + math.pi / 4, rel=1e-3)


@pytest.mark.parametrize("vmax, acc", [(0.5, 3.0), (25.0, 150.0), (1.0, 2.0)])
def test_profile_starts_and_stops_without_jerk(vmax, acc):
    """s increases monotonically 0 -> 1, speed and acceleration start and end at 0, peak acceleration = acc."""
    v, ta, tc = sm.smooth_profile(vmax, acc)
    T = 2 * ta + tc
    h = T / 4000
    ts = np.arange(0, T + 5 * h, h)
    s = np.array([sm.profile_position(t, v, ta, tc) for t in ts])
    assert s[0] == 0.0 and s[-1] == 1.0 and np.all(np.diff(s) >= -1e-12)
    vel = np.gradient(s, h)
    accel = np.gradient(vel, h)
    assert max(vel) <= vmax * 1.001
    assert np.max(np.abs(accel[5:-5])) <= acc * 1.01
    for i in (2, len(ts) - 7):                                       # acceleration ~ 0 near the start / end
        assert abs(accel[i]) < 0.05 * acc


def test_line_script_structure():
    s = sm.build_line_script([0.1, 0.2, 0.3, 3.1, 0.0, 0.2], 0.5, 3.0, [0, 0, 0.15, 0, 0, 0], SERVO)
    assert s.startswith("def pp_servo_line():") and s.rstrip().endswith("end")
    assert _blocks_balanced(s)
    assert "set_tcp([0.000000, 0.000000, 0.150000" in s                 # matched to the current active TCP
    assert "sv_p1 = [0.100000, 0.200000, 0.300000, 3.100000" in s
    assert "get_inverse_kin_has_solution(sv_p1" in s and "halt" in s      # no motion without an IK solution
    assert "interpolate_pose(sv_p0, sv_p1, sv_s)" in s
    assert "servoj(sv_q, t = sv_dt, lookahead_time = sv_look, gain = sv_gain)" in s
    assert "controller_timestamp()" in s and "stopj(2.0)" in s
    assert re.search(r"\bp\b\s*=|\bj\b\s*=", s) is None                 # the names p and j clash with literals


def test_joint_script_structure():
    s = sm.build_joint_script([0.1, -1.5, 1.2, -1.2, -1.57, 0.3], 0.4, 2.0, [0] * 6, SERVO)
    assert s.startswith("def pp_servo_joint():") and _blocks_balanced(s)
    assert "sv_q1 = [0.100000, -1.500000, 1.200000" in s
    assert s.count("sv_q0[") == 12 and "servoj(sv_q" in s


class _FakeRealRobot:
    """Imitates a real robot with send_script: only collects the scripts sent."""

    def __init__(self):
        self.scripts, self.last_error = [], ""
        self.pose = [0.0, 0.4, 0.5, math.pi, 0.0, 0.0]

    def get_state(self, max_age_s=1.0):
        return RobotState(self.pose, [0, 0, 0.1, 0, 0, 0], [0.0] * 6, True, False, False, time.time(),
                          [0.0, -1.5, 1.2, -1.2, -1.57, 0.0])

    def send_script(self, line):
        self.scripts.append(line)
        return True

    def wait_until_reached(self, target, tol, timeout, cancel=None):
        self.waited = (list(target), timeout)
        return True, ""


def test_mover_sends_scripts_with_active_tcp_and_timeout():
    robot = _FakeRealRobot()
    mover = sm.ServoMover(robot, SERVO)
    target = [0.1, 0.4, 0.5, math.pi, 0.0, 0.0]
    assert mover.line(target, 0.05) == (True, "")
    assert "pp_servo_line" in robot.scripts[-1] and "set_tcp([0.000000, 0.000000, 0.100000" in robot.scripts[-1]
    assert robot.waited[0] == target and robot.waited[1] > 2.0          # expected time + margin
    assert mover.joint([0.2, -1.4, 1.1, -1.2, -1.57, 0.0], target) == (True, "")
    assert "pp_servo_joint" in robot.scripts[-1]
    assert mover.digital_out([4, 5, 6, 7], False)                       # four suction outputs in one script
    s = robot.scripts[-1]
    assert s.startswith("def pp_digital_out():") and _blocks_balanced(s)
    assert re.findall(r"set_standard_digital_out\((\d+), (\w+)\)", s) == [(str(i), "False") for i in (4, 5, 6, 7)]


def test_mover_on_sim_robot_reaches_target():
    robot = SimRobot([0.0, 0.4, 0.5, math.pi, 0.0, 0.0])
    mover = sm.ServoMover(robot, SERVO)
    ok, why = mover.line([0.03, 0.4, 0.48, math.pi, 0.0, 0.0], 0.2)
    assert ok, why
    assert np.allclose(robot.current_pose()[:3], [0.03, 0.4, 0.48], atol=1e-4)
