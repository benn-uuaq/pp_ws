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
    assert "sv_t = sv_t + sv_dt" in s and "controller_timestamp" not in s   # fixed time base: servoj blocks for sv_dt
    assert "get_actual_joint_speeds()" in s and "stopj(1.000)" in s        # settle check, then a gentle stopj
    assert re.search(r"\bp\b\s*=|\bj\b\s*=", s) is None                 # the names p and j clash with literals


def test_joint_script_structure():
    s = sm.build_joint_script([0.1, -1.5, 1.2, -1.2, -1.57, 0.3], 0.4, 2.0, [0] * 6, SERVO)
    assert s.startswith("def pp_servo_joint():") and _blocks_balanced(s)
    assert "sv_q1 = [0.100000, -1.500000, 1.200000" in s
    assert s.count("sv_q0[") == 12 and "servoj(sv_q" in s


class _ServoSim:
    """Runs a generated Elite script in Python against a simple servo model.

    Joint space = pose space here (identity kinematics). servoj: moving average of the targets over the lookahead
    time (as the manual describes), then the actual joints follow that with a first-order lag. Records every call.
    """

    def __init__(self, q0, servo, lag_s=0.05, noise=0.0):
        self.q = np.array(q0, dtype=float)
        self.qd = np.zeros(6)
        self.dt, self.look, self.lag, self.noise = servo["dt"], servo["lookahead"], lag_s, noise
        self.buf, self.sent, self.stops, self.at_stop = [], [], [], None
        self.rng = np.random.default_rng(0)

    def servoj(self, q, t=0.004, lookahead_time=0.1, gain=300):
        self.sent.append(np.array(q, dtype=float))
        self.buf = (self.buf + [np.array(q, dtype=float)])[-max(1, int(round(lookahead_time / t))):]
        target = np.mean(self.buf, axis=0)
        new = self.q + (target - self.q) * min(1.0, t / self.lag)
        self.qd = (new - self.q) / t + self.noise * self.rng.standard_normal(6)
        self.q = new

    def stopj(self, a):
        self.stops.append(a)
        self.at_stop = (self.q.copy(), self.qd.copy())

    def run(self, script):
        lines = []
        for line in script.splitlines():                       # Elite script -> Python: drop "end", map "halt"
            if line.strip() == "end":
                continue
            lines.append(line.replace("halt", "raise RuntimeError('halt')") if line.strip() == "halt" else line)
        ns = {
            "ceil": math.ceil, "sin": math.sin, "textmsg": print, "set_tcp": lambda p: None,
            "servoj": self.servoj, "stopj": self.stopj,
            "get_actual_joint_positions": lambda: list(self.q), "get_actual_joint_speeds": lambda: list(self.qd),
            "get_actual_tcp_pose": lambda: list(self.q),
            "get_inverse_kin_has_solution": lambda p, qnear=None: True,
            "get_inverse_kin": lambda p, qnear=None: list(p),
            "interpolate_pose": lambda a, b, s: list(np.add(a, np.subtract(b, a) * s)),
        }
        exec("\n".join(lines), ns)                               # noqa: S102 - our own generated script
        name = re.match(r"def (\w+)\(", script).group(1)
        ns[name]()


@pytest.mark.parametrize("kind", ["line", "joint"])
def test_script_runs_smoothly_and_ends_only_after_settling(kind):
    servo = dict(SERVO)
    q0, q1 = [0.0, -1.5, 1.2, -1.2, -1.57, 0.0], [0.3, -1.3, 1.0, -1.0, -1.4, 0.4]
    vmax, acc, T = sm.joint_profile(q0, q1, servo)
    sim = _ServoSim(q0, servo)
    script = (sm.build_joint_script(q1, vmax, acc, [0] * 6, servo) if kind == "joint"
              else sm.build_line_script(q1, vmax, acc, [0] * 6, servo))
    sim.run(script)
    sent = np.array(sim.sent)
    n_profile = int(math.ceil(T / servo["dt"]))
    assert np.allclose(sent[n_profile - 1], q1) and np.allclose(sent[-1], q1)
    assert np.allclose(sent[n_profile - 2], sent[n_profile - 1], atol=1e-4)   # arrives with ~zero speed
    # evenly timed, smooth targets: per-cycle steps never jump (second difference bounded by the peak acceleration)
    span = np.max(np.abs(np.subtract(q1, q0)))
    d2 = np.abs(np.diff(sent[:n_profile], n=2, axis=0)).max()
    assert d2 <= acc * span * servo["dt"] ** 2 * 1.05
    # it held the target until the robot really settled, then stopped once, gently
    hold = len(sent) - n_profile
    assert math.ceil(2 * servo["lookahead"] / servo["dt"]) <= hold < servo["settle_max_s"] / servo["dt"]
    q_end, qd_end = sim.at_stop
    assert sim.stops == [servo["stop_acc"]]
    assert np.max(np.abs(q_end - q1)) <= servo["settle_pos_tol_rad"]
    assert np.max(np.abs(qd_end)) <= servo["settle_vel_tol_rad_s"]


def test_script_gives_up_settling_after_settle_max():
    servo = dict(SERVO)
    q0, q1 = [0.0] * 6, [0.05, 0, 0, 0, 0, 0]
    vmax, acc, T = sm.joint_profile(q0, q1, servo)
    sim = _ServoSim(q0, servo, noise=0.05)                       # speed reading never quiet
    sim.run(sm.build_joint_script(q1, vmax, acc, [0] * 6, servo))
    hold = len(sim.sent) - int(math.ceil(T / servo["dt"]))
    assert hold == int(math.ceil(servo["settle_max_s"] / servo["dt"])) and sim.stops == [servo["stop_acc"]]


class _FakeRealRobot:
    """Imitates a real robot with send_script: collects the scripts sent.

    running_for: if set, the robot reports "program running" for that long after each script (None = flag unknown).
    """

    def __init__(self, running_for=None):
        self.scripts, self.last_error, self.events = [], "", []
        self.pose = [0.0, 0.4, 0.5, math.pi, 0.0, 0.0]
        self.running_for, self.sent_at = running_for, 0.0

    def get_state(self, max_age_s=1.0):
        running = None if self.running_for is None else (time.time() - self.sent_at < self.running_for)
        return RobotState(self.pose, [0, 0, 0.1, 0, 0, 0], [0.0] * 6, True, False, False, time.time(),
                          [0.0, -1.5, 1.2, -1.2, -1.57, 0.0], running)

    def send_script(self, line):
        if self.running_for is not None and time.time() - self.sent_at < self.running_for:
            self.events.append("ABORTED a running program")
        self.scripts.append(line)
        self.sent_at = time.time()
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


def test_next_script_waits_until_the_robot_program_has_ended():
    """The robot keeps its program running (holding / settling) 0.6 s after each send: the next script must wait."""
    robot = _FakeRealRobot(running_for=0.6)
    mover = sm.ServoMover(robot, SERVO)
    target = [0.1, 0.4, 0.5, math.pi, 0.0, 0.0]
    t0 = time.time()
    assert mover.line(target, 0.05) == (True, "")
    assert time.time() - t0 >= 0.6                                   # waited for the program end ...
    assert mover.digital_out([4, 5, 6, 7], False)                     # ... so the next script aborts nothing
    assert mover.line([0.1, 0.45, 0.5, math.pi, 0.0, 0.0], 0.05) == (True, "")
    assert robot.events == [] and len(robot.scripts) == 3


def test_flag_not_reported_falls_back_to_the_guard_wait():
    robot = _FakeRealRobot(running_for=None)                          # controller does not report the flag
    mover = sm.ServoMover(robot, dict(SERVO, end_guard_s=0.2))
    t0 = time.time()
    assert mover.line([0.1, 0.4, 0.5, math.pi, 0.0, 0.0], 0.05) == (True, "")
    assert time.time() - t0 >= 0.2 and mover._flag_works is False


def test_mover_on_sim_robot_reaches_target():
    robot = SimRobot([0.0, 0.4, 0.5, math.pi, 0.0, 0.0])
    mover = sm.ServoMover(robot, SERVO)
    ok, why = mover.line([0.03, 0.4, 0.48, math.pi, 0.0, 0.0], 0.2)
    assert ok, why
    assert np.allclose(robot.current_pose()[:3], [0.03, 0.4, 0.48], atol=1e-4)
