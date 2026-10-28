"""
servo_motion.py - all robot motion as servoj (Elite CS scripts sent to port 30001).

Same approach as dusan_v5 (dus5_pass_r.script):
  * a script loop on the robot computes the next target every control cycle (sv_dt = 4 ms) and sends it with servoj.
  * progress advances by the **actually elapsed time** from controller_timestamp().
  * velocity profile: trapezoid whose acceleration / deceleration phases are sine-shaped — acceleration starts and ends at 0,
    so starting and stopping do not jerk (peak acceleration = configured acceleration).
  * after reaching the end it keeps sending for twice the lookahead so the robot catches up, then stopj.
  * Note: the pendant speed ratio does not apply to servoj. The speed is exactly the value given here.

Two kinds of motion
  line  : linear (interpolate_pose + get_inverse_kin) — instead of movel. Pick / place approach and retreat.
  joint : joint interpolation — instead of movej. Going to saved poses (home, pick standby).
The progress variable is normalised 0 -> 1, and speed / acceleration are normalised by the distance (or angle) too.
The slower of position and rotation sets the timing.

Scripts sent to 30001 run in a different context from the pendant task, so the active TCP currently reported by the robot is
set again with set_tcp to match the target pose (relative to the active TCP) with the moving TCP.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from robot_camera_calibration import pose_math as pm

DEFAULT_SERVO = {
    "dt": 0.004,               # [s] control cycle (servoj t)
    "lookahead": 0.1,          # [s] servoj lookahead_time (0.03 ~ 0.2). Longer = smoother, follows slightly later
    "gain": 300,               # servoj gain (100 ~ 2000)
    "lin_acc": 0.3,            # [m/s^2] linear acceleration / deceleration
    "rot_speed_deg_s": 30.0,   # [deg/s] max tool rotation speed during linear moves
    "rot_acc_deg_s2": 60.0,    # [deg/s^2]
    "joint_speed_deg_s": 30.0, # [deg/s] max joint speed (joint that turns the most)
    "joint_acc_deg_s2": 60.0,  # [deg/s^2]
}


def _fmt(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.6f}" for v in values) + "]"


def normalized_profile(dist: float, speed: float, acc: float) -> Tuple[float, float]:
    """Distance / max speed / acceleration -> max speed and acceleration of the progress variable (0->1). Zero distance gives (inf, inf)."""
    if dist <= 1e-9:
        return math.inf, math.inf
    return speed / dist, acc / dist


def smooth_profile(vmax: float, acc: float) -> Tuple[float, float, float]:
    """Normalised distance 1 as a trapezoid with sine acceleration / deceleration: (max speed, acceleration time ta, cruise time tc).

    Speed in the acceleration phase v(t) = vmax (1 - cos(pi t / ta)) / 2 — acceleration 0 -> acc -> 0 (peak acc).
    If the distance is too short to reach the max speed, the max speed is lowered (no cruise phase).
    """
    if not math.isfinite(vmax) or not math.isfinite(acc):
        return math.inf, 0.0, 0.0
    ta = 0.5 * math.pi * vmax / acc
    if vmax * ta >= 1.0:
        vmax = math.sqrt(2.0 * acc / math.pi)
        return vmax, 0.5 * math.pi * vmax / acc, 0.0
    return vmax, ta, (1.0 - vmax * ta) / vmax


def profile_position(t: float, vmax: float, ta: float, tc: float) -> float:
    """Progress variable s(t) (0 -> 1) of smooth_profile. Same formula as the robot script loop."""
    T = 2.0 * ta + tc
    if t <= 0.0:
        return 0.0
    if t < ta:
        return 0.5 * vmax * (t - ta / math.pi * math.sin(math.pi * t / ta))
    if t < ta + tc:
        return 0.5 * vmax * ta + vmax * (t - ta)
    if t < T:
        r = T - t
        return 1.0 - 0.5 * vmax * (r - ta / math.pi * math.sin(math.pi * r / ta))
    return 1.0


def profile_duration(vmax: float, acc: float) -> float:
    """Time [s] to go 0 -> 1 with the normalised profile."""
    v, ta, tc = smooth_profile(vmax, acc)
    return 0.0 if not math.isfinite(v) else 2.0 * ta + tc


def line_profile(p0: Sequence[float], p1: Sequence[float], lin_speed: float, servo: dict,
                 convention: str = "rpy") -> Tuple[float, float, float]:
    """(normalised max speed, normalised acceleration, expected time s) of a linear move."""
    d = float(np.linalg.norm(np.subtract(p1[:3], p0[:3])))
    th = math.radians(pm.rotation_angle_deg(pm.rotation_from_pose(p0, convention),
                                            pm.rotation_from_pose(p1, convention)))
    v1, a1 = normalized_profile(d, lin_speed, float(servo["lin_acc"]))
    v2, a2 = normalized_profile(th, math.radians(float(servo["rot_speed_deg_s"])),
                                math.radians(float(servo["rot_acc_deg_s2"])))
    vmax, acc = min(v1, v2), min(a1, a2)
    return vmax, acc, profile_duration(vmax, acc)


def joint_profile(q0: Sequence[float], q1: Sequence[float], servo: dict) -> Tuple[float, float, float]:
    """Joint move: the joint that turns the most stays within the max speed / acceleration."""
    dq = float(np.max(np.abs(np.subtract(q1, q0))))
    vmax, acc = normalized_profile(dq, math.radians(float(servo["joint_speed_deg_s"])),
                                   math.radians(float(servo["joint_acc_deg_s2"])))
    return vmax, acc, profile_duration(vmax, acc)


def _loop_script(name: str, head: str, target_expr: str, vmax: float, acc: float,
                 tcp_offset: Sequence[float], servo: dict) -> str:
    """Common servoj loop. head sets sv_q and the start / end; target_expr builds the joint values every cycle.

    The progress variable sv_s is a function of the elapsed time sv_t (same formula as profile_position, sine acceleration / deceleration).
    """
    v, ta, tc = smooth_profile(vmax, acc)
    return (
        f"def {name}():\n"
        f"  set_tcp({_fmt(tcp_offset)})\n"
        f"  sv_dt = {float(servo['dt']):.4f}\n"
        f"  sv_look = {float(servo['lookahead']):.3f}\n"
        f"  sv_gain = {int(servo['gain'])}\n"
        f"{head}"
        f"  sv_vmax = {v:.6f}\n"
        f"  sv_ta = {ta:.6f}\n"
        f"  sv_tc = {tc:.6f}\n"
        "  sv_te = 2.0 * sv_ta + sv_tc\n"
        "  sv_pi = 3.14159265\n"
        "  sv_t = 0\n"
        "  sv_s = 0\n"
        "  sv_set = 0\n"
        "  sv_nset = ceil(sv_look * 2.0 / sv_dt)\n"
        "  sv_tp = controller_timestamp()\n"
        "  while (sv_set < sv_nset):\n"
        "    sv_tn = controller_timestamp()\n"
        "    sv_h = sv_tn - sv_tp\n"
        "    sv_tp = sv_tn\n"
        "    if (sv_h <= 0 or sv_h > 0.05):\n"
        "      sv_h = sv_dt\n"
        "    end\n"
        "    if (sv_s < 1.0):\n"
        "      sv_t = sv_t + sv_h\n"
        "      if (sv_t < sv_ta):\n"
        "        sv_s = 0.5 * sv_vmax * (sv_t - sv_ta / sv_pi * sin(sv_pi * sv_t / sv_ta))\n"
        "      elif (sv_t < sv_ta + sv_tc):\n"
        "        sv_s = 0.5 * sv_vmax * sv_ta + sv_vmax * (sv_t - sv_ta)\n"
        "      elif (sv_t < sv_te):\n"
        "        sv_r = sv_te - sv_t\n"
        "        sv_s = 1.0 - 0.5 * sv_vmax * (sv_r - sv_ta / sv_pi * sin(sv_pi * sv_r / sv_ta))\n"
        "      else:\n"
        "        sv_s = 1.0\n"
        "      end\n"
        "    else:\n"
        "      sv_set = sv_set + 1\n"
        "    end\n"
        f"{target_expr}"
        "    servoj(sv_q, t = sv_dt, lookahead_time = sv_look, gain = sv_gain)\n"
        "  end\n"
        "  stopj(2.0)\n"
        "end\n"
    )


def build_line_script(target_pose: Sequence[float], vmax: float, acc: float,
                      tcp_offset: Sequence[float], servo: dict) -> str:
    """Linear servoj from the current TCP to target_pose. If there is no inverse kinematics solution it stops without moving."""
    head = (
        "  sv_p0 = get_actual_tcp_pose()\n"
        f"  sv_p1 = {_fmt(target_pose)}\n"
        "  sv_q = get_actual_joint_positions()\n"
        "  if (get_inverse_kin_has_solution(sv_p1, qnear = sv_q) == False):\n"
        '    textmsg("PP SERVO LINE: no inverse kinematics solution")\n'
        "    halt\n"
        "  end\n"
    )
    target = ("    sv_q = get_inverse_kin(interpolate_pose(sv_p0, sv_p1, sv_s), qnear = sv_q)\n")
    return _loop_script("pp_servo_line", head, target, vmax, acc, tcp_offset, servo)


def build_joint_script(target_joints: Sequence[float], vmax: float, acc: float,
                       tcp_offset: Sequence[float], servo: dict) -> str:
    """Joint-interpolated servoj from the current joints to target_joints (instead of movej)."""
    head = (
        "  sv_q0 = get_actual_joint_positions()\n"
        f"  sv_q1 = {_fmt(target_joints)}\n"
        "  sv_q = sv_q0\n"
    )
    terms = ", ".join(f"sv_q0[{i}] + (sv_q1[{i}] - sv_q0[{i}]) * sv_s" for i in range(6))
    target = f"    sv_q = [{terms}]\n"
    return _loop_script("pp_servo_joint", head, target, vmax, acc, tcp_offset, servo)


def build_digital_out_script(indices: Sequence[int], value: bool) -> str:
    """Switch several standard digital outputs one after another in one program (effectively simultaneously)."""
    b = "True" if value else "False"
    return ("def pp_digital_out():\n"
            + "".join(f"  set_standard_digital_out({int(i)}, {b})\n" for i in indices)
            + "end\n")


class ServoMover:
    """Send robot motion as servoj scripts and wait for arrival. The simulated robot imitates the same duration."""

    def __init__(self, robot, servo: Optional[dict] = None, convention: str = "rpy",
                 tolerance_mm: float = 1.0, log: Callable[[str], None] = lambda s: None):
        self.robot = robot
        self.servo = dict(DEFAULT_SERVO, **(servo or {}))
        self.convention = convention
        self.tol_m = float(tolerance_mm) / 1000.0
        self.log = log
        self.real = hasattr(robot, "send_script")

    def _state(self):
        st = self.robot.get_state()
        if st is None:
            raise RuntimeError("No robot state received")
        if st.emergency_stopped or st.protective_stopped:
            raise RuntimeError("The robot is in emergency / protective stop")
        return st

    def _wait(self, check_pose, duration: float, cancel) -> Tuple[bool, str]:
        timeout = duration * 1.5 + 2.0 * float(self.servo["lookahead"]) + 3.0
        return self.robot.wait_until_reached(check_pose, self.tol_m, timeout, cancel=cancel)

    def line(self, target_pose: Sequence[float], lin_speed: float, cancel=None) -> Tuple[bool, str]:
        st = self._state()
        vmax, acc, T = line_profile(st.tcp_pose, target_pose, lin_speed, self.servo, self.convention)
        if T <= 0.0:
            return True, ""
        self.log(f"  servoj linear {T:.1f} s")
        if self.real:
            ok = self.robot.send_script(build_line_script(target_pose, vmax, acc, st.tcp_offset, self.servo))
        else:
            d = float(np.linalg.norm(np.subtract(target_pose[:3], st.tcp_pose[:3])))
            ok = self.robot.move_linear(target_pose, 1.0, max(d / max(T, 1e-3), 1e-4))
        if not ok:
            return False, f"Failed to send the script: {self.robot.last_error}"
        return self._wait(target_pose, T, cancel)

    def joint(self, target_joints: Sequence[float], check_pose: Sequence[float], cancel=None) -> Tuple[bool, str]:
        st = self._state()
        q0 = st.joint_positions
        if self.real and q0 is None:
            return False, "No robot joint values received"
        vmax, acc, T = joint_profile(q0 if q0 is not None else target_joints, target_joints, self.servo)
        if not self.real:                            # the simulated robot has no joints, so imitate with the TCP
            d = float(np.linalg.norm(np.subtract(check_pose[:3], st.tcp_pose[:3])))
            T = max(T, d / 0.2) if d > 0 else T
            if T <= 0.0:
                return True, ""
            self.robot.move_linear(check_pose, 1.0, max(d / max(T, 1e-3), 1e-4))
            return self._wait(check_pose, T, cancel)
        if T <= 0.0:
            return True, ""
        self.log(f"  servoj joint {T:.1f} s")
        if not self.robot.send_script(build_joint_script(target_joints, vmax, acc, st.tcp_offset, self.servo)):
            return False, f"Failed to send the script: {self.robot.last_error}"
        return self._wait(check_pose, T, cancel)

    def digital_out(self, indices: Sequence[int], value: bool) -> bool:
        if not self.real:
            return True
        return self.robot.send_script(build_digital_out_script(indices, value))
