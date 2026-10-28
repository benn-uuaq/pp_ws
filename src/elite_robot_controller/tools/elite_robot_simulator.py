#!/usr/bin/env python3
r"""Fake robot for testing elite_robot_controller without an Elite CS612.

Imitates the three channels 29999 (Dashboard), 30001 (Primary) and Modbus TCP.
Register addresses are read directly from `config/modbus_registers.json`, so the node and
the simulator never use different addresses.

## Running

```bash
python3 tools/elite_robot_simulator.py
```

Listens on 127.0.0.1:29999 / 30001 / 5502 by default. 502 needs privileges on
Linux, so the default is 5502. Start the node like this.

```bash
ros2 run elite_robot_controller robot_control_node --ros-args \\
    -p robot_ip:=127.0.0.1 -p modbus_port:=5502
```

## What it imitates

- **Connection**: accepts connections on all three ports. Connecting / disconnecting many times is fine.
- **Dashboard commands**: answers robotMode/status/robotControl -on|off/brakeRelease/
  play/pause/stop with short text. stop also sets the jog speed to 0.
- **Jog**: reads `speedj(...)`/`speedl(...)` scripts arriving on 30001 and
  keeps updating the joint (73~78) and TCP (384~389) registers at that speed.
  Stopped by a `stop` command or by disconnecting. If the target speed is not reached,
  the old value is not kept in use.
- **Home move**: on a `movej(...)` script the joint registers move to the
  target after a short delay.
- **Write registers**: values the UI writes to 306 (work speed), 307 (speed ratio), 308 (pose_src),
  310~321 (reference poses) are stored and returned as is.
- **Task progress**: on `play` the registers are filled in the same order as the current task
  — wait for vehicle lock (309) → 3-point measurement (300~305 / 330~355) → wait at the origin
  (267 permission) → wetting → zigzag scan (segment signal 277) → done. The home flag (276) and
  task state (500) move along.
- **Digital IO**: keeps the bits written to register 2 (standard outputs).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import socket
import struct
import sys
import threading
import time

# fix the path so the package is found even when run directly from the source tree.
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from elite_robot_controller import register_map  # noqa: E402


def to_unsigned16(value: int) -> int:
    return value & 0xFFFF


def to_signed16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value > 0x7FFF else value


class RegisterBank:
    """Keeps register values and protects access from several threads with a lock."""

    def __init__(self, registers: register_map.RegisterMap):
        self.registers = registers
        self._lock = threading.Lock()
        self._values: dict[int, int] = {}
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        defaults = {
            "robot_mode": [7],                          # RUNNING
            "control_method": [2],                       # remote control
            "operation_mode": [0],                        # automatic
            "joint_position": [207, -1466, -1875, -1371, 1570, 207],
            "tcp_absolute": [6368, -473, 5810, 3141, 0, -1570],
            "tcp_zero_relative": [0, 0, 1000, 0, 0, 0],
        }
        for name, values in defaults.items():
            entry = self.registers.read_entry(name)
            if not entry.available:
                continue
            for offset, value in enumerate(values):
                self._values[entry.address + offset] = value
        for name in self.registers.write.keys():
            entry = self.registers.write_entry(name)
            for offset in range(entry.count):
                self._values.setdefault(entry.address + offset, 0)

    def read(self, address: int, count: int) -> list[int]:
        with self._lock:
            return [self._values.get(address + i, 0) for i in range(count)]

    def write(self, address: int, value: int) -> None:
        with self._lock:
            self._values[address] = value

    def write_many(self, address: int, values: list[int]) -> None:
        with self._lock:
            for offset, value in enumerate(values):
                self._values[address + offset] = value

    def entry_by_address(self, target: str) -> register_map.RegisterEntry:
        return self.registers.read_entry(target)


class ModbusServer(threading.Thread):
    """Minimal Modbus TCP server supporting only FC3 (read) / FC6·FC16 (write)."""

    def __init__(self, bank: RegisterBank, host: str, port: int):
        super().__init__(daemon=True)
        self.bank = bank
        self.host, self.port = host, port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(4)

    def run(self) -> None:
        print(f"[Modbus] listening on {self.host}:{self.port}")
        while True:
            conn, addr = self._sock.accept()
            threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()

    def _handle(self, conn: socket.socket, addr) -> None:
        with conn:
            while True:
                header = self._recv_exact(conn, 7)
                if header is None:
                    return
                tx_id, proto_id, length, unit_id = struct.unpack(">HHHB", header)
                pdu = self._recv_exact(conn, length - 1)
                if pdu is None:
                    return
                response_pdu = self._handle_pdu(pdu)
                resp_header = struct.pack(">HHHB", tx_id, proto_id, len(response_pdu) + 1, unit_id)
                conn.sendall(resp_header + response_pdu)

    @staticmethod
    def _recv_exact(conn: socket.socket, size: int) -> bytes | None:
        buf = b""
        while len(buf) < size:
            chunk = conn.recv(size - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _handle_pdu(self, pdu: bytes) -> bytes:
        func = pdu[0]
        if func == 0x03:  # Read Holding Registers
            address, count = struct.unpack(">HH", pdu[1:5])
            values = [to_unsigned16(v) for v in self.bank.read(address, count)]
            payload = struct.pack(">B", count * 2) + b"".join(struct.pack(">H", v) for v in values)
            return struct.pack(">B", 0x03) + payload
        if func == 0x06:  # Write Single Register
            address, value = struct.unpack(">HH", pdu[1:5])
            self.bank.write(address, to_signed16(value))
            return pdu[:5]
        if func == 0x10:  # Write Multiple Registers
            address, count, byte_count = struct.unpack(">HHB", pdu[1:6])
            raw = pdu[6:6 + byte_count]
            values = [to_signed16(v) for v, in struct.iter_unpack(">H", raw)]
            self.bank.write_many(address, values)
            return struct.pack(">BHH", 0x10, address, count)
        # unsupported function code: exception response (0x01 illegal function)
        return struct.pack(">BB", func | 0x80, 0x01)


class ScanTaskSim(threading.Thread):
    """On `play`, imitates one cell of the current robot task (dusan_v4/v5).

    Fills the same registers in the same order as the real task. If this does not
    match, the operating UI cannot get past the first cell.

      290 progress   : 0 idle · 14 waiting for vehicle lock · 2 3-point measurement · 3 back to origin ·
                       7 waiting at origin (for scan permission) · 8 wetting · 6 scan · 5 done
      277 scan segment: 0 none · 1 forward · 2 row change (up) · 3 backward  (TPAC DO signal)
      276 home       : 1 at home, 0 when away
      267 scan permit: waiting at the origin ends when RCS writes 1 (cleared to 0 when passing)
      309 vehicle lock: must be 1 to move. Otherwise it waits with 290 = 14
      500 task       : 1 running · 3 stopped
      300~305 / 330~355 : 3-point measurement results and contact poses (read by the calibration)

    The zigzag path is not actually drawn; only the steps, row count and segment signals are imitated.
    The contact points are placed on a circle offset by `wall_error_mm` from the entered radius (260 + 261)
    — to check that the calibration catches the difference.
    """

    # registers 290~299. Same meaning as the robot task's dus_*.script.
    STATE, ROW_IDX, ROWS, ALIVE = 290, 291, 292, 293
    ZERO_OK, FINISHED, PITCH, PROGRESS = 294, 295, 296, 298
    PROBE_ERROR = 299
    # work area (256~264). The task reads it to compute the zigzag rows and the arc.
    APP_WIDTH, APP_HEIGHT, SCAN_H, OVERLAP = 256, 257, 258, 259
    RADIUS, THICKNESS, PARAM_SRC = 260, 261, 266
    # signals exchanged.
    SCAN_GO, SEGMENT, HOME_FLAG, VEHICLE_READY, TASK_STATE = 267, 277, 276, 309, 500
    ARC_POS, ARC_TOTAL = 286, 287
    # 3-point measurement results.
    PROBE_CX, PROBE_LX, PROBE_RX, PRESS_OFF, N_PROBE, N_HIT = 300, 301, 302, 303, 304, 305
    POSE_CENTER, POSE_RIGHT, POSE_EOAT, POSE_ZERO = 330, 340, 347, 350

    STEP_SECONDS = 0.35
    #: limit [s] for waiting for the vehicle lock (309). Beyond it the task stops with 299 = 8.
    VEHICLE_WAIT_S = 15.0
    #: assume the real wall is off from the input by this much [mm].
    WALL_ERROR_MM = 0.3
    #: upper limit of zigzag rows. The goal is to check the traversal, so not all rows need to be drawn.
    MAX_ROWS = 6

    def __init__(self, bank: RegisterBank, wall_error_mm: float = WALL_ERROR_MM,
                 max_rows: int = MAX_ROWS, motion: "MotionSim | None" = None):
        super().__init__(daemon=True)
        self.bank = bank
        self.motion = motion
        self.wall_error_mm = float(wall_error_mm)
        self.max_rows = max(1, int(max_rows))
        self._start_requested = threading.Event()
        self._abort = threading.Event()
        # stopped with 29999 pause. On play it continues from there instead of restarting.
        self._paused = threading.Event()
        self._alive_count = 0
        self._set(self.HOME_FLAG, 1)
        self._set(self.TASK_STATE, 3)

    #: marking task state (290): 10 moving · 11 waiting in place · 13 going home · 12 done.
    MARK_GO = 278
    #: task path loaded with `task -p`. If the name contains mark, it is the marking task.
    task_path = ""

    def load(self, path: str) -> None:
        """29999 `task -p` — decides which task the next play runs."""
        self.task_path = path

    def play(self) -> None:
        """If paused, continue from there; otherwise run the program from the start."""
        if self._paused.is_set():
            self._paused.clear()
            self._set(self.TASK_STATE, 1)
            print("[Task] resume — continuing where it stopped")
            return
        self._abort.set()          # if running, stop it first
        self._start_requested.set()

    def pause(self) -> bool:
        """Stop the running task where it is (500 = 2). Refused if it is not running."""
        if self._get(self.TASK_STATE) != 1:
            return False
        self._paused.set()
        self._set(self.TASK_STATE, 2)
        print("[Task] paused")
        return True

    def stop(self) -> None:
        self._paused.clear()
        self._abort.set()
        self._start_requested.clear()
        self._set(self.STATE, 0)
        self._set(self.FINISHED, 0)
        self._set(self.SEGMENT, 0)
        self._set(self.TASK_STATE, 3)
        # leave the home flag as is — stopping the task does not move the arm home.

    def _set(self, address: int, value: int) -> None:
        self.bank.write(address, int(value))
        if address == self.HOME_FLAG and getattr(self, "motion", None) is not None:
            # move the home flag and the joints together (the node also decides from the joints).
            self.motion.set_pose(bool(value))

    def _get(self, address: int) -> int:
        return self.bank.read(address, 1)[0]

    def _hold_while_paused(self) -> bool:
        """Stay in place while paused (time does not advance). False if aborted."""
        while self._paused.is_set():
            if self._abort.is_set():
                return False
            time.sleep(0.05)
        return not self._abort.is_set()

    def _tick(self, seconds: float) -> bool:
        """Wait for the given time. False on an abort request. Pauses while paused."""
        end = time.time() + seconds
        while time.time() < end:
            if self._paused.is_set():
                left = end - time.time()
                if not self._hold_while_paused():
                    return False
                end = time.time() + left
            if self._abort.is_set():
                return False
            self._alive_count = (self._alive_count + 1) % 30000
            self._set(self.ALIVE, self._alive_count)
            time.sleep(0.05)
        return True

    def _wait_for(self, address: int, value: int, limit_s: float) -> bool:
        """Wait until a register has the wanted value. False on abort / timeout."""
        end = time.time() + limit_s
        while time.time() < end:
            if self._paused.is_set() and not self._hold_while_paused():
                return False
            if self._abort.is_set():
                return False
            if self._get(address) == value:
                return True
            self._alive_count = (self._alive_count + 1) % 30000
            self._set(self.ALIVE, self._alive_count)
            time.sleep(0.05)
        return False

    def _planned_rows(self) -> int:
        """Count the zigzag rows from the work area. Same rule as the robot task."""
        height = self.bank.read(self.APP_HEIGHT, 1)[0]
        scan_h = (self.bank.read(self.SCAN_H, 1)[0] or 1500) / 10.0
        overlap = self.bank.read(self.OVERLAP, 1)[0]
        pitch = max(scan_h - overlap, 1)
        rows = 1
        while (rows - 1) * pitch + scan_h < height:
            rows += 1
            if rows > 200:      # never loop forever even with odd values
                break
        return min(rows, self.max_rows)

    def run(self) -> None:
        while True:
            self._start_requested.wait()
            self._start_requested.clear()
            self._abort.clear()
            if "mark" in self.task_path:
                self._run_mark()
            else:
                self._run_once()

    def _run_mark(self) -> None:
        """One marking task point: 3-point measurement → to the spot (10) → wait in contact (11, 278) → home (13 → 12).

        The marker belongs to ERUT, so the robot **waits at the spot until 278 = 1** —
        same as the real dus_mark_point.
        """
        self._set(self.TASK_STATE, 1)
        self._set(self.PROBE_ERROR, 0)
        self._set(self.HOME_FLAG, 0)
        self._set(self.STATE, 2)
        if not self._tick(self.STEP_SECONDS):
            return
        self._measure_wall()
        self._set(self.STATE, 10)
        if not self._tick(self.STEP_SECONDS):
            return
        self._set(self.MARK_GO, 0)
        self._set(self.STATE, 11)
        if not self._wait_for(self.MARK_GO, 1, 600.0):
            return
        self._set(self.MARK_GO, 0)
        self._set(self.STATE, 13)
        if not self._tick(self.STEP_SECONDS):
            return
        self._set(self.STATE, 12)
        self._set(self.HOME_FLAG, 1)
        self._set(self.TASK_STATE, 3)
        print("[Task] one marking point done")

    # ---- steps ----------------------------------------------------------
    def _fail(self, code: int, why: str) -> None:
        self._set(self.PROBE_ERROR, code)
        self._set(self.STATE, 9)
        self._set(self.SEGMENT, 0)
        print(f"[Task] stopped — {why} (299 = {code})")

    def _wait_for_vehicle(self) -> bool:
        """Wait for the vehicle lock confirmation (309). Same interlock as the real task."""
        if self._get(self.PARAM_SRC) != 1 or self._get(self.VEHICLE_READY) == 1:
            return True
        self._set(self.STATE, 14)
        if self._wait_for(self.VEHICLE_READY, 1, self.VEHICLE_WAIT_S):
            return True
        if self._abort.is_set():
            return False
        self._fail(8, "no vehicle lock confirmation (309)")
        return False

    def _measure_wall(self) -> None:
        """Leave the 3-point measurement results in the registers (300~305 / 330~355).

        The three points lie on a circle of input radius + thickness + wall_error_mm.
        Units are 0.1 mm, same as the real task.
        """
        radius_mm = (self._get(self.RADIUS) + self._get(self.THICKNESS)) / 10.0
        radius_mm = (radius_mm or 845.0) + self.wall_error_mm
        arc_mm = self._get(self.APP_WIDTH) or 700
        half = min(arc_mm / (2 * radius_mm), 0.6) if radius_mm else 0.3
        for base, angle in ((self.POSE_CENTER, 0.0), (self.POSE_RIGHT, half),
                            (self.POSE_ZERO, -half)):
            x = radius_mm * math.sin(angle)
            y = radius_mm * math.cos(angle)
            self.bank.write_many(base, [round(x * 10), round(y * 10), 0, 0, 0, 0])
        # probe centre offset (347~349) is 0 — in the simulator the flange is the probe.
        self.bank.write_many(self.POSE_EOAT, [0, 0, 0])
        self._set(self.PROBE_CX, round(0.0))
        self._set(self.PROBE_LX, round(-radius_mm * math.sin(half) * 10))
        self._set(self.PROBE_RX, round(radius_mm * math.sin(half) * 10))
        self._set(self.PRESS_OFF, 50)       # pushed 5 mm further and found the wall
        self._set(self.N_PROBE, 3)
        self._set(self.N_HIT, 3)

    def _scan_rows(self, rows: int) -> bool:
        """Zigzag scan. Emits the segment signal (277) per row in the real order."""
        arc_mm = self._get(self.APP_WIDTH) or 700
        self._set(self.STATE, 6)
        for row in range(rows):
            self._set(self.ROW_IDX, row)
            forward = row % 2 == 0
            self._set(self.SEGMENT, 1 if forward else 3)
            if not self._tick(self.STEP_SECONDS):
                return False
            self._set(self.ARC_POS, round(arc_mm * 10))
            self._set(self.ARC_TOTAL, round(arc_mm * (row + 1)))
            self._set(self.SEGMENT, 0)          # end of row — scan frozen
            self._set(self.PROGRESS, int((row + 1) / rows * 100))
            if row + 1 < rows:
                self._set(self.SEGMENT, 2)      # row change (up)
                if not self._tick(self.STEP_SECONDS / 2):
                    return False
        self._set(self.SEGMENT, 0)
        return True

    def _run_once(self) -> None:
        rows = self._planned_rows()
        pitch = max(
            self.bank.read(self.SCAN_H, 1)[0] // 10 - self.bank.read(self.OVERLAP, 1)[0], 1
        )
        self._set(self.TASK_STATE, 1)
        self._set(self.PROBE_ERROR, 0)
        self._set(self.FINISHED, 0)
        self._set(self.ROWS, rows)
        self._set(self.PITCH, pitch * 10)       # 0.1 mm (same as dus5_init)
        self._set(self.PROGRESS, 0)
        self._set(self.ZERO_OK, 0)
        self._set(self.SEGMENT, 0)
        self._set(self.STATE, 0)

        if not self._wait_for_vehicle():
            self._set(self.TASK_STATE, 3)
            return
        self._set(self.HOME_FLAG, 0)            # leaving home

        # 3-point measurement → back to origin (290 = 2, 3)
        self._set(self.STATE, 2)
        if not self._tick(self.STEP_SECONDS):
            return
        self._measure_wall()
        self._set(self.STATE, 3)
        if not self._tick(self.STEP_SECONDS):
            return
        self._set(self.ZERO_OK, 1)

        # wait at the origin (290 = 7) until RCS gives permission (267).
        self._set(self.STATE, 7)
        if not self._wait_for(self.SCAN_GO, 1, 600.0):
            return
        self._set(self.SCAN_GO, 0)              # cleared when passing

        # wetting (290 = 8) → scan (290 = 6)
        self._set(self.STATE, 8)
        if not self._tick(self.STEP_SECONDS):
            return
        if not self._scan_rows(rows):
            return

        # finish (290 = 5). The operating UI decides cell completion from this value.
        self._set(self.STATE, 5)
        self._set(self.FINISHED, 1)
        self._set(self.PROGRESS, 100)
        self._set(self.HOME_FLAG, 1)            # back home
        self._set(self.TASK_STATE, 3)           # task finished
        print(f"[Task] cell scan done ({rows} rows)")


class DashboardServer(threading.Thread):
    """29999 text command server."""

    def __init__(self, bank: RegisterBank, motion: "MotionSim",
                 task: "ScanTaskSim", host: str, port: int):
        super().__init__(daemon=True)
        self.bank, self.motion, self.task = bank, motion, task
        self.host, self.port = host, port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(4)

    def run(self) -> None:
        print(f"[Dashboard 29999] listening on {self.host}:{self.port}")
        while True:
            conn, addr = self._sock.accept()
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            try:
                conn.sendall(b"Connected: Elite Robot Dashboard Simulator\n")
                buf = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        return
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        self._respond(conn, line.decode("utf-8", "ignore").strip())
            except (ConnectionError, OSError):
                return

    def _respond(self, conn: socket.socket, command: str) -> None:
        replies = {
            "robotMode": "Robotmode: RUNNING",
            "status": "Status: RUNNING",
            "robotControl -on": "Powering on",
            "robotControl -off": "Powering off",
            "brakeRelease": "Brake releasing",
        }
        if command == "variable -get Home_joint":
            home = [round(v / 1000.0, 6) for v in self.motion.home_joints_mrad()]
            reply = f"Home_joint = {home}"
        elif command.startswith("task -p "):
            self.task.load(command[len("task -p "):].strip())
            reply = f"ok: {command}"
        elif command == "play":
            # if paused, continue from there; otherwise start from the beginning.
            self.task.play()
            reply = "Starting program"
        elif command == "pause":
            reply = ("Pausing program" if self.task.pause()
                     else "Failed to execute: pause")
        elif command == "stop":
            self.motion.stop_all()
            self.task.stop()
            reply = "Stopping program"
        else:
            reply = replies.get(command, f"ok: {command}")
        print(f"[Dashboard] {command!r} -> {reply!r}")
        conn.sendall((reply + "\n").encode("utf-8"))


class MotionSim:
    """Physics loop that applies the speeds received with speedj/speedl to the registers."""

    RATE_HZ = 20

    def __init__(self, bank: RegisterBank):
        self.bank = bank
        self._lock = threading.Lock()
        self._joint_velocity = [0.0] * 6   # rad/s
        self._tcp_velocity = [0.0] * 6      # [m/s]*3 + [rad/s]*3
        self._joint_entry = bank.registers.read_entry("joint_position")
        self._tcp_entry = bank.registers.read_entry("tcp_absolute")
        self._joint_pos = [float(v) for v in bank.read(self._joint_entry.address, 6)]
        self._home_seed = list(self._joint_pos)
        self._tcp_pos = [float(v) for v in bank.read(self._tcp_entry.address, 6)]

    #: offset [mrad] from home on joint 2 when treated as the extended (working) pose.
    DEPLOYED_OFFSET_MRAD = 400.0

    def home_joints_mrad(self) -> list[float]:
        """Simulator home joints [mrad]. If 308 is 1, 310~315, otherwise the initial values."""
        if self.bank.read(308, 1)[0] == 1:
            regs = self.bank.read(310, 6)
            if any(regs):
                return [float(v) for v in regs]
        return list(self._home_seed)

    def set_pose(self, at_home: bool) -> None:
        """Move the joints to home / the extended pose.

        The node decides home from the joints, so the home flag (276) and the joints are changed
        together to stay consistent.
        """
        joints = self.home_joints_mrad()
        if not at_home:
            joints[1] += self.DEPLOYED_OFFSET_MRAD
        with self._lock:
            self._joint_pos = joints

    def set_joint_velocity(self, qd: list[float]) -> None:
        with self._lock:
            self._joint_velocity = list(qd)

    def set_tcp_velocity(self, xd: list[float]) -> None:
        with self._lock:
            self._tcp_velocity = list(xd)

    def stop_all(self) -> None:
        with self._lock:
            self._joint_velocity = [0.0] * 6
            self._tcp_velocity = [0.0] * 6

    def move_joint_to(self, joints_rad: list[float] | None,
                      writes: list[tuple[int, int]] | None = None) -> None:
        """Move to the target joints (movej) immediately (after a short delay).

        `writes` are registers written **after** the move in the same script —
        when the node's home script arrives it writes 1 to 276 (home flag).
        """
        def apply():
            time.sleep(0.8)
            if joints_rad is not None:
                with self._lock:
                    self._joint_pos = [v * 1000.0 for v in joints_rad]  # rad -> mrad(raw)
            else:
                # the target is a variable (tgt_j), so the value is unknown — it is the node's home script, so use the home pose.
                self.set_pose(True)
            for address, value in writes or []:
                self.bank.write(address, value)
            print(f"[Motion] moved to the home joints: {joints_rad}")
        threading.Thread(target=apply, daemon=True).start()

    def run_forever(self) -> None:
        dt = 1.0 / self.RATE_HZ
        joint_scale = self.bank.registers.scales_for(self._joint_entry)
        tcp_scale = self.bank.registers.scales_for(self._tcp_entry)
        while True:
            time.sleep(dt)
            with self._lock:
                jv, tv = list(self._joint_velocity), list(self._tcp_velocity)
                for i in range(6):
                    # rad/s or m·rad/s -> milli units -> register counts
                    self._joint_pos[i] += (jv[i] * 1000.0 / joint_scale[i]) * dt
                    self._tcp_pos[i] += (tv[i] * 1000.0 / tcp_scale[i]) * dt
                joint_raw = [int(round(v)) for v in self._joint_pos]
                tcp_raw = [int(round(v)) for v in self._tcp_pos]
            self.bank.write_many(self._joint_entry.address, joint_raw)
            self.bank.write_many(self._tcp_entry.address, tcp_raw)


_SPEEDJ_RE = re.compile(r"speedj\((\[[^\]]*\])")
_SPEEDL_RE = re.compile(r"speedl\((\[[^\]]*\])")
_MOVEJ_RE = re.compile(r"movej\((\[[^\]]*\])")
_WRITE_RE = re.compile(r"write_port_register\((\d+),\s*(-?\d+)\)")


class PrimaryServer(threading.Thread):
    """30001 script server. Interprets the jog and home move scripts."""

    def __init__(self, motion: MotionSim, host: str, port: int):
        super().__init__(daemon=True)
        self.motion = motion
        self.host, self.port = host, port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(4)

    def run(self) -> None:
        print(f"[Primary 30001] listening on {self.host}:{self.port}")
        while True:
            conn, addr = self._sock.accept()
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            buf = ""
            try:
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        return
                    buf += chunk.decode("utf-8", "ignore")
                    if len(buf) > 8192:
                        buf = buf[-4096:]
                    if self._scan(buf):
                        buf = ""
            except (ConnectionError, OSError):
                self.motion.stop_all()
                return

    def _scan(self, buf: str) -> bool:
        """Interpret a script. True if a motion command was handled (clears the buffer)."""
        m = _SPEEDJ_RE.search(buf)
        if m:
            self.motion.set_joint_velocity(eval(m.group(1)))  # noqa: S307 - trusted internal script
        m = _SPEEDL_RE.search(buf)
        if m:
            self.motion.set_tcp_velocity(eval(m.group(1)))  # noqa: S307
        m = _MOVEJ_RE.search(buf)
        if m:
            writes = [(int(a), int(v)) for a, v in _WRITE_RE.findall(buf[m.end():])]
            self.motion.move_joint_to(eval(m.group(1)), writes)  # noqa: S307
            return True
        # the node's home script passes the target as a variable (tgt_j) — the value is unknown,
        # so the joints stay and only the register written after the script (276) is applied.
        idx = buf.find("movej(")
        if idx >= 0 and buf.rstrip().endswith("end"):
            writes = [(int(a), int(v)) for a, v in _WRITE_RE.findall(buf[idx:])]
            self.motion.move_joint_to(None, writes)
            return True
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dash-port", type=int, default=29999)
    parser.add_argument("--primary-port", type=int, default=30001)
    parser.add_argument("--modbus-port", type=int, default=5502)
    parser.add_argument(
        "--wall-error", type=float, default=ScanTaskSim.WALL_ERROR_MM,
        help="assume the real wall is off from the input radius by this much [mm]")
    parser.add_argument(
        "--max-rows", type=int, default=ScanTaskSim.MAX_ROWS,
        help="upper limit of zigzag rows (to finish tests quickly)")
    args = parser.parse_args()

    registers = register_map.load()
    bank = RegisterBank(registers)
    motion = MotionSim(bank)
    task = ScanTaskSim(bank, wall_error_mm=args.wall_error, max_rows=args.max_rows,
                       motion=motion)
    task.start()

    ModbusServer(bank, args.host, args.modbus_port).start()
    DashboardServer(bank, motion, task, args.host, args.dash_port).start()
    PrimaryServer(motion, args.host, args.primary_port).start()

    print(
        f"Ready. Start the node like this:\n"
        f"  ros2 run elite_robot_controller robot_control_node --ros-args "
        f"-p robot_ip:={args.host} -p modbus_port:={args.modbus_port}\n"
        f"Exit with Ctrl+C."
    )
    try:
        motion.run_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
