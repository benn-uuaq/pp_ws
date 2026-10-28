import struct
import socket
import select
import time
import queue
import threading
from pyModbusTCP.client import ModbusClient

DEFAULT_TIMEOUT = 10.0
MESSAGE_TYPE_ROBOT_STATE = 16
MESSAGE_TYPE_ROBOT_MESSAGE = 20

FMT_HEADER = 'IB'
FMT_ROBOT_MODE = 'IBQ???????BBdddB??I'
FMT_JOINT_HEADER = 'IB'
FMT_JOINT_DATA = 'dddiiiffffBI'
FMT_CARTESIAN = 'IBdddddddddddd'
FMT_CONFIG = 'IB'+'dd'*6+'dd'*6+'ddddd'+'d'*6+'d'*6+'d'*6+'d'*6+'IIIBBBB'
FMT_MASTERBOARD = 'IBIIBBBdddBBBdddffffB???B'
FMT_ADDITIONAL = 'IB????B'
FMT_TOOL = 'IBBBddfBffB'
FMT_SAFETY = 'IBIbBdddd'
FMT_TOOL_COMM = 'IB?III?Bff'


class RobotDataConfig():
    def __init__(self):
        self.names_pre = [
            'total_message_len', 'total_message_type',
            'mode_sub_len', 'mode_sub_type', 'timestamp', 'reserved_1', 'reserved_2',
            'is_robot_power_on', 'is_emergency_stopped', 'is_robot_protective_stopped',
            'is_task_running', 'is_task_paused', 'robot_mode', 'robot_control_mode',
            'target_speed_fraction', 'speed_scaling', 'target_speed_fraction_limit',
            'get_robot_speed_mode', 'reserved_3', 'is_in_package_mode', 'reserved_4',
            'joint_sub_len', 'joint_sub_type'
        ]
        self.names_joint = [
            'actual_joint', 'target_joint', 'actual_velocity',
            'joint_reserved_1', 'joint_reserved_2', 'joint_reserved_3',
            'current', 'voltage', 'temperature', 'torques', 'mode', 'joint_reserved_4'
        ]
        self.names_post = [
            'cartesial_sub_len', 'cartesial_sub_type',
            'tcp_x', 'tcp_y', 'tcp_z', 'rot_x', 'rot_y', 'rot_z',
            'offset_px', 'offset_py', 'offset_pz', 'offset_rotx', 'offset_roty', 'offset_rotz',
            'configuration_sub_len', 'configuration_sub_type',
            'limit_min_joint_0', 'limit_max_joint_0', 'limit_min_joint_1', 'limit_max_joint_1',
            'limit_min_joint_2', 'limit_max_joint_2', 'limit_min_joint_3', 'limit_max_joint_3',
            'limit_min_joint_4', 'limit_max_joint_4', 'limit_min_joint_5', 'limit_max_joint_5',
            'max_velocity_joint_0', 'max_acc_joint_0', 'max_velocity_joint_1', 'max_acc_joint_1',
            'max_velocity_joint_2', 'max_acc_joint_2', 'max_velocity_joint_3', 'max_acc_joint_3',
            'max_velocity_joint_4', 'max_acc_joint_4', 'max_velocity_joint_5', 'max_acc_joint_5',
            'default_velocity_joint', 'default_acc_joint', 'default_tool_velocity',
            'default_tool_acc', 'internal_use',
            'dh_a_joint_0', 'dh_a_joint_1', 'dh_a_joint_2',
            'dh_a_joint_3', 'dh_a_joint_4', 'dh_a_joint_5',
            'dh_d_joint_0', 'dh_d_joint_1', 'dh_d_joint_2',
            'dh_d_joint_3', 'dh_d_joint_4', 'dh_d_joint_5',
            'dh_alpha_joint_0', 'dh_alpha_joint_1', 'dh_alpha_joint_2',
            'dh_alpha_joint_3', 'dh_alpha_joint_4', 'dh_alpha_joint_5',
            'dh_theta_joint_0', 'dh_theta_joint_1', 'dh_theta_joint_2',
            'dh_theta_joint_3', 'dh_theta_joint_4', 'dh_theta_joint_5',
            'masterboard_version', 'control_box_type', 'robot_type', 'robot_structure',
            'tool_io_type', 'reserved_cfg2', 'reserved_cfg3',
            'masterboard_sub_len', 'masterboard_sub_type',
            'digital_input_bits', 'digital_output_bits',
            'standard_analog_input_domain0', 'standard_analog_input_domain1',
            'tool_analog_input_domain',
            'standard_analog_input_value0', 'standard_analog_input_value1',
            'tool_analog_input_value',
            'standard_analog_output_domain0', 'standard_analog_output_domain1',
            'tool_analog_output_domain',
            'standard_analog_output_value0', 'standard_analog_output_value1',
            'tool_analog_output_value',
            'masterrbord_temperature', 'robot_voltage', 'robot_current', 'io_current',
            'safety_mode', 'is_robot_in_reduced_mode', 'operational_mode_selector_input',
            'threeposition_enabling_device_input', 'internal_use_mb',
            'additional_sub_len', 'additional_sub_type',
            'is_freedrive_button_pressed', 'reserved_add', 'is_freedrive_io_enabled',
            'is_dynamic_collision_detect_enabled', 'reserved_add2',
            'tool_sub_len', 'tool_sub_type',
            'tool_analog_output_domain', 'tool_analog_input_domain',
            'tool_analog_output_value', 'tool_analog_input_value',
            'tool_voltage', 'tool_output_voltage', 'tool_current', 'tool_temperature',
            'tool_mode',
            'safe_sub_len', 'safe_sub_type',
            'safety_crc_num', 'safety_operational_mode', 'reserved_safe',
            'current_elbow_position_x', 'current_elbow_position_y',
            'current_elbow_position_z', 'elbow_radius',
            'tool_comm_sub_len', 'tool_comm_sub_type',
            'is_enable', 'baudrate', 'parity', 'stopbits', 'tci_modbus_status',
            'tci_usage', 'reserved_tc1', 'reserved_tc2'
        ]
        self.fmt = (
            '>' +
            FMT_HEADER + FMT_ROBOT_MODE +
            FMT_JOINT_HEADER + (FMT_JOINT_DATA * 6) +
            FMT_CARTESIAN + FMT_CONFIG + FMT_MASTERBOARD +
            FMT_ADDITIONAL + FMT_TOOL + FMT_SAFETY + FMT_TOOL_COMM
        )


class RobotHeader():
    __slots__ = ['type', 'size']

    @staticmethod
    def unpack(buf):
        rmd = RobotHeader()
        (rmd.size, rmd.type) = struct.unpack_from('>iB', buf)
        return rmd


class RobotData():
    @staticmethod
    def unpack(buf, config):
        data = RobotData()
        try:
            unpacked = struct.unpack(config.fmt, buf)
            it = iter(unpacked)
            for name in config.names_pre:
                setattr(data, name, next(it))
            for name in config.names_joint:
                setattr(data, name, [])
            for _ in range(6):
                for name in config.names_joint:
                    getattr(data, name).append(next(it))
            for name in config.names_post:
                setattr(data, name, next(it))
            return data
        except (struct.error, StopIteration):
            return None


class AlarmData:
    def __init__(self, code=None, sub=None, level=None, msg=None):
        self.code = code
        self.sub = sub
        self.level = level
        self.msg = msg
        self.timestamp = time.time()
        self.active = True


class ReadAlarm():
    @staticmethod
    def unpack(buf):
        data_length = struct.unpack(">i", buf[0:4])[0]
        data = buf[0:data_length]
        msg_type = data[14]
        if msg_type == 10:
            msg = bytearray(data[23:data_length-1]).decode()
            return AlarmData(msg=msg)
        if msg_type == 6:
            error_code = struct.unpack(">i", data[15:19])[0]
            sub_error_code = struct.unpack(">i", data[19:23])[0]
            level = struct.unpack(">i", data[23:27])[0]
            return AlarmData(code=error_code, sub=sub_error_code, level=level)
        return None


class Robot_30001():
    def __init__(self, ip, port2) -> None:
        self.__data_config = RobotDataConfig()
        self.ip = ip
        self.port2 = port2
        self.alarm_queue = queue.Queue()
        self.__sock = None
        self.__buf = b""

    def connect_30001(self):
        try:
            self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__sock.settimeout(0.5)
            self.__sock.connect((self.ip, self.port2))
            self.__sock.settimeout(10.0)
            print(f"[DEBUG][Robot Primary] Connected to {self.ip} on port {self.port2}")
            self.__buf = b""
            return self.__sock
        except Exception as e:
            print(f"[ERROR][Robot Primary] Connecting to {self.ip} on port {self.port2}: {e}")
            self.__sock = None
            return None

    def disconnect_30001(self):
        if self.__sock:
            self.__sock.close()
            self.__sock = None
            print("[Robot Primary] Disconnect")

    def send_script(self, script):
        """Send a script body to 30001. There is no reply.

        Used to move the robot outside the task, e.g. jog (speedl/speedj) and the home move.
        Several lines are sent as they are with a newline appended at the end.
        """
        if self.__sock is None:
            return False
        try:
            body = script if script.endswith("\n") else script + "\n"
            self.__sock.sendall(body.encode("utf-8"))
            return True
        except Exception as e:
            print(f"[ERROR][Robot Primary] Sending script: {e}")
            return False

    def get_data(self):
        return self.__recv()

    def __recv(self):
        try:
            self.__read_socket_no_wait()
        except Exception:
            return None
        last_valid_data = None
        while len(self.__buf) >= 5:
            try:
                head = RobotHeader.unpack(self.__buf)
            except Exception:       # broken header — drop the buffer and resynchronise
                self.__buf = b""
                break
            if len(self.__buf) < head.size:
                break
            payload = self.__buf[:head.size]
            self.__buf = self.__buf[head.size:]
            if head.type == MESSAGE_TYPE_ROBOT_MESSAGE:
                try:
                    alarm = ReadAlarm.unpack(payload)
                    if alarm:
                        self.alarm_queue.put(alarm)
                except Exception:   # receiving continues even if one alarm is broken
                    pass
                continue
            if head.type == MESSAGE_TYPE_ROBOT_STATE:
                try:
                    last_valid_data = RobotData.unpack(payload, self.__data_config)
                except Exception:   # receiving continues even if one state packet is broken
                    pass
        return last_valid_data

    def __read_socket_no_wait(self):
        while True:
            readable, _, _ = select.select([self.__sock], [], [], 0)
            if not readable:
                break
            try:
                more = self.__sock.recv(4096)
                if not more:
                    raise ConnectionError("Socket closed")
                self.__buf += more
            except BlockingIOError:
                break
            except Exception:
                break


class AlarmManager:
    """Filter only duplicates of the same alarm within a short time; report it again if it recurs.

    Previously a (code, sub, msg) key stayed in the dict forever once seen, so
    when the same alarm happened again later (e.g. an inverse kinematics failure repeating for the same cause)
    it was never reported again while the process ran — RCS operators missed recurrences.
    Now only the time last seen is remembered, and once
    `dedup_window` seconds have passed the same alarm is reported again as new.
    (Only one physical event split into several packets at once is suppressed.)
    """

    def __init__(self, dedup_window=2.0):
        self.dedup_window = dedup_window
        self._last_seen = {}

    def process(self, alarm):
        key = (alarm.code, alarm.sub, alarm.msg)
        now = time.time()
        last = self._last_seen.get(key)
        if last is not None and (now - last) < self.dedup_window:
            return False
        self._last_seen[key] = now
        print("================================")
        print("[ALARM TRIGGERED]")
        if alarm.msg:
            print(f"[ALARM MSG] {alarm.msg}")
        else:
            print(f"[ALARM CODE] E{alarm.code} S{alarm.sub} (level={alarm.level})")
        print("================================")
        return True


class Robot_29999():
    """Robot dashboard (29999) client.

    **Reconnects broken sockets.** Previously, when one command failed with an exception,
    the broken socket was kept, so every later command returned None forever
    (`[play] Result: None`). The controller drops this connection when the task is reloaded
    on the pendant or the power is cycled, and there was no code to reconnect.
    Now on failure the socket is dropped, a new one is connected and the command is sent **once more**.

    The failure reason is kept in `last_error` — returning only None leaves the receiver
    nothing to show but "Result: None".
    """

    #: wait time [s] for one command's reply. The dashboard usually answers immediately.
    #: Too long and the node (single-threaded executor) stops Modbus polling meanwhile.
    RECV_TIMEOUT_S = 3.0
    #: wait time [s] when connecting.
    CONNECT_TIMEOUT_S = 2.0

    def __init__(self, ip, port1):
        self.sock = None
        self.ip = ip
        self.port1 = port1
        self.last_error = ""
        # keeps each command / reply pair from mixing with other commands. One socket is
        # shared by speed changes, play and stop.
        self._lock = threading.Lock()

    def connect_29999(self):
        # close an already connected socket and connect anew (no leaked sockets).
        self._close_quietly()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.CONNECT_TIMEOUT_S)
            sock.connect((self.ip, self.port1))
            sock.settimeout(self.RECV_TIMEOUT_S)
            print(f"[DEBUG][Robot Dashboard] Connected to {self.ip} on port {self.port1}")
            # flush the connection greeting first. Otherwise the first command reads the greeting
            # as its own reply.
            try:
                sock.recv(4096)
            except Exception:
                pass
            self.sock = sock
            self.last_error = ""
            return self.sock
        except Exception as e:
            self.last_error = f"dashboard ({self.ip}:{self.port1}) connection failed: {e}"
            print(f"[ERROR][Robot Dashboard] {self.last_error}")
            self.sock = None
            return None

    def send_command_29999(self, command):
        """Send a command and return one reply line. None on failure.

        On failure the socket is dropped, reconnected and the command tried once more. If that also fails,
        None is returned and the reason is kept in `last_error`.
        """
        with self._lock:
            for attempt in (1, 2):
                if self.sock is None and self.connect_29999() is None:
                    return None
                try:
                    self._drain()
                    self.sock.sendall(f"{command}\n".encode("utf-8"))
                    raw = self.sock.recv(4096)
                    if not raw:
                        # the controller closed the connection. Reconnect and try once more.
                        raise ConnectionError("the dashboard closed the connection")
                    self.last_error = ""
                    return raw.decode("utf-8", errors="replace").strip()
                except Exception as e:
                    self.last_error = f"failed to send '{command}': {e}"
                    print(f"[ERROR][Robot Dashboard] {self.last_error} "
                          f"(attempt {attempt}/2)")
                    self._close_quietly()
            return None

    def _drain(self):
        """Drop a late reply to an earlier command if one is left.

        If left, this command would read it as its own reply and every later reply would be
        shifted by one (play reading the reply of stop, etc.).
        """
        if self.sock is None:
            return
        try:
            self.sock.setblocking(False)
            while True:
                if not self.sock.recv(4096):
                    break
        except (BlockingIOError, InterruptedError):
            pass
        except OSError:
            pass
        finally:
            if self.sock is not None:
                self.sock.settimeout(self.RECV_TIMEOUT_S)

    def _close_quietly(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def disconnect_29999(self):
        with self._lock:
            if self.sock:
                self._close_quietly()
                print("[Robot Dashboard] Disconnect")

    def robot_mode(self): return self.send_command_29999("robotMode")
    def robot_status(self): return self.send_command_29999("status")
    def robot_power_on(self): return self.send_command_29999("robotControl -on")
    def robot_power_off(self): return self.send_command_29999("robotControl -off")
    def robot_brakeRelease(self): return self.send_command_29999("brakeRelease")
    # play/stop are only accepted in remote control mode. In local control mode the controller
    # refuses with "not supported in local control mode", so it must be turned on first.
    def robot_remote_control_on(self): return self.send_command_29999("remoteControl -on")
    def robot_remote_control_off(self): return self.send_command_29999("remoteControl -off")

    def robot_set_speed(self, percent):
        """Change the robot's overall motion speed ratio [%] in real time. 2~100.

        The format in the dashboard command table (robot_task/29999 dashboard command.png) is
        `speed -v 50` ("Set the robot speed to 50% (Range: 2%~100%)").
        Previously `speed -set N`, which is not in the table, was sent and the controller did not
        take the ratio properly. Writing register 17 directly gives the same value, but
        the control path is kept in one place (29999). Register 17 is used for reading.
        """
        percent = max(2, min(100, int(percent)))
        return self.send_command_29999(f"speed -v {percent}")

    def robot_play(self): return self.send_command_29999("play")
    def robot_pause(self): return self.send_command_29999("pause")
    def robot_stop(self): return self.send_command_29999("stop")

    def robot_task_status(self):
        """Ask via 29999 for the state of the task currently loaded on the robot.

        The task is fixed on the pendant / robot side, so the operating UI does not choose it —
        it only shows this reply as is. The reply format is exactly what the controller
        gives (e.g. "Task is running"); it is not cut or processed — showing the original text is safer
        than parsing it wrongly and losing information.
        """
        return self.send_command_29999("task -s")

    def set_variable(self, name, value):
        """Set a global variable on the robot via 29999."""
        return self.send_command_29999(f"variable -set {name} {value}")

    def get_variable(self, var_name):
        """Read a global variable on the robot via 29999 and return it as a Python type.

        - list ("[1.1, 2.2]") -> list[float]
        - boolean ("True"/"False") -> bool
        - number -> int/float
        - otherwise -> str
        None if there is no value or an error, "NOT_FOUND" if the variable is not found.
        """
        try:
            response = self.send_command_29999(f"variable -get {var_name}")
            if response is None:
                return None
            if "Error" in response or "undefined" in response or "Can not find" in response:
                return "NOT_FOUND"

            if "[" in response and "]" in response:
                content = response[response.find("[") + 1:response.find("]")]
                result = []
                for item in content.split(","):
                    item = item.strip()
                    if not item:
                        continue
                    try:
                        result.append(float(item))
                    except ValueError:
                        result.append(item)
                return result

            raw = response.split(",")[-1].strip() if "," in response else response.strip()
            if raw.lower() == "true":
                return True
            if raw.lower() == "false":
                return False
            try:
                return int(raw)
            except ValueError:
                pass
            try:
                return float(raw)
            except ValueError:
                pass
            if (raw.startswith('"') and raw.endswith('"')) or \
               (raw.startswith("'") and raw.endswith("'")):
                return raw[1:-1]
            return raw
        except Exception as e:
            print(f"[ERROR][Robot Dashboard] variable -get {var_name}: {e}")
            return None


class Robot_modbus():
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client = None
        self.is_running = False

    def connect(self):
        self.client = ModbusClient(host=self.host, port=self.port, unit_id=255, timeout=1.0)

        if self.client.is_open:
            self.client.close()

        is_open = self.client.open()
        if is_open:
            self.is_running = True
            print(f"[DEBUG][Robot Modbus] Connected to {self.host} on port {self.port}")
            return True
        else:
            print("[ERROR][Robot Modbus] Connecting to "
                  f"{self.host} on port {self.port}: timed out")
            return False

    def disconnect(self):
        if self.client:
            self.client.close()
        self.is_running = False
        print("[Robot Modbus] Disconnect")

    def set_register(self, address, value):
        try:
            write_data = int(float(value))
        except ValueError:
            write_data = 1 if str(value).strip().lower() in ['true', '1'] else 0
        modbus_write_val = write_data & 0xFFFF
        is_success = self.client.write_single_register(address, modbus_write_val)
        if not is_success:
            return False
        import time
        time.sleep(0.05)

        signed_read_val = self.get_register(address)
        if signed_read_val is not None and write_data == signed_read_val:
            return True
        return False

    def get_all_registers(self, start_address, count) -> list:
        regs = self.client.read_holding_registers(start_address, count)
        if regs is None:
            regs = self.client.read_input_registers(start_address, count)
        if regs:
            return [val - 65536 if val > 32767 else val for val in regs]
        return []

    def get_register(self, address) -> int:
        result = self.client.read_holding_registers(address, 1)
        if result is None:
            result = self.client.read_input_registers(address, 1)
        if result and len(result) > 0:
            val = result[0]
            return val - 65536 if val > 32767 else val
        return None
