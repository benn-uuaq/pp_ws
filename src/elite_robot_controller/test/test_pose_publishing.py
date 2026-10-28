"""Checks applying the register map, publishing poses and refusing writes.

The Modbus client and publishers are replaced so it runs without a robot.
RobotControlNode.__init__ requires connecting three channels, so it is bypassed.
"""

import threading
from unittest.mock import patch

from elite_robot_controller import register_map
from elite_robot_controller.robot.robot_control_node import RobotControlNode


class FakeModbus:
    """Modbus stand-in that answers only for the given start addresses."""

    def __init__(self, registers=None):
        self.registers = registers or {}
        self.reads = []
        self.writes = []
        self.write_result = True

    def get_all_registers(self, start_address, count):
        self.reads.append((start_address, count))
        return self.registers.get(start_address, [])

    def get_register(self, address):
        self.reads.append((address, 1))
        values = self.registers.get(address)
        return values[0] if values else None

    def set_register(self, address, value):
        self.writes.append((address, value))
        return self.write_result


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(list(msg.data) if hasattr(msg.data, "__len__") else msg.data)


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warn(self, message):
        self.warnings.append(message)

    def info(self, message):
        self.infos.append(message)

    def error(self, message):
        self.warnings.append(message)


class FakeResponse:
    success = False
    message = ""


def make_node(registers=None, map_data=None):
    node = RobotControlNode.__new__(RobotControlNode)
    node.robot_modbus = FakeModbus(registers)
    node.registers = register_map.RegisterMap(map_data or _default_map())
    node._logger = FakeLogger()
    node.get_logger = lambda: node._logger
    # __init__ is skipped, so fill in the connection watch state directly.
    node._connect_lock = threading.Lock()
    node._want_connected = True
    node._link_fails = 0
    return node


def _default_map():
    return {
        "scale": {"position_per_count": 0.1, "rotation_per_count": 1.0},
        "read": {
            "robot_mode": {"address": 66, "count": 1},
            "tcp_absolute": {"address": 384, "count": 6},
            "tcp_zero_relative": {"address": 280, "count": 6},
            "joint_position": {"address": None, "count": 6},
        },
        "write": {
            "linear_speed": {"address": 300, "count": 1},
            "save_home_pose": {"address": None, "count": 1},
        },
    }


def test_shipped_register_map_matches_known_addresses():
    """The config file shipped with the package must keep the confirmed addresses."""
    registers = register_map.load()

    assert registers.read_entry("tcp_absolute").address == 384
    assert registers.read_entry("tcp_absolute").count == 6
    assert registers.read_entry("tcp_zero_relative").address == 280
    assert registers.read_entry("robot_mode").address == 66
    assert registers.read_entry("control_method").address == 71
    assert registers.read_entry("operation_mode").address == 72
    assert registers.read_entry("joint_position").address == 73
    # write addresses must be in the general-purpose register range read by the robot task.
    for name in ("linear_speed", "speed_ratio", "home_joint", "standby_joint",
                 "start_pose", "home_src", "pose_src"):
        entry = registers.write_entry(name)
        assert entry.available, f"{name} address is empty"
        assert 256 <= entry.address <= 383, f"{name} is outside the general-purpose range"
    # 2026-10-16: 310~315 is the scan standby pose (formerly home), home is 322~327 + 328.
    assert registers.write_entry("standby_joint").address == 310
    assert registers.write_entry("home_joint").address == 322
    assert registers.write_entry("home_src").address == 328
    for name in ("home_joint", "standby_joint"):
        assert registers.write_entry(name).count == 6
        assert registers.write_entry(name).kind == "angle"


def test_joint_angles_use_rotation_scale_for_every_axis():
    """Joint angles are mrad for all 6 axes, so the position conversion must not be applied."""
    registers = register_map.load()
    entry = registers.read_entry("joint_position")

    assert entry.kind == "angle"
    assert registers.scales_for(entry) == [1.0] * 6
    # for poses only the first 3 get the position conversion.
    pose = registers.read_entry("tcp_absolute")
    assert registers.scales_for(pose) == [0.1, 0.1, 0.1, 1.0, 1.0, 1.0]


def test_angle_entry_is_published_without_position_scaling():
    node = make_node(
        {73: [207, -1466, -1875, -1371, 1570, 207]},
        {
            "scale": {"position_per_count": 0.1, "rotation_per_count": 1.0},
            "read": {"joint_position": {"address": 73, "count": 6, "kind": "angle"}},
            "write": {},
        },
    )
    publisher = FakePublisher()

    node.publish_pose("joint_position", publisher)

    assert publisher.messages[0] == [207.0, -1466.0, -1875.0, -1371.0, 1570.0, 207.0]


def test_pose_uses_addresses_from_map():
    node = make_node({384: [0] * 6, 280: [0] * 6})
    absolute, zero = FakePublisher(), FakePublisher()

    node.publish_pose("tcp_absolute", absolute)
    node.publish_pose("tcp_zero_relative", zero)

    assert node.robot_modbus.reads == [(384, 6), (280, 6)]
    assert len(absolute.messages) == 1
    assert len(zero.messages) == 1


def test_position_and_rotation_scaling():
    """Positions convert in 0.1 mm units, rotations in 1 mrad units."""
    node = make_node({384: [1205, -3400, 5000, 1571, 0, -3141]})
    publisher = FakePublisher()

    node.publish_pose("tcp_absolute", publisher)

    assert publisher.messages[0] == [120.5, -340.0, 500.0, 1571.0, 0.0, -3141.0]


def test_short_or_missing_response_is_not_published():
    """If not all registers can be read, no wrong pose is published."""
    node = make_node({384: [1, 2, 3]})
    publisher = FakePublisher()

    node.publish_pose("tcp_absolute", publisher)
    node.publish_pose("tcp_zero_relative", publisher)

    assert publisher.messages == []


def test_entry_without_address_is_never_read():
    """Entries without an address make no Modbus request at all."""
    node = make_node()
    publisher = FakePublisher()

    node.publish_pose("joint_position", publisher)

    assert node.robot_modbus.reads == []
    assert publisher.messages == []


def test_write_is_rejected_when_address_is_unset():
    """Entries without a decided address are not written. Touching a wrong register is dangerous."""
    node = make_node()

    success, message = node.write_register("save_home_pose", 1)

    assert success is False
    assert "address is not configured" in message
    assert node.robot_modbus.writes == []


def test_write_uses_address_from_map():
    node = make_node()

    success, message = node.write_register("linear_speed", 150)

    assert success is True
    assert node.robot_modbus.writes == [(300, 150)]
    assert "300" in message


def test_write_failure_is_reported():
    node = make_node()
    node.robot_modbus.write_result = False

    success, _ = node.write_register("linear_speed", 150)

    assert success is False


def test_pose_write_fills_every_register():
    """Saving a pose fills 6 registers at once."""
    node = make_node(map_data={
        "scale": {"position_per_count": 0.1, "rotation_per_count": 1.0},
        "read": {},
        "write": {
            "home_joint": {"address": 310, "count": 6, "kind": "angle"},
            "pose_src": {"address": 308, "count": 1},
        },
    })

    success, _ = node.write_pose("home_joint", [207.0, -1466.0, -1875.0, -1371.0, 1570.0, 207.0])

    assert success is True
    assert node.robot_modbus.writes == [
        (310, 207), (311, -1466), (312, -1875), (313, -1371), (314, 1570), (315, 207)
    ]


def test_pose_write_rejects_wrong_length():
    """If components are missing, nothing is half-written."""
    node = make_node()
    node.registers = register_map.RegisterMap({
        "scale": {}, "read": {},
        "write": {"home_joint": {"address": 310, "count": 6, "kind": "angle"}},
    })

    success, message = node.write_pose("home_joint", [1.0, 2.0])

    assert success is False
    assert "expected 6 values" in message
    assert node.robot_modbus.writes == []


def test_jog_code_maps_axis_and_direction():
    """The sign is the direction, the absolute value the axis number. Out of range does not move."""
    node = make_node()

    assert node._jog_vector(1) == [1.0, 0, 0, 0, 0, 0]
    assert node._jog_vector(-6) == [0, 0, 0, 0, 0, -1.0]
    assert node._jog_vector(7) is None
    assert node._jog_vector(0) is None


class FakeChannel:
    """Communication channel that only imitates connect / disconnect."""

    def __init__(self, ok=True):
        self.ok = ok
        self.opened = 0
        self.closed = 0

    def connect(self):
        self.opened += 1
        return self.ok

    def disconnect(self):
        self.closed += 1


def make_connectable_node(dash_ok=True, primary_ok=True, modbus_ok=True):
    node = make_node()
    node.robot_ip = "10.0.0.9"
    node.robot_dash = FakeChannel(dash_ok)
    node.robot_primary = FakeChannel(primary_ok)
    node.robot_modbus_channel = FakeChannel(modbus_ok)
    node.robot_dash.connect_29999 = node.robot_dash.connect
    node.robot_dash.disconnect_29999 = node.robot_dash.disconnect
    node.robot_primary.connect_30001 = node.robot_primary.connect
    node.robot_primary.disconnect_30001 = node.robot_primary.disconnect
    modbus = node.robot_modbus
    modbus.connect = node.robot_modbus_channel.connect
    modbus.disconnect = node.robot_modbus_channel.disconnect
    node.connected = False
    node.pub_connected = FakePublisher()
    return node


def test_connect_requires_all_three_channels():
    """If any channel fails, it does not count as connected."""
    node = make_connectable_node(modbus_ok=False)

    assert node.connect_all_servers() is False
    assert node.connected is False

    node = make_connectable_node()
    assert node.connect_all_servers() is True
    assert node.connected is True


def test_connect_service_reports_result():
    node = make_connectable_node(dash_ok=False)
    response = node.cb_connect(None, FakeResponse())

    assert response.success is False
    assert "10.0.0.9" in response.message


def test_disconnect_closes_every_channel():
    """Even if cleaning up one channel fails, the others are cleaned up."""
    node = make_connectable_node()
    node.connect_all_servers()

    def boom():
        raise RuntimeError("socket error")

    node.robot_dash.disconnect_29999 = boom
    node.cb_disconnect(None, FakeResponse())

    assert node.connected is False
    assert node.robot_primary.closed == 1
    assert node.robot_modbus_channel.closed == 1


def test_loop_does_not_touch_modbus_before_connect():
    """Touching the sockets before connecting raises an exception. Reads must be skipped."""
    node = make_connectable_node()
    node.pub_robot_mode = FakePublisher()

    node.update_robot_loop()

    assert node.robot_modbus.reads == []
    assert node.pub_connected.messages == [False]


class FakeParameter:
    def __init__(self, value):
        self.value = value


def make_jog_node(ratio=100):
    node = make_connectable_node()
    node.connected = True
    node.speed_ratio = ratio
    params = {
        'jog_joint_speed_max': 0.50,
        'jog_tcp_speed_max': 0.10,
        'jog_tcp_rot_speed_max': 0.50,
        'jog_accel_max': 1.00,
        'jog_tcp_accel_max': 0.40,
        'jog_hold_time': 0.5,
    }
    node.get_parameter = lambda name: FakeParameter(params[name])
    node.sent = []
    node.robot_primary.send_script = lambda s: node.sent.append(s) or True
    node.robot_dash.robot_stop = lambda: node.sent.append("STOP")
    return node


def test_jog_speed_follows_speed_ratio():
    """The jog speed is reduced by the ratio in register 307."""
    full = make_jog_node(100)
    full.cb_jog_joint(FakeMessageInt(1))
    assert "speedj" in full.sent[0]
    assert "0.5" in full.sent[0]

    slow = make_jog_node(20)
    slow.cb_jog_joint(FakeMessageInt(1))
    # at 20 %, 1/5 of 0.5 rad/s, and the acceleration is reduced by the same ratio.
    assert slow.sent[0] == "speedj([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], 0.2, 0.5)"


def test_jog_command_carries_hold_time():
    """A short hold time (t) makes the robot stop by itself if the commands stop."""
    node = make_jog_node()
    node.cb_jog_joint(FakeMessageInt(1))

    # the last argument is jog_hold_time. If long, the robot keeps moving that long
    # when the stop command fails.
    assert node.sent[0].endswith(", 0.5)")


def test_jog_values_have_no_float_noise():
    """They go out as script strings, so there must be no floating-point noise."""
    node = make_jog_node(20)
    node.cb_jog_tcp(FakeMessageInt(3))

    assert node.sent[0] == "speedl([0.0, 0.0, 0.02, 0.0, 0.0, 0.0], 0.08, 0.5)"


def test_jog_tcp_uses_separate_linear_and_angular_speeds():
    """Linear and rotation have different units, so they are scaled separately."""
    node = make_jog_node(50)
    node.cb_jog_tcp(FakeMessageInt(1))     # X+
    node.cb_jog_tcp(FakeMessageInt(4))     # RX+

    assert node.sent[0] == "speedl([0.05, 0.0, 0.0, 0.0, 0.0, 0.0], 0.2, 0.5)"
    assert node.sent[1] == "speedl([0.0, 0.0, 0.0, 0.25, 0.0, 0.0], 0.2, 0.5)"


def test_jog_stop_uses_dashboard_stop():
    """Stopping uses 29999 stop so the robot stops without deceleration."""
    node = make_jog_node()
    node.cb_jog_joint(FakeMessageInt(0))
    node.cb_jog_tcp(FakeMessageInt(0))

    assert node.sent == ["STOP", "STOP"]


class FakeMessageInt:
    def __init__(self, data):
        self.data = data


def test_connected_status_is_republished_every_cycle():
    """Publish every cycle even without a change, so a late subscriber still gets the value."""
    node = make_connectable_node()
    node.connect_all_servers()
    node.pub_robot_mode = FakePublisher()
    node.pub_control_method = FakePublisher()
    node.pub_op_mode = FakePublisher()
    node.pub_tcp_pose = FakePublisher()
    node.pub_tcp_pose_zero = FakePublisher()
    node.pub_joint_position = FakePublisher()
    node.pub_scan_state = FakePublisher()
    node.pub_task_state = FakePublisher()
    node.pub_home_flag = FakePublisher()
    node.pub_speed_scale = FakePublisher()
    node.pub_digital_in = FakePublisher()
    node.pub_digital_out = FakePublisher()
    node.pub_probe_result = FakePublisher()
    node.pub_probe_poses = FakePublisher()
    node.pub_scan_arc = FakePublisher()
    node.pub_alarm = FakePublisher()
    node.alarm_mgr = type("M", (), {"process": lambda self, a: False})()
    node.robot_primary.get_data = lambda: None
    node.robot_primary.alarm_queue = type("Q", (), {"empty": lambda self: True})()

    node.update_robot_loop()
    node.update_robot_loop()

    assert node.pub_connected.messages == [True, True, True]  # connect() once + loop twice


def test_work_area_register_is_raw_mm():
    """work_area is stored as plain integer mm, not 0.1 mm."""
    registers = register_map.load()
    entry = registers.write_entry("work_area")

    assert entry.address == 256
    # 256~264: arc length, height, scanner height, overlap, radius, thickness, EOAT width/height, EOAT type.
    assert entry.count == 9
    assert entry.kind == "raw"
    assert registers.scales_for(entry) == [1.0] * 9
    assert registers.write_entry("param_src").address == 266


def test_work_area_write_sets_param_src_flag():
    """Writing the work area must also send 256~259 and param_src (266)=1."""
    node = make_node(map_data={
        "scale": {"position_per_count": 0.1, "rotation_per_count": 1.0},
        "read": {},
        "write": {
            "work_area": {"address": 256, "count": 4, "kind": "raw"},
            "param_src": {"address": 266, "count": 1},
        },
    })

    node._write_pose_topic(
        "work_area", FakeMessageList([600.0, 800.0, 150.0, 20.0]), flag_name="param_src"
    )

    assert node.robot_modbus.writes == [
        (256, 600), (257, 800), (258, 150), (259, 20), (266, 1),
    ]


def test_work_area_write_skipped_without_param_src_address():
    """A flag register without an address: only the values are written, the flag is not set."""
    node = make_node(map_data={
        "scale": {"position_per_count": 0.1, "rotation_per_count": 1.0},
        "read": {},
        "write": {"work_area": {"address": 256, "count": 4, "kind": "raw"}},
    })

    node._write_pose_topic(
        "work_area", FakeMessageList([600.0, 800.0, 150.0, 20.0]), flag_name="param_src"
    )

    assert node.robot_modbus.writes == [(256, 600), (257, 800), (258, 150), (259, 20)]
    assert node.get_logger().warnings


class FakeMessageList:
    def __init__(self, data):
        self.data = data


def test_connect_rebuilds_channels_when_ip_changes():
    """When the operating UI changes the robot_ip parameter, the node must reconnect to that address.

    Previously the sockets created with the address read at node start kept being used, so however the
    IP was changed in the UI it connected only to the old address (screen and real connection disagreed).
    """
    node = make_connectable_node()
    node.robot_ip = "10.0.0.9"
    built = []

    class _Chan:
        def __init__(self, *args):
            built.append(args)

        def connect(self):
            return True

        connect_29999 = connect_30001 = connect

        def disconnect(self):
            return True

        disconnect_29999 = disconnect_30001 = disconnect

    with patch.multiple(
        "elite_robot_controller.robot.robot_control_node",
        Robot_29999=_Chan, Robot_30001=_Chan, Robot_modbus=_Chan,
    ):
        node.connect_all_servers("192.168.1.101")

    assert node.robot_ip == "192.168.1.101"
    assert built, "channels were not rebuilt for the new address"
    assert all(args[0] == "192.168.1.101" for args in built)


def test_connect_keeps_channels_when_ip_is_unchanged():
    """With the same address, sockets are not rebuilt (avoids needless disconnects)."""
    node = make_connectable_node()
    node.robot_ip = "10.0.0.9"
    dash_before = node.robot_dash

    assert node.connect_all_servers("10.0.0.9") is True
    assert node.robot_dash is dash_before


# ---- connection watch: detecting a lost link and reconnecting automatically ---------------------------------------------
def _linked_node(fail_reads):
    """Connected node. If fail_reads is True, Modbus reads return None (robot not responding)."""
    node = make_connectable_node()
    assert node.connect_all_servers() is True
    for name in ("pub_robot_mode", "pub_control_method", "pub_op_mode", "pub_tcp_pose",
                 "pub_tcp_pose_zero", "pub_joint_position", "pub_scan_state", "pub_task_state",
                 "pub_home_flag", "pub_speed_scale", "pub_digital_in",
                 "pub_digital_out", "pub_probe_result", "pub_probe_poses",
                 "pub_scan_arc", "pub_alarm"):
        setattr(node, name, FakePublisher())
    node.alarm_mgr = type("M", (), {"process": lambda self, a: False})()
    node.robot_primary.get_data = lambda: None
    node.robot_primary.alarm_queue = type("Q", (), {"empty": lambda self: True})()
    node.robot_modbus.get_register = (lambda address: None) if fail_reads else (lambda address: 5)
    return node


def test_link_is_dropped_after_consecutive_read_failures():
    """If the robot does not respond, it stops publishing 'connected' and switches to disconnected."""
    node = _linked_node(fail_reads=True)
    for _ in range(node.LINK_FAIL_LIMIT - 1):
        node.update_robot_loop()
    assert node.connected is True, "one or two failures do not disconnect"
    node.update_robot_loop()
    assert node.connected is False
    assert node.pub_connected.messages[-1] is False
    assert node._want_connected is True, "the operator did not disconnect, so it reconnects"


def test_a_good_read_resets_the_failure_count():
    node = _linked_node(fail_reads=True)
    node.update_robot_loop()
    node.update_robot_loop()
    node.robot_modbus.get_register = lambda address: 5
    node.update_robot_loop()
    assert node._link_fails == 0 and node.connected is True


def test_auto_reconnect_only_when_the_operator_wants_it():
    node = _linked_node(fail_reads=True)
    for _ in range(node.LINK_FAIL_LIMIT):
        node.update_robot_loop()
    assert node.connected is False
    assert node._try_reconnect() is True and node.connected is True

    res = type("R", (), {})()
    node.cb_disconnect(None, res)
    assert node._want_connected is False and node.connected is False
    assert node._try_reconnect() is False and node.connected is False, "no reconnect after a disconnect"
    node.cb_connect(None, res)
    assert node._want_connected is True and node.connected is True
