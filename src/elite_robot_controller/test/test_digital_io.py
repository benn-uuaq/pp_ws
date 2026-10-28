"""Switch one digital output bit on and off.

The controller packs the digital IO as 16 bits in one register (user manual
15.2: 0 standard inputs / 2 standard outputs). So to change one output the current value must be
read, only that bit flipped and written back. The water spray valve and the marker will be attached
through this path, so it is important **not to write** when the value cannot be read — otherwise the other
outputs all switch off to 0 at once.
"""

from elite_robot_controller import register_map
from test_pose_publishing import FakePublisher, make_node


def _io_map():
    return {
        'scale': {'position_per_count': 0.1, 'rotation_per_count': 1.0},
        'read': {'digital_out': {'address': 2, 'count': 1, 'kind': 'raw'},
                 'digital_in': {'address': 0, 'count': 1, 'kind': 'raw'}},
        'write': {'digital_out': {'address': 2, 'count': 1, 'kind': 'raw'}},
    }


def _node(value=0b0000_0101):
    node = make_node({2: [value], 0: [0b0000_0010]}, _io_map())
    node.connected = True
    node.pub_digital_out = FakePublisher()
    return node


def test_shipped_map_has_the_documented_io_addresses():
    registers = register_map.load()
    assert registers.read_entry('digital_in').address == 0
    assert registers.read_entry('digital_out').address == 2
    assert registers.write_entry('digital_out').address == 2


def test_turning_one_output_on_keeps_the_others():
    node = _node(0b0000_0101)

    ok, message = node.set_digital_out(1, True)

    assert ok, message
    assert node.robot_modbus.writes == [(2, 0b0000_0111)]
    assert node.pub_digital_out.messages == [0b0000_0111]


def test_turning_one_output_off_keeps_the_others():
    node = _node(0b0000_0101)

    assert node.set_digital_out(0, False)[0]

    assert node.robot_modbus.writes == [(2, 0b0000_0100)]


def test_high_bit_is_written_as_a_signed_value():
    """The driver reads the written value back as signed 16 bit to confirm, so send it matched."""
    node = _node(0)

    assert node.set_digital_out(15, True)[0]

    assert node.robot_modbus.writes == [(2, -32768)]


def test_nothing_is_written_when_the_current_value_is_unknown():
    node = _node()
    node.robot_modbus.registers[2] = []

    ok, message = node.set_digital_out(3, True)

    assert not ok
    assert node.robot_modbus.writes == []
    assert 'could not read' in message


def test_out_of_range_and_disconnected_are_refused():
    node = _node()
    assert not node.set_digital_out(16, True)[0]
    assert not node.set_digital_out(-1, True)[0]
    node.connected = False
    assert not node.set_digital_out(0, True)[0]
    assert node.robot_modbus.writes == []


def test_topic_carries_index_and_value():
    node = _node(0)

    class Msg:
        data = [2, 1]

    node.cb_digital_out(Msg())

    assert node.robot_modbus.writes == [(2, 0b0000_0100)]

    class Short:
        data = [2]

    node.cb_digital_out(Short())
    assert node.robot_modbus.writes == [(2, 0b0000_0100)]
    assert node.get_logger().warnings


def test_mark_go_is_written_to_its_register():
    """Marking done signal (278) — the robot waits at the marking spot until this becomes 1."""
    node = make_node({}, {
        'scale': {'position_per_count': 0.1, 'rotation_per_count': 1.0},
        'read': {},
        'write': {'mark_go': {'address': 278, 'count': 1}},
    })

    class Msg:
        data = 1

    node.cb_mark_go(Msg())

    assert node.robot_modbus.writes == [(278, 1)]
    assert register_map.load().write_entry('mark_go').address == 278
    assert register_map.load().read_entry('scan_arc').address == 286
    # 288/289 (press state / correction) are read as well (2026-10-19).
    assert register_map.load().read_entry('scan_arc').count == 4
