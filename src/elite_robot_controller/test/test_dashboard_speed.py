"""Dashboard (29999) speed ratio command format.

Command table (robot_task/29999 dashboard command.png): `speed -v 50` = speed 50 %
(range 2 ~ 100 %). The previously sent `speed -set N` is not in the table, so
on the real robot only a reply came back and the ratio (register 17) did not change.
"""

from elite_robot_controller.robot.robot_driver import Robot_29999


def _sent(percent):
    dash = Robot_29999.__new__(Robot_29999)
    sent = []
    dash.send_command_29999 = lambda command: sent.append(command) or "ok"
    dash.robot_set_speed(percent)
    return sent


def test_speed_ratio_uses_the_documented_command():
    assert _sent(50) == ["speed -v 50"]


def test_speed_ratio_is_kept_in_range():
    assert _sent(0) == ["speed -v 2"]
    assert _sent(150) == ["speed -v 100"]
