"""AlarmManager regression tests.

Checks that duplicate suppression is a 'short time window', not a 'permanent block'. Previously
a (code, sub, msg) key stayed in the dict forever once seen, so when the same
alarm happened again later (e.g. an inverse kinematics failure repeating for the same cause)
it was never reported to RCS again while the process ran.
"""

from unittest.mock import patch

from elite_robot_controller.robot.robot_driver import AlarmData, AlarmManager


def test_same_alarm_reported_again_after_dedup_window():
    mgr = AlarmManager(dedup_window=2.0)
    alarm = AlarmData(msg="PROBE_R: NO IK SOLUTION")

    with patch("elite_robot_controller.robot.robot_driver.time.time", return_value=1000.0):
        assert mgr.process(alarm) is True

    # within the window the same alarm must be suppressed (e.g. split packets).
    with patch("elite_robot_controller.robot.robot_driver.time.time", return_value=1000.5):
        assert mgr.process(alarm) is False

    # after the window the same alarm must be reported again — with the old bug
    # this was False forever.
    with patch("elite_robot_controller.robot.robot_driver.time.time", return_value=1003.0):
        assert mgr.process(alarm) is True


def test_different_alarms_are_never_deduped_against_each_other():
    mgr = AlarmManager(dedup_window=2.0)
    with patch("elite_robot_controller.robot.robot_driver.time.time", return_value=1000.0):
        assert mgr.process(AlarmData(msg="A")) is True
        assert mgr.process(AlarmData(msg="B")) is True
