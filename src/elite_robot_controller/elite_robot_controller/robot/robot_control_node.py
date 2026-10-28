import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Int32, Int32MultiArray, Float32MultiArray
from std_srvs.srv import Trigger

from elite_robot_controller import register_map
# makes the internal robot.py modules importable
from elite_robot_controller.robot.robot_driver import (
    AlarmManager, Robot_29999, Robot_30001, Robot_modbus,
)


class RobotControlNode(Node):
    # register addresses are managed only in config/modbus_registers.json.
    # entries without an address yet refuse requests so nothing is written to a wrong register.

    def __init__(self):
        super().__init__('robot_control_node')

        self.declare_parameter('robot_ip', '192.168.227.134')
        self.declare_parameter('register_map', '')
        # 502 needs privileges on Linux, so it is a parameter to allow testing with the simulator.
        self.declare_parameter('modbus_port', 502)
        # the jog speed uses register 307 (speed ratio %) directly. The units differ, so
        # the values below are for 100 % and are scaled down by the ratio.
        #   speedj  qd [rad/s], a [rad/s^2]   (script manual 3.1.26)
        #   speedl  xd [m/s],   a [m/s^2]     (script manual 3.1.27)
        self.declare_parameter('jog_joint_speed_max', 0.50)    # rad/s
        self.declare_parameter('jog_tcp_speed_max', 0.10)      # m/s
        self.declare_parameter('jog_tcp_rot_speed_max', 0.50)  # rad/s
        self.declare_parameter('jog_accel_max', 1.00)          # speedj rad/s^2
        # TCP jog (speedl) acceleration [m/s^2]. Operating standard for linear moves: 400 mm/s^2.
        self.declare_parameter('jog_tcp_accel_max', 0.40)
        # t of speedj/speedl. After this time the robot stops by itself.
        # The UI sends the command exactly once when the button is pressed (before 2026-08-31 it
        # repeated at short intervals while held, and every time the robot ran a new script and
        # toggled STOPPED→RUNNING, so the motion jerked). The actual
        # stop is the 29999 stop sent on release, so this value is only a safety timeout "in case that
        # stop never arrives" — generous enough not to interfere with normal jogging.
        #
        self.declare_parameter('jog_hold_time', 3.0)
        # home tolerance [rad]. Same value as home_tol_j in the task.
        self.declare_parameter('home_tol_j', 0.02)
        # tolerance [rad] for deciding the robot has **left home** once it was home (2026-10-19).
        # Larger than the entry tolerance (home_tol_j) so home holds while the vehicle / lift moves and the arm sways a little
        # — about 3 degrees per joint. Moving more than this
        # counts as deployed and clears 276 (RCS stops moving and raises E2003).
        self.declare_parameter('home_leave_tol_j', 0.05)
        # interval [s] for re-reading the robot's Home_joint via 29999.
        self.declare_parameter('home_refresh_s', 60.0)
        robot_ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        map_path = self.get_parameter('register_map').get_parameter_value().string_value
        self.registers = register_map.load(map_path or None)

        self.robot_ip = robot_ip
        self.robot_dash = Robot_29999(robot_ip, 29999)
        self.robot_primary = Robot_30001(robot_ip, 30001)
        modbus_port = self.get_parameter('modbus_port').get_parameter_value().integer_value
        self.robot_modbus = Robot_modbus(robot_ip, modbus_port)
        self.alarm_mgr = AlarmManager()
        self.connected = False
        # connection watch. If Modbus reads from the robot fail for LINK_FAIL_LIMIT consecutive cycles,
        # the link counts as lost and connected goes down. Unless the operator pressed 'disconnect'
        # (_want_connected), a background thread reconnects every RECONNECT_S.
        self._want_connected = True
        self._link_fails = 0
        self._connect_lock = threading.Lock()
        # robot's own speed ratio [%]. The jog speed follows it too.
        self.speed_ratio = 100
        # last register values read (publish_code) / joints [mrad].
        self._last_codes = {}
        self._last_joint_mrad = None
        # robot variables read via 29999 (Home_joint / Standby_joint) and when they were read.
        # Used to decide 276 directly when the task is not running, and to plan the home move path.
        self._joint_variables = {}

        # published topic definitions
        self.pub_robot_mode = self.create_publisher(Int32, 'robot/status/robot_mode', 10)
        self.pub_control_method = self.create_publisher(Int32, 'robot/status/control_method', 10)
        self.pub_op_mode = self.create_publisher(Int32, 'robot/status/operation_mode', 10)
        self.pub_tcp_pose = self.create_publisher(Float32MultiArray, 'robot/status/tcp_pose', 10)
        self.pub_tcp_pose_zero = self.create_publisher(
            Float32MultiArray, 'robot/status/tcp_pose_zero', 10)
        self.pub_joint_position = self.create_publisher(
            Float32MultiArray, 'robot/status/joint_position', 10)
        self.pub_alarm = self.create_publisher(String, 'robot/status/alarms', 10)
        self.pub_connected = self.create_publisher(Bool, 'robot/status/connected', 10)
        # scan progress written by the robot task (290~298). The operating UI's grid traversal
        # uses it to decide that a cell is done (state==5 && finished==1).
        self.pub_scan_state = self.create_publisher(Int32MultiArray, 'robot/status/scan_state', 10)
        # position on the arc during a scan (286/287). Used to tell ERUT the actual inspection coordinates.
        self.pub_scan_arc = self.create_publisher(
            Int32MultiArray, 'robot/status/scan_arc', 10)
        # task state from the robot controller (1 running, 2 paused, 3 stopped).
        # Asking via 29999 means sending a command every time, so it is read continuously via Modbus.
        self.pub_task_state = self.create_publisher(Int32, 'robot/status/task_state', 10)
        # whether the robot is at home (register 276). The task writes it every control cycle,
        # and this node manages it when the task is not running (jog 0 / home reached 1).
        # RCS looks at this value together with the task state before moving the vehicle / lift.
        self.pub_home_flag = self.create_publisher(Int32, 'robot/status/at_home', 10)
        # speed ratio [%] actually used by the robot. Changes on the pendant show up here too.
        self.pub_speed_scale = self.create_publisher(Int32, 'robot/status/speed_scale', 10)
        # digital I/O bit fields (register 0 = standard DI, 2 = standard DO).
        # One register has 16 bits; bit 0 upwards maps to the I/O number.
        self.pub_digital_in = self.create_publisher(Int32, 'robot/status/digital_in', 10)
        self.pub_digital_out = self.create_publisher(Int32, 'robot/status/digital_out', 10)
        # 3-point measurement result (300~305) and contact poses (330~355). ERUT calibration
        # uses them to measure the real wall radius and compute the error.
        self.pub_probe_result = self.create_publisher(
            Int32MultiArray, 'robot/status/probe_result', 10)
        self.pub_probe_poses = self.create_publisher(
            Int32MultiArray, 'robot/status/probe_poses', 10)

        # connect / disconnect are exposed as services too, so the operating UI can handle them.
        self.create_service(Trigger, 'robot/dashboard/connect', self.cb_connect)
        self.create_service(Trigger, 'robot/dashboard/disconnect', self.cb_disconnect)

        # dashboard command service mapping
        self.create_service(Trigger, 'robot/dashboard/robot_mode', self.cb_dash_mode)
        self.create_service(Trigger, 'robot/dashboard/status', self.cb_dash_status)
        self.create_service(Trigger, 'robot/dashboard/power_on', self.cb_dash_power_on)
        self.create_service(Trigger, 'robot/dashboard/power_off', self.cb_dash_power_off)
        self.create_service(Trigger, 'robot/dashboard/brake_release', self.cb_dash_brake)
        self.create_service(Trigger, 'robot/dashboard/play', self.cb_dash_play)
        self.create_service(Trigger, 'robot/dashboard/remote_control_on',
                            self.cb_dash_remote_on)
        self.create_service(Trigger, 'robot/dashboard/pause', self.cb_dash_pause)
        self.create_service(Trigger, 'robot/dashboard/stop', self.cb_dash_stop)
        # the task is fixed on the pendant / robot side, so the operating UI does not choose it —
        # it only asks 29999 "task -s" what is loaded and shows it.
        self.create_service(Trigger, 'robot/dashboard/task_status', self.cb_dash_task_status)

        # home move is a one-shot command without a value, so it uses Trigger. Saving reference
        # poses sends values from the UI, so they come through the topics below.
        self.create_service(Trigger, 'robot/command/move_home', self.cb_move_home)

        # commands with a value come in as topics.
        # work speed is mm/s, speed ratio is 2~100 [%]; both are written to registers.
        self.create_subscription(Int32, 'robot/command/linear_speed', self.cb_linear_speed, 10)
        self.create_subscription(Int32, 'robot/command/speed_ratio', self.cb_speed_ratio, 10)
        # jogging is done with 30001 scripts, not registers.
        # the sign of the value is the direction, the absolute value the axis number (1~6), 0 = stop.
        self.create_subscription(Int32, 'robot/command/jog_joint', self.cb_jog_joint, 10)
        self.create_subscription(Int32, 'robot/command/jog_tcp', self.cb_jog_tcp, 10)

        # reference poses are written to 6 registers at once. Home (322~327) and the scan standby
        # pose (310~315, formerly home) are saved separately (2026-10-16).
        self.create_subscription(
            Float32MultiArray, 'robot/command/home_joint', self.cb_home_joint, 10)
        self.create_subscription(
            Float32MultiArray, 'robot/command/standby_joint', self.cb_standby_joint, 10)
        self.create_subscription(
            Float32MultiArray, 'robot/command/start_pose', self.cb_start_pose, 10)

        # work area: [width, height, scanner height, overlap] mm. The grid block of the MQTT job_cmd
        # arrives here through the UI. Written to 256~259; 266 (param_src) must be set
        # to 1 for the task to read these values.
        self.create_subscription(
            Float32MultiArray, 'robot/command/work_area', self.cb_work_area, 10)
        # scan start permission (267). Releases the robot waiting at the origin.
        self.create_subscription(Int32, 'robot/command/scan_go', self.cb_scan_go, 10)
        # vehicle locked confirmation (309). 1 when RCS confirms the vehicle has stopped, the outriggers are locked
        # and the lift has stopped; 0 if anything moves. The robot
        # task only moves when this is 1 (dus5_init / dus5_goto_zero).
        self.create_subscription(
            Int32, 'robot/command/vehicle_ready', self.cb_vehicle_ready, 10)
        # marking spot [u, v] mm (268~269). Written before starting the marking task.
        self.create_subscription(
            Float32MultiArray, 'robot/command/mark_target', self.cb_mark_target, 10)
        # marking done (278). The robot waits at the marking spot (290 = 11) until this becomes 1.
        self.create_subscription(Int32, 'robot/command/mark_go', self.cb_mark_go, 10)
        # switch one digital output: [number, value]. Number is standard DO 0~15,
        # value is 0 or 1. Used to check devices attached to robot outputs (water spray valve,
        # marker) by hand from the screen.
        self.create_subscription(
            Int32MultiArray, 'robot/command/digital_out', self.cb_digital_out, 10)
        # swap tasks. Marking is a different task from scanning, so it is loaded via 29999.
        # The paths are the real locations inside the controller and must be checked on site once.
        # The reference version is v5 (servoj scan). RCS overwrites it directly with the chosen version.
        self.declare_parameter('mark_task_path', 'Dusan/dusan_v5/dusan_v5_mark.task')
        self.declare_parameter('scan_task_path', 'Dusan/dusan_v5/dusan_v5.task')
        self.create_service(Trigger, 'robot/dashboard/load_mark_task', self.cb_load_mark_task)
        self.create_service(Trigger, 'robot/dashboard/load_scan_task', self.cb_load_scan_task)

        # update Modbus data and collect 30001 alarms at 10 Hz
        self.timer = self.create_timer(0.1, self.update_robot_loop)
        self.get_logger().info("[DEBUG]] ELITE Robot control ROS2 node is active.")

        # try to connect once at start. Even if it fails the node must stay alive so the operating UI
        # can use the connect button.
        if not self.connect_all_servers(robot_ip):
            self.get_logger().warn(
                f"[WARN] Could not connect to the robot — retrying every {self.RECONNECT_S:g} s."
            )
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop, name="robot-reconnect", daemon=True)
        self._reconnect_thread.start()

    #: Modbus reads failing this many times in a row count as a lost link (cycle 0.1 s + read wait).
    LINK_FAIL_LIMIT = 3
    #: interval [s] for reconnecting after the link is lost.
    RECONNECT_S = 2.0

    def _reconnect_loop(self):
        """While disconnected and the operator wants a connection, reconnect periodically.

        Connecting from the timer callback (update_robot_loop) would freeze every other service / topic of the node
        during the attempt (a few seconds per socket), so it runs separately.
        """
        while rclpy.ok():
            time.sleep(self.RECONNECT_S)
            self._try_reconnect()

    def _try_reconnect(self):
        """If disconnected and a connection is wanted, try once. True if connected."""
        if self.connected or not self._want_connected:
            return False
        if not self._connect_lock.acquire(blocking=False):
            return False            # a connect pressed by the operator is already in progress
        try:
            if self.connected or not self._want_connected:
                return False
            self.disconnect_all_servers(quiet=True)
            if self._connect_unlocked():
                self.get_logger().info(f"[connect] {self.robot_ip} reconnected")
                return True
            return False
        finally:
            self._connect_lock.release()

    def _param_str(self, name, fallback):
        """The connection proceeds even if the parameter cannot be read (keeps the existing address)."""
        try:
            return self.get_parameter(name).get_parameter_value().string_value or fallback
        except Exception:
            return fallback

    def _param_int(self, name, fallback):
        try:
            return self.get_parameter(name).get_parameter_value().integer_value or fallback
        except Exception:
            return fallback

    def connect_all_servers(self, robot_ip=None):
        with self._connect_lock:
            return self._connect_unlocked(robot_ip)

    def _connect_unlocked(self, robot_ip=None):
        """Connect all three channels. If any fails, it does not count as connected.

        The `robot_ip` parameter is re-read on every connect — so that when the operating UI changes the
        address on screen, sets this parameter and calls connect, the node connects **to that address**.
        Previously the sockets were created with the address read at node start
        and the argument was ignored, so however the IP was changed in the UI the node kept
        connecting to the old address (the cause of the screen and the real connection disagreeing).
        """
        if robot_ip is None:
            robot_ip = self._param_str('robot_ip', self.robot_ip)
        if robot_ip and robot_ip != self.robot_ip:
            self.get_logger().info(f"[connect] address changed: {self.robot_ip} -> {robot_ip}")
            self.disconnect_all_servers()
            modbus_port = self._param_int('modbus_port', 502)
            self.robot_ip = robot_ip
            self.robot_dash = Robot_29999(robot_ip, 29999)
            self.robot_primary = Robot_30001(robot_ip, 30001)
            self.robot_modbus = Robot_modbus(robot_ip, modbus_port)
        try:
            d_ok = self.robot_dash.connect_29999()
            p_ok = self.robot_primary.connect_30001()
            m_ok = self.robot_modbus.connect()
            self.connected = bool(d_ok and p_ok and m_ok)
            self._link_fails = 0
        except Exception as e:
            self.get_logger().error(f"[ERROR] Exception while connecting the sockets: {e}")
            self.connected = False
        self.publish_connected()
        return self.connected

    def disconnect_all_servers(self, quiet=False):
        """Clean up the three channels. A failure on one does not stop cleaning up the others."""
        for close in (self.robot_dash.disconnect_29999,
                      self.robot_primary.disconnect_30001,
                      self.robot_modbus.disconnect):
            try:
                close()
            except Exception as e:
                if not quiet:
                    self.get_logger().warn(f"[WARN] Exception while disconnecting: {e}")
        self.connected = False
        self.publish_connected()
        return True

    def publish_connected(self):
        """Report whether the three channels are connected. The operating UI shows the state from this."""
        self.pub_connected.publish(Bool(data=self.connected))

    def cb_connect(self, req, res):
        # the operator wants a connection — even if it fails now, the background keeps reconnecting.
        self._want_connected = True
        ok = self.connect_all_servers()
        res.success = ok
        res.message = (
            f"[connect] {self.robot_ip} connected" if ok
            else f"[connect] {self.robot_ip} connection failed"
        )
        if not ok:
            self.get_logger().warn(res.message)
        return res

    def cb_disconnect(self, req, res):
        # the operator disconnected — do not reconnect automatically.
        self._want_connected = False
        self.disconnect_all_servers()
        res.success = True
        res.message = "[disconnect] Disconnected."
        return res

    def update_robot_loop(self):
        # publish every cycle. If only changes were sent, a UI that subscribes later
        # (or restarts) would never get values that already went by.
        self.publish_connected()
        # touching the sockets before connecting raises an exception.
        if not self.connected:
            return

        # the first read is the liveness check. If it fails, the rest of this cycle's reads
        # are skipped — reading all ten while disconnected waits for the timeout (1 s) each time,
        # freezing the node for seconds while it keeps publishing 'connected'.
        if not self.publish_code('robot_mode', self.pub_robot_mode):
            self._link_fails += 1
            if self._link_fails >= self.LINK_FAIL_LIMIT:
                self._link_lost()
            return
        self._link_fails = 0
        self.publish_code('control_method', self.pub_control_method)
        self.publish_code('operation_mode', self.pub_op_mode)
        self.publish_pose('tcp_absolute', self.pub_tcp_pose)
        self.publish_pose('tcp_zero_relative', self.pub_tcp_pose_zero)
        self.publish_pose('joint_position', self.pub_joint_position)
        self.publish_raw('scan_state', self.pub_scan_state)
        self.publish_raw('scan_arc', self.pub_scan_arc)
        self.publish_code('task_state', self.pub_task_state)
        self._watch_home()
        self.publish_code('home_flag', self.pub_home_flag)
        self.publish_code('speed_scale', self.pub_speed_scale)
        self.publish_code('digital_in', self.pub_digital_in)
        self.publish_code('digital_out', self.pub_digital_out)
        self.publish_raw('probe_result', self.pub_probe_result)
        self.publish_raw('probe_poses', self.pub_probe_poses)

        # process the asynchronous background real-time alarm stream of port 30001
        self.robot_primary.get_data()
        while not self.robot_primary.alarm_queue.empty():
            alarm = self.robot_primary.alarm_queue.get()
            if self.alarm_mgr.process(alarm):
                alarm_msg = String()
                alarm_msg.data = (f"[ALARM] {alarm.msg}" if alarm.msg
                                  else f"[ALARM CODE] E{alarm.code} S{alarm.sub}")
                self.pub_alarm.publish(alarm_msg)

    def _link_lost(self):
        """The robot stopped responding. Report the lost link and clean up the sockets (the background thread reconnects)."""
        self.get_logger().warn(
            f"[connect] {self.robot_ip} not responding ({self._link_fails} times in a row) — link lost. "
            f"Reconnecting every {self.RECONNECT_S:g} s.")
        self._link_fails = 0
        self.disconnect_all_servers(quiet=True)

    def publish_code(self, name, publisher):
        """Read one register and publish it as is. True if read."""
        entry = self.registers.read_entry(name)
        if not entry.available:
            return True
        try:
            value = self.robot_modbus.get_register(entry.address)
        except Exception:
            value = None
        if value is None:
            return False
        self.__dict__.setdefault('_last_codes', {})[name] = value
        publisher.publish(Int32(data=value))
        return True

    def publish_pose(self, name, publisher):
        """Read the 6 pose registers and publish them as [X, Y, Z, Rx, Ry, Rz].

        get_all_registers already converts to signed 16 bit, so only the unit
        conversion happens here. X, Y, Z are mm, Rx, Ry, Rz are mrad.
        """
        entry = self.registers.read_entry(name)
        if not entry.available:
            return

        regs = self.robot_modbus.get_all_registers(entry.address, entry.count)
        if not regs or len(regs) != entry.count:
            return

        if name == 'joint_position':
            self._last_joint_mrad = list(regs)
        scales = self.registers.scales_for(entry)
        pose_msg = Float32MultiArray()
        pose_msg.data = [value * scale for value, scale in zip(regs, scales)]
        publisher.publish(pose_msg)

    def publish_raw(self, name, publisher):
        """Publish several registers as integers without conversion.

        Used for unitless values such as scan progress (state codes, counts, flags).
        A conversion could introduce floating-point errors that break == comparisons.
        """
        entry = self.registers.read_entry(name)
        if not entry.available:
            return

        regs = self.robot_modbus.get_all_registers(entry.address, entry.count)
        if not regs or len(regs) != entry.count:
            return

        publisher.publish(Int32MultiArray(data=[int(value) for value in regs]))

    def write_register(self, name, value):
        """Write a value to a write register. Returns (success, message).

        Entries without an address are not attempted. Writing to a wrong
        register could move the robot unexpectedly.
        """
        entry = self.registers.write_entry(name)
        if not entry.available:
            return False, f"[{name}] Modbus address is not configured."
        try:
            ok = self.robot_modbus.set_register(entry.address, value)
        except Exception as exc:
            return False, f"[{name}] register write failed: {exc}"
        if not ok:
            return False, f"[{name}] could not confirm the write to register {entry.address}."
        return True, f"[{name}] register {entry.address} <- {value}"

    def _write_service(self, name, value, response):
        """Fill a Trigger service response with the write result."""
        success, message = self.write_register(name, value)
        if not success:
            self.get_logger().warn(message)
        response.success = success
        response.message = message
        return response

    def _write_topic(self, name, msg):
        """Write a value received on a topic to a register and log only failures."""
        success, message = self.write_register(name, int(msg.data))
        if not success:
            self.get_logger().warn(message)

    def cb_linear_speed(self, msg):
        self._write_topic('linear_speed', msg)

    def cb_speed_ratio(self, msg):
        """Change the robot's overall motion speed ratio [%]. 2~100.

        Applying it in real time via 29999 is the main path. Register 307 is the value the task
        reads at start, so it is written as well (kept after a task restart).
        The jog speed is scaled by this value too.
        """
        self.speed_ratio = max(2, min(100, int(msg.data)))
        # real-time: applies immediately even to a running motion.
        result = self.robot_dash.robot_set_speed(self.speed_ratio)
        if result is None:
            self.get_logger().warn(
                f"[speed] failed to send speed ratio {self.speed_ratio} %")
        elif any(word in str(result).lower() for word in self._DASH_REFUSALS):
            # a reply came but it was refused (wrong command format, local mode, etc.).
            self.get_logger().warn(
                f"[speed] the controller refused speed ratio {self.speed_ratio} %: {result}")
        self._write_topic('speed_ratio', Int32(data=self.speed_ratio))

    def _jog_scale(self):
        """Convert the speed ratio (2~100 %) to a factor."""
        return self.speed_ratio / 100.0

    @staticmethod
    def _round6(values):
        """Remove floating-point noise, because the values go out as script strings."""
        return [round(float(v), 6) for v in values]

    def write_pose(self, name, values):
        """Write 6 pose values to consecutive registers. The unit conversion is the reverse of reading."""
        entry = self.registers.write_entry(name)
        if not entry.available:
            return False, f"[{name}] Modbus address is not configured."
        if len(values) != entry.count:
            return False, f"[{name}] expected {entry.count} values, got {len(values)}"

        scales = self.registers.scales_for(entry)
        for offset, (value, scale) in enumerate(zip(values, scales)):
            raw = int(round(float(value) / scale))
            ok = self.robot_modbus.set_register(entry.address + offset, raw)
            if not ok:
                return False, f"[{name}] register {entry.address + offset} write failed"
        return True, f"[{name}] registers {entry.address}~{entry.address + entry.count - 1} updated"

    def _write_pose_topic(self, name, msg, flag_name='pose_src'):
        """Write N values to registers and set a flag register to 1 so the task reads them.

        standby_joint/start_pose use pose_src (308), home_joint uses home_src (328),
        work_area uses param_src (266). Only the flag name differs; the behaviour is the same.
        """
        success, message = self.write_pose(name, list(msg.data))
        if not success:
            self.get_logger().warn(message)
            return
        flag_entry = self.registers.write_entry(flag_name)
        if flag_entry.available:
            self.robot_modbus.set_register(flag_entry.address, 1)
        else:
            self.get_logger().warn(f"[{name}] {flag_name} address is not configured.")

    def cb_home_joint(self, msg):
        """Home joints -> 322~327, home_src (328) = 1. The task reads it as Home_joint."""
        self._write_pose_topic('home_joint', msg, flag_name='home_src')

    def cb_standby_joint(self, msg):
        """Scan standby joints -> 310~315, pose_src (308) = 1 (the former home slot)."""
        self._write_pose_topic('standby_joint', msg)

    def cb_start_pose(self, msg):
        self._write_pose_topic('start_pose', msg)

    def cb_work_area(self, msg):
        # [width, height, scanner height, overlap] mm -> 256~259, on success 266=1.
        self._write_pose_topic('work_area', msg, flag_name='param_src')

    def cb_mark_target(self, msg):
        """Write the marking spot [u, v] (mm) to 268~269."""
        success, message = self.write_pose('mark_target', list(msg.data))
        if not success:
            self.get_logger().warn(message)

    def _load_task(self, param_name, response):
        """Load a task with 29999 `task -p <path>`.

        The path is a parameter (mark_task_path / scan_task_path). Without a sensor, start with
        scan_task_path set to dusan_v5_nosensor_seq.task and mark_task_path set to
        dusan_v5_nosensor_mark.task.
        """
        path = self._param_str(param_name, '')
        if not path:
            response.success = False
            response.message = f"[task -p] the {param_name} parameter is empty."
            return response
        return self._execute_dash_cmd(
            lambda: self.robot_dash.send_command_29999(f"task -p {path}"),
            f"task -p {path}", response)

    def cb_load_mark_task(self, req, res):
        return self._load_task('mark_task_path', res)

    def cb_load_scan_task(self, req, res):
        return self._load_task('scan_task_path', res)

    def cb_scan_go(self, msg):
        """Write the scan start permission (267).

        When the robot reaches the origin it sets state (290) to 7 and waits until this value
        becomes 1. The robot resets it to 0 itself when passing, so it does not need to be
        cleared here.
        """
        self._write_topic('scan_go', msg)

    # ------------------------------------------------------------ digital outputs
    #: number of bits in one digital output register (manual 15.2 register mapping).
    DIGITAL_OUT_BITS = 16

    def cb_digital_out(self, msg):
        """Switch one digital output with [number, value]."""
        data = list(msg.data)
        if len(data) < 2:
            self.get_logger().warn(f"[digital_out] two values [number, value] are required: {data}")
            return
        ok, message = self.set_digital_out(int(data[0]), bool(data[1]))
        if not ok:
            self.get_logger().warn(message)

    def set_digital_out(self, index, on):
        """Change a single standard digital output bit. (success, message).

        The outputs are packed as 16 bits in one register (default 2), so a single one cannot
        be touched. So the current value is read, only that bit is changed, and it is written
        back — if it cannot be read, writing could push the other outputs to 0, so it is
        not written.
        """
        if not 0 <= int(index) < self.DIGITAL_OUT_BITS:
            return False, f"[digital_out] output number out of range: {index}"
        if not self.connected:
            return False, "[digital_out] not connected to the robot."
        entry = self.registers.read_entry('digital_out')
        if not entry.available:
            return False, "[digital_out] Modbus address is not configured."
        try:
            current = self.robot_modbus.get_register(entry.address)
        except Exception as exc:
            current = None
            self.get_logger().debug(f"[digital_out] read failed: {exc}")
        if current is None:
            return False, "[digital_out] could not read the current output value, so nothing was written."
        mask = int(current) & 0xFFFF
        bit = 1 << int(index)
        mask = (mask | bit) if on else (mask & ~bit)
        # the driver reads the written value back as signed 16 bit to confirm.
        value = mask - 0x10000 if mask > 0x7FFF else mask
        ok, message = self.write_register('digital_out', value)
        if ok:
            self.pub_digital_out.publish(Int32(data=value))
            message = f"[digital_out] DO{int(index)} <- {'ON' if on else 'OFF'}"
        return ok, message

    # ---------------------------------------------------------------- jog
    def _jog_vector(self, code, count=6):
        """Convert a code whose sign is the direction and absolute value the axis number (1~6) to a velocity vector."""
        axis = abs(int(code)) - 1
        if axis < 0 or axis >= count:
            return None
        vector = [0.0] * count
        vector[axis] = 1.0 if code > 0 else -1.0
        return vector

    def _jog_stop(self):
        """Stop immediately with 29999 stop. A jog must move only while pressed."""
        self.robot_dash.robot_stop()

    def cb_mark_go(self, msg):
        """Write the marking done signal (278). The robot clears it to 0 itself when passing."""
        self._write_topic('mark_go', msg)

    def cb_vehicle_ready(self, msg):
        """Pass the vehicle locked confirmation to register 309 as is (1 locked / 0 not)."""
        self._write_topic('vehicle_ready', Int32(data=1 if int(msg.data) else 0))

    # ---------------------------------------------------------------- home check
    #: task states (500) in which the task writes 276 itself (running).
    _TASK_RUNNING = 1

    def _saved_joints(self, flag_name, entry_name):
        """Saved joints [rad] if the flag is 1 and values are present, otherwise None.

        If all registers are 0, nothing was ever saved — using them as is would go to a wrong pose
        like movej([0, 0, 0, 0, 0, 0]).
        """
        flag = self.registers.read_entry(flag_name)
        if not flag.available:
            flag = self.registers.write_entry(flag_name)
        entry = self.registers.write_entry(entry_name)
        if not flag.available or not entry.available:
            return None
        if self.robot_modbus.get_register(flag.address) != 1:
            return None
        regs = self.robot_modbus.get_all_registers(entry.address, entry.count)
        if regs and len(regs) == 6 and any(regs):
            return [v / 1000.0 for v in regs]
        return None

    def _robot_joint_variable(self, name):
        """Re-read the robot variables (pendant values) every home_refresh_s. None if unknown."""
        cache = getattr(self, '_joint_variables', None)
        if cache is None:
            cache = self._joint_variables = {}
        now = time.monotonic()
        refresh = self.get_parameter('home_refresh_s').value
        value, read_at = cache.get(name, (None, None))
        if read_at is None or now - read_at > refresh:
            fresh = self.robot_dash.get_variable(name)
            if isinstance(fresh, list) and len(fresh) == 6:
                value = [float(v) for v in fresh]
            cache[name] = (value, now)
        return value

    def _home_joint_target(self):
        """Home joints [rad]. If 328 is 1, 322~327, otherwise the robot's Home_joint.

        Same rule as the task (dus5_init) (since 2026-10-16, 310~315 is not home but
        the scan standby pose). The value read via 29999 is re-read every home_refresh_s.
        """
        saved = self._saved_joints('home_src', 'home_joint')
        if saved is not None:
            return saved
        return self._robot_joint_variable('Home_joint')

    def _standby_joint_target(self):
        """Scan standby joints [rad]. If 308 is 1, 310~315, otherwise Standby_joint.

        None if the robot has no Standby_joint variable yet (task not loaded).
        """
        saved = self._saved_joints('pose_src', 'standby_joint')
        if saved is not None:
            return saved
        return self._robot_joint_variable('Standby_joint')

    def _watch_home(self):
        """While the task is not running, decide the home flag (276) from the actual joints and write it.

        While the task runs, the task's publishing thread writes 276 every control cycle.
        When it does not run nobody writes it, so right after RCS starts (register 0) or after
        moving the robot by hand with the pendant the value disagreed with reality — the robot was home but ERUT
        saw deployed and could not get calibration (if-0.5 evt/home). The joints are compared with
        the home joints (same tolerance as the task) and the value is rewritten only when it differs.
        If the home joints are unknown, nothing is touched.
        """
        if getattr(self, '_last_codes', {}).get('task_state') == self._TASK_RUNNING:
            return
        joints = getattr(self, '_last_joint_mrad', None)
        if not joints or len(joints) != 6:
            return
        target = self._home_joint_target()
        if not target:
            return
        entry = self.registers.read_entry('home_flag')
        if not entry.available:
            return
        current = self.robot_modbus.get_register(entry.address)
        # entering uses home_tol_j; once home, up to home_leave_tol_j still counts as home —
        # so the flag does not flicker from sway while moving (hysteresis).
        tol = float(self.get_parameter('home_tol_j').value)
        if current == 1:
            tol = max(tol, float(self.get_parameter('home_leave_tol_j').value))
        at_home = all(abs(q / 1000.0 - h) <= tol for q, h in zip(joints, target))
        if current is not None and bool(current) != at_home:
            self.write_register('home_flag', 1 if at_home else 0)

    def _leaving_home(self):
        """Clear the home flag right before moving the robot out of home.

        While the task runs, its publishing thread updates 276 every control cycle,
        but jog and home moves run **outside** the task as 30001 scripts,
        so nobody updates the value during them. If left as is,
        the flag stays 1 after jogging out of home, and RCS thinks the robot is at
        home and moves the vehicle.
        """
        if self.connected:
            self.write_register('home_flag', 0)

    def cb_jog_joint(self, msg):
        if not self.connected:
            return
        if int(msg.data) == 0:
            self._jog_stop()
            return
        unit = self._jog_vector(msg.data)
        if unit is None:
            self.get_logger().warn(f"[jog_joint] axis number out of range: {msg.data}")
            return
        scale = self._jog_scale()
        speed = self.get_parameter('jog_joint_speed_max').value * scale
        accel = self.get_parameter('jog_accel_max').value * scale
        qd = self._round6([value * speed for value in unit])
        hold = self.get_parameter('jog_hold_time').value
        self._leaving_home()
        self.robot_primary.send_script(f"speedj({qd}, {round(accel, 6)}, {hold})")

    def cb_jog_tcp(self, msg):
        if not self.connected:
            return
        if int(msg.data) == 0:
            self._jog_stop()
            return
        unit = self._jog_vector(msg.data)
        if unit is None:
            self.get_logger().warn(f"[jog_tcp] axis number out of range: {msg.data}")
            return
        # the first 3 are linear speeds and the last 3 rotation speeds, so the units differ.
        scale = self._jog_scale()
        linear = self.get_parameter('jog_tcp_speed_max').value * scale
        angular = self.get_parameter('jog_tcp_rot_speed_max').value * scale
        accel = self.get_parameter('jog_tcp_accel_max').value * scale
        xd = self._round6([
            value * (linear if index < 3 else angular)
            for index, value in enumerate(unit)
        ])
        hold = self.get_parameter('jog_hold_time').value
        self._leaving_home()
        self.robot_primary.send_script(f"speedl({xd}, {round(accel, 6)}, {hold})")

    # last safety net used only when get_variable fails. On the normal path the value
    # read directly from a register or from the robot via 29999 is always used. (The home
    # before 2026-10-16 = today's scan standby pose value.)
    _HOME_JOINT_FALLBACK = [0.79941, -1.55029, -2.70026, 1.10839, 0.81203, -3.14139]

    def cb_move_home(self, req, res):
        """Go home via the scan standby pose (2026-10-16, same order as the robot task).

            retreat from the wall (not further back than the standby pose) -> line up with the standby pose
            (back the rest of the way to the standby X, raise or lower to its height)
            -> movej scan standby pose -> movej home

        If already home, only movej home.

        Sending if/else at the top level to the 30001 socket (real-time command channel)
        fails because each line is read as a separate command ("No 'if' command").
        It must be sent as one program wrapped in def name(): ... end, and once sent it
        runs as soon as parsing finishes, so it must not be called separately
        ("move_home_now not defined"). This program runs in a separate context outside the task
        preamble, so task globals (Home_joint etc.) are not visible. So home / standby are decided
        here and embedded as values:
          home    = 322~327 if 328 is 1 (RCS "Save home pose"), otherwise Home_joint
          standby = 310~315 if 308 is 1 (RCS "Save scan standby pose"),
                    otherwise Standby_joint. If both are unknown, home is used as the standby pose.
        Empty registers (all 0) count as never saved — once, looking only at
        308 and reading 310~315 sent the robot to movej([0,0,0,0,0,0]).

        The retreat distance is along tool -Z, the smaller of probe_dist and "the distance left to the standby X".
        If the tool is not facing the wall (+X), there is no retreat.
        The position is taken at the flange (after setting the TCP to 0) — so the distance is right
        even if an EOAT tip TCP is still active.
        """
        if not self.connected:
            res.success = False
            res.message = "[move_home] not connected to the robot."
            return res

        home = self._home_joint_target()
        if not home:
            self.get_logger().warn(
                "[move_home] could not read the home joints from the robot. Using fixed values instead.")
            home = list(self._HOME_JOINT_FALLBACK)
        standby = self._standby_joint_target()
        if not standby:
            self.get_logger().warn(
                "[move_home] scan standby pose (Standby_joint) unknown — using home as the standby pose.")
            standby = list(home)
        # distance [m] to retreat from the wall first. Same as probe_dist (mm) used by the task.
        probe_dist = self.robot_dash.get_variable("probe_dist")
        if not isinstance(probe_dist, (int, float)) or probe_dist <= 0:
            probe_dist = self._HOME_RETREAT_FALLBACK_MM
        retreat_m = round(float(probe_dist) / 1000.0, 4)
        tol = float(self.get_parameter('home_tol_j').value)
        home = [round(float(v), 5) for v in home]
        standby = [round(float(v), 5) for v in standby]

        # p and j clash with the built-in names used for pose/joint literals (p[..], j[..]).
        # Use names with the non-overlapping prefix mh_.
        script = (
            "def move_home_now():\n"
            f"  mh_home = {home}\n"
            f"  mh_sb = {standby}\n"
            "  set_tcp([0, 0, 0, 0, 0, 0])\n"
            "  mh_q = get_actual_joint_positions()\n"
            "  mh_far = 0\n"
            "  mh_i = 0\n"
            "  while (mh_i < 6):\n"
            "    mh_d = mh_q[mh_i] - mh_home[mh_i]\n"
            "    if (mh_d < 0):\n"
            "      mh_d = -mh_d\n"
            "    end\n"
            "    if (mh_d > mh_far):\n"
            "      mh_far = mh_d\n"
            "    end\n"
            "    mh_i = mh_i + 1\n"
            "  end\n"
            f"  if (mh_far > {tol}):\n"
            # retreat from the wall — not further back than the standby pose.
            "    mh_ref = get_forward_kin(mh_sb)\n"
            "    mh_cur = get_actual_tool_flange_pose()\n"
            "    mh_cz = pose_trans(mh_cur, [0, 0, 1, 0, 0, 0])[0] - mh_cur[0]\n"
            "    mh_back = 0\n"
            "    if (mh_cz > 0.2):\n"
            "      mh_back = (mh_cur[0] - mh_ref[0]) / mh_cz\n"
            f"      if (mh_back > {retreat_m}):\n"
            f"        mh_back = {retreat_m}\n"
            "      end\n"
            "    end\n"
            "    if (mh_back > 0.001):\n"
            "      movel(pose_trans(mh_cur, [0, 0, -mh_back, 0, 0, 0]), a=0.4, v=0.1)\n"
            "    end\n"
            # line up with the standby pose — if still in front (base +X), back straight to the standby X
            # and raise or lower to the standby height.
            "    mh_up = get_actual_tool_flange_pose()\n"
            "    if (mh_up[0] > mh_ref[0] + 0.001):\n"
            "      mh_up[0] = mh_ref[0]\n"
            "      movel(mh_up, a=0.4, v=0.1)\n"
            "    end\n"
            "    mh_up = get_actual_tool_flange_pose()\n"
            "    if (mh_up[2] < mh_ref[2] - 0.001 or mh_up[2] > mh_ref[2] + 0.001):\n"
            "      mh_up[2] = mh_ref[2]\n"
            "      movel(mh_up, a=0.4, v=0.1)\n"
            "    end\n"
            "    movej(mh_sb, a=1.4, v=0.5)\n"
            "  end\n"
            "  movej(mh_home, a=1.4, v=0.5)\n"
            # home reached — set register 276. While the task is not running,
            # nobody but this script updates the value.
            "  write_port_register(276, 1)\n"
            "end"
        )
        # stop the running task (scan / marking) **first.** Sending a script to 30001 while
        # the task runs makes the controller refuse it or mix it with the task.
        # The home move must run on its own whenever it is called (ERUT / RCS).
        stop_reply = self.robot_dash.robot_stop()
        if stop_reply is None:
            self.get_logger().warn(
                f"[move_home] no reply to the task stop — {self.robot_dash.last_error}")
        time.sleep(self._HOME_AFTER_STOP_S)
        # **while** moving home the robot is not home. Clear it after stopping the task —
        # clearing it first lets the publishing thread of the still-running task overwrite it.
        self._leaving_home()
        ok = self.robot_primary.send_script(script)
        res.success = bool(ok)
        res.message = "[move_home] Home move requested." if ok else "[move_home] failed to send the script"
        return res

    #: retreat limit [mm] used when probe_dist cannot be read from the robot.
    _HOME_RETREAT_FALLBACK_MM = 100.0

    #: wait [s] between stopping the task and sending the script.
    #: Sending before the stop finishes makes the controller refuse it as "running".
    _HOME_AFTER_STOP_S = 0.5

    #: words in dashboard replies that **answered but refused**.
    #: e.g. "Failed to execute: play", "... not supported in local control mode"
    _DASH_REFUSALS = ("fail", "not supported", "error", "not allowed", "can not", "cannot")

    def _execute_dash_cmd(self, func, name, response):
        """Send a dashboard command and put the result in the service response.

        Previously any reply **at all** counted as success — even when the controller refused with
        "Failed to execute: play", RCS got success.
        And when the link dropped and None came back, only "Result: None" remained and the cause
        was unknown. Now a refusal counts as failure, and None returns the disconnect reason
        along with it.
        """
        res_str = func()
        if res_str is None:
            reason = getattr(self.robot_dash, "last_error", "") or "no reply"
            response.success = False
            response.message = f"[{name}] no dashboard reply — {reason}"
            self.get_logger().warn(response.message)
            return response
        refused = any(word in res_str.lower() for word in self._DASH_REFUSALS)
        response.success = not refused
        response.message = f"[{name}] Result: {res_str}"
        if refused:
            self.get_logger().warn(f"[{name}] refused by the controller: {res_str}")
        return response

    def cb_dash_mode(self, req, res):
        return self._execute_dash_cmd(self.robot_dash.robot_mode, "robotMode", res)

    def cb_dash_status(self, req, res):
        return self._execute_dash_cmd(self.robot_dash.robot_status, "status", res)

    def cb_dash_power_on(self, req, res):
        return self._execute_dash_cmd(self.robot_dash.robot_power_on, "robotControl -on", res)

    def cb_dash_power_off(self, req, res):
        return self._execute_dash_cmd(self.robot_dash.robot_power_off, "robotControl -off", res)

    def cb_dash_brake(self, req, res):
        return self._execute_dash_cmd(self.robot_dash.robot_brakeRelease, "brakeRelease", res)

    def cb_dash_play(self, req, res):
        return self._execute_dash_cmd(self.robot_dash.robot_play, "play", res)

    def cb_dash_remote_on(self, req, res):
        """Turn on remote control mode. play/stop only work with it on."""
        return self._execute_dash_cmd(
            self.robot_dash.robot_remote_control_on, "remoteControl -on", res)

    def cb_dash_pause(self, req, res):
        return self._execute_dash_cmd(self.robot_dash.robot_pause, "pause", res)

    def cb_dash_stop(self, req, res):
        return self._execute_dash_cmd(self.robot_dash.robot_stop, "stop", res)

    def cb_dash_task_status(self, req, res):
        return self._execute_dash_cmd(self.robot_dash.robot_task_status, "task -s", res)

    def destroy_node(self):
        self.robot_dash.disconnect_29999()
        self.robot_primary.disconnect_30001()
        self.robot_modbus.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = RobotControlNode()
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
