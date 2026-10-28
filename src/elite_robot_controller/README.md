# elite_robot_controller

Package for controlling the Elite CS612 collaborative robot from ROS 2. It is the control part of the manipulator
mounted on an AMR.

Copied from `~/dusan_ws` (originally ported from the `robot_controller` package in `~/ws_elt`: **only the driver,
control node and launch file**). The original `ui/` and `tasks/` (palletising, unloading, pick & place) were leftovers
of the ws_argo project, depended on `argo_project_interfaces` and were unrelated to wall inspection, so they were not
taken over.

In pp_ws the calibration and piece-picking GUIs use the 30001 driver (`Robot_30001`) from this package directly;
the ROS 2 node below is not required for them.

## Communication

The node talks to the robot controller over three channels at the same time. All three must connect for the node
to count as connected.

| Port | Class | Role |
| --- | --- | --- |
| 29999 | `Robot_29999` | Dashboard. Text commands for power, brakes, play / pause / stop |
| 30001 | `Robot_30001` | Primary. Robot state and alarms from the binary packet stream |
| 502 | `Robot_modbus` | Modbus TCP. Reading and setting state per register |

## Node

### `robot_control_node`

Reads the Modbus registers and collects the 30001 alarms every 0.1 s (10 Hz) and publishes them.

**Parameters**

| Name | Default | Description |
| --- | --- | --- |
| `robot_ip` | `192.168.227.134` | Elite controller IP |
| `modbus_port` | `502` | Modbus TCP port. 502 needs privileges on Linux, so it can be changed for simulator tests |
| `jog_joint_speed_max` etc. | — | jog / home move parameters, see "Jog and home move" below |

**Published topics**

| Topic | Type | Source |
| --- | --- | --- |
| `robot/status/robot_mode` | `std_msgs/Int32` | Modbus 66 |
| `robot/status/control_method` | `std_msgs/Int32` | Modbus 71 |
| `robot/status/operation_mode` | `std_msgs/Int32` | Modbus 72 |
| `robot/status/tcp_pose` | `std_msgs/Float32MultiArray` | Modbus 384~389 (current TCP, base frame) |
| `robot/status/tcp_pose_zero` | `std_msgs/Float32MultiArray` | Modbus 280~285 (pose relative to the origin) |
| `robot/status/scan_state` | `std_msgs/Int32MultiArray` | Modbus 290~298 (scan progress of the robot task) |
| `robot/status/task_state` | `std_msgs/Int32` | Modbus 500 (1 running, 2 paused, 3 stopped) |
| `robot/status/speed_scale` | `std_msgs/Int32` | Modbus 17 (robot motion speed ratio %, 2~100) |
| `robot/status/joint_position` | `std_msgs/Float32MultiArray` | Modbus 73~78 (joint angles) |
| `robot/status/alarms` | `std_msgs/String` | 30001 alarms |
| `robot/status/connected` | `std_msgs/Bool` | whether the three channels are connected |

## Modbus register map

**Addresses are not kept in code but managed in one place, [`config/modbus_registers.json`](config/modbus_registers.json).**
When an address changes or is newly confirmed, only this file needs to change, not the code. The operating UI reads
the same file, so the definitions never diverge.

A different path can be given with the `register_map` parameter.

```bash
ros2 run elite_robot_controller robot_control_node --ros-args -p register_map:=/path/my_registers.json
```

### Current content

| Entry | Address | Content |
| --- | --- | --- |
| `read.robot_mode` | 66 | robot mode |
| `read.control_method` | 71 | control method |
| `read.operation_mode` | 72 | operation mode |
| `read.joint_position` | 73~78 | joint angles (base, shoulder, elbow, wrist 1~3) |
| `read.tcp_absolute` | 384~389 | current TCP (base frame) |
| `read.tcp_zero_relative` | 280~285 | pose relative to the origin |
| `read.scan_state` | 290~298 | scan progress, in the order `state, row_idx, rows, alive, zero_ok, finished, pitch, ...` |
| `read.task_state` | 500 | task state (1 running, 2 paused, 3 stopped). An **input register**, not holding, so it is read through the driver's fallback |
| `read.speed_scale` | 17 | robot motion speed ratio [%] 2~100. **Used read only** — the setting is sent with 29999 `speed -v N` |
| `write.linear_speed` | 306 | work speed [mm/s]. The task's `movel` speed |
| `write.speed_ratio` | 307 | robot's own speed ratio [%] 2~100 |
| `write.pose_src` | 308 | with 1 the task uses the reference poses in 310~321 |
| `write.home_joint` | 310~315 | home joints [mrad] |
| `write.start_pose` | 316~321 | start pose (length 0.1 mm, rotation mrad) |
| `write.work_area` | 256~259 | work area [width, height, scanner height, overlap] mm. Stored as **plain integer mm** (`kind: "raw"`, not 0.1 mm) |
| `write.param_src` | 266 | with 1 the task uses the work area in 256~259 |

The write addresses are in the general-purpose register range read by the robot task (256~383). 256~305, which the
task already writes, are avoided.

An entry with `address` `null` means the address is not confirmed. **The node refuses that request and does not touch
Modbus at all**, because writing to a wrong register could move the robot unexpectedly. The operating UI reads the
same file and disables the corresponding buttons.

### Pose value conversion

A pose is 6 consecutive registers, one per axis, in `[X, Y, Z, Rx, Ry, Rz]` order. `Robot_modbus.get_all_registers()`
already converts to signed 16 bit, so the node only converts units.

The entry's `kind` decides the conversion. `pose` treats the first 3 as position and the last 3 as rotation; `angle`
treats all 6 as rotations. Joint angles are mrad on all 6 axes, so they are `angle`.

| Component | Setting | Conversion | Published unit |
| --- | --- | --- | --- |
| X, Y, Z (`pose`) | `scale.position_per_count` | register × 0.1 | mm |
| Rx, Ry, Rz (`pose`), 6 joints (`angle`) | `scale.rotation_per_count` | register × 1.0 | mrad |

If not all 6 registers can be read, publishing is skipped so no wrong pose goes out.

> **To check:** the position factor 0.1 was inherited from the pre-port code and has not been verified on the real
> robot. If the registers are 1 mm rather than 0.1 mm, change only `position_per_count` in the config to 1.0.

## Command interface

| Kind | Name | Write entry |
| --- | --- | --- |
| service (`std_srvs/Trigger`) | `robot/command/save_home_pose` | `save_home_pose` |
| service (`std_srvs/Trigger`) | `robot/command/save_start_pose` | `save_start_pose` |
| service (`std_srvs/Trigger`) | `robot/command/move_home` | `move_home` |
| topic (`std_msgs/Int32`) | `robot/command/linear_speed` | `linear_speed` |
| topic (`std_msgs/Int32`) | `robot/command/speed_ratio` | `speed_ratio` |
| topic (`std_msgs/Float32MultiArray`) | `robot/command/home_joint` | `home_joint` |
| topic (`std_msgs/Float32MultiArray`) | `robot/command/start_pose` | `start_pose` |
| topic (`std_msgs/Float32MultiArray`) | `robot/command/work_area` | `work_area` (on success `param_src` is set to 1 too) |
| topic (`std_msgs/Int32`) | `robot/command/jog_joint` | none (30001 script) |
| topic (`std_msgs/Int32`) | `robot/command/jog_tcp` | none (30001 script) |

One-shot commands without a value need to return success, so they are services; commands with a value come in as topics.

### Jog and home move

Jog and home moves are handled by **sending scripts to port 30001**, not through registers.

- The jog value's **sign is the direction and its absolute value the axis number (1~6)**; `0` is stop.
- Moving sends `speedj` / `speedl`; **stopping uses 29999 `stop`** to stop immediately without deceleration.
- **The jog speed follows register 307 (speed ratio %).** The units differ, so the parameters below are for 100 %
  and are scaled down by the ratio.

| Parameter | Default | Unit | Target |
| --- | --- | --- | --- |
| `jog_joint_speed_max` | 0.50 | rad/s | `qd` of `speedj` |
| `jog_tcp_speed_max` | 0.10 | m/s | first 3 of `xd` of `speedl` |
| `jog_tcp_rot_speed_max` | 0.50 | rad/s | last 3 of `xd` of `speedl` |
| `jog_accel_max` | 1.00 | rad/s² · m/s² | `a` |
| `jog_hold_time` | 0.5 | s | `t` |

For example at 20 % the joint jog is 0.1 rad/s and the acceleration 0.2.

Keep `t` (`jog_hold_time`) **short**. Per the manual the robot keeps moving for `t`, so a long value means it does not
stop for that long if the stop command fails. The operating UI sends the command once on press, and the 29999 stop on
release does the actual stopping.
- `robot/command/move_home` goes home via the scan standby pose (same order as the robot task). Home = 322~327 if
  328 (`home_src`) is 1, otherwise the robot's `Home_joint`; standby = 310~315 if 308 (`pose_src`) is 1, otherwise
  `Standby_joint`. Empty registers (all 0) count as never saved (once the robot went to `movej([0,0,0,0,0,0], ...)`
  because of that).
  - Sending `if`/`else` at the top level to the 30001 socket (real-time command channel) fails because each line is read
    as a separate command (`No 'if' command`). It must be sent as **one program** wrapped in
    `def move_home_now(): ... end`, and it runs as soon as parsing finishes, so it must not be called separately
    (`move_home_now` not defined error).
  - This program runs in a separate context outside the task preamble, so task globals such as `Home_joint` are not
    visible. The values are read from the robot via 29999 `variable -get` (`Robot_29999.get_variable`) or the registers
    and embedded in the script. Only if reading fails are the fixed fallback values used.
  - Avoid the variable names `p` and `j`; they clash with pose / joint literals (`p[..]`, `j[..]`).
- `robot/command/work_area` passes the width / height / scanner height / overlap of the current grid from the `grid`
  block of the MQTT `job_cmd`, used when a large cylinder is scanned in AMR circumferential zones + cobot vertical grids.
  The `work_area` entry is `kind: "raw"`, so unlike the pose entries the values are stored as plain integer mm
  (not 0.1 mm).

> `speedj`/`speedl` are affected by the controller's speed percentage (script manual 3.1.26/3.1.27). Below 100 % the real
> speed is reduced accordingly.

**Services** (all `std_srvs/Trigger`)

Under `robot/dashboard/`:

| Service | Action |
| --- | --- |
| `connect` / `disconnect` | connect or disconnect the three channels. Does not write Modbus registers |
| `robot_mode`, `status` | 29999 dashboard queries |
| `power_on`, `power_off`, `brake_release` | power and brakes |
| `play`, `pause`, `stop` | program control |

Dashboard commands go out on the 29999 socket, so they work regardless of register addresses. `robot/command/*`, on the
other hand, write Modbus registers and only work once the addresses are set.

## Running

```bash
ros2 run elite_robot_controller robot_control_node --ros-args -p robot_ip:=192.168.227.134
```

### Testing without a robot (simulator)

To test without a real robot, start [`tools/elite_robot_simulator.py`](tools/elite_robot_simulator.py) first. It imitates
the three channels 29999/30001/Modbus, and jog commands actually move the joint / TCP registers.

```bash
# terminal 1: fake robot
python3 src/elite_robot_controller/tools/elite_robot_simulator.py

# terminal 2: node (connects to the simulator's 5502 instead of 502)
ros2 run elite_robot_controller robot_control_node --ros-args \
    -p robot_ip:=127.0.0.1 -p modbus_port:=5502
```

All three channels connect and the joint / TCP values move in real time on jog commands. The simulator reads the
register addresses from `config/modbus_registers.json`, so it never uses different addresses from the node.

```bash
ros2 launch elite_robot_controller elite_cs612.launch.py robot_ip:=192.168.227.134
```

**The node does not exit when the robot is not connected.** It tries to connect once at start; on failure it only logs a
warning and waits, reconnecting in the background. The `robot/dashboard/connect` service retries as well. Before
connecting it skips the Modbus reads and only reports the state on `robot/status/connected`.

## Dependencies

Besides `rclpy`, `std_msgs`, `std_srvs` in `package.xml`, **`pyModbusTCP` is needed as a pip dependency**. It has no
rosdep key, so it is not declared in `package.xml`.

```bash
pip install pyModbusTCP
```

## Tests

```bash
colcon test --packages-select elite_robot_controller && colcon test-result --verbose
```

`test_pose_publishing.py` replaces the Modbus client and checks the pose register addresses and unit conversion without
a robot.

`flake8` and `pep257` currently fail. `robot_driver.py` and `robot_control_node.py` were ported from `ws_elt` without
logic changes and the original already did not follow the style rules. A cleanup is postponed until after real-robot
verification.

## Note

`robot_driver.py` is tightly coupled to the Elite controller's binary packet struct formats (`FMT_*` constants). A
controller firmware change can break the parsing, so do not modify the format strings lightly.
