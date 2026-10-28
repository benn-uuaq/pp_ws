# pp_ws — Robot piece-picking workspace

ROS 2 (Humble) workspace for a vision-guided **piece-picking cell**: an Elite CS-series robot with a suction gripper
picks products out of a box at positions found by a LUCID **Helios2+ Chroma** camera (Helios ToF + Triton RGB) fixed
above the cell.

The workspace contains two operator programs and the robot driver they share:

| Package | What it does |
| --- | --- |
| [`robot_camera_calibration`](src/robot_camera_calibration/README.md) | **Eye-to-hand calibration GUI.** A marker on the flange is moved through a 3×3×3 grid in front of the camera to solve `base_T_cam` — separately for the RGB camera (PnP) and the depth camera (rigid fit). Includes jogging, start-pose saving and a simulator. |
| [`piece_picking`](src/piece_picking/README.md) | **Picking GUI.** Takes the vision result (product position + surface normal in the camera frame), converts it with a calibration result and runs pick → place → home with servoj motion, tilted picks along the normal, suction outputs and safety checks. |
| [`elite_robot_controller`](src/elite_robot_controller/README.md) | Elite CS robot driver (29999 dashboard / 30001 primary / Modbus) and a ROS 2 control node. The two GUIs use its 30001 driver directly; the node itself is optional here. |
| [`lucid_helios`](src/lucid_helios/README.md) | Not a ROS package: where the LUCID Arena SDK and `arena_api` wheel go (not in git). |

```
                 ┌──────────────── Helios2+ Chroma (fixed above the cell) ───────────────┐
                 │   Triton RGB (2048×1536)          Helios ToF (640×480, XYZ per pixel) │
                 └───────────────┬───────────────────────────────┬───────────────────────┘
                                 │ GigE (arena_api)              │
        robot_camera_calibration │ calib_gui                     │   vision system (separate)
        marker on flange ──────► │   → calibration_rgb.yaml      │   → candidates.json
                                 │   → calibration_depth.yaml    │     position_m + normal
                                 ▼                               ▼
                         piece_picking  picking_gui  ── p_base = R·p_cam + t,  N_base = R·N_cam
                                 │
                                 │ servoj scripts / suction DO (port 30001)
                                 ▼
                         Elite CS robot (+ suction gripper, DO4~7)
```

## Hardware (as set up on 2026-10-28)

| Item | Value |
| --- | --- |
| Robot | Elite CS series, controller `192.168.1.101`, scripts over port 30001 |
| Camera | LUCID Helios2+ Chroma: Helios `HTP003S-001` (ToF) `192.168.1.35`, Triton `TRI032S-C` (RGB) `192.168.1.41` |
| Gripper | suction, standard digital outputs **DO4~7 together, normally ON, OFF = grip** |
| PC | Windows 11 + WSL2 Ubuntu 22.04 (mirrored networking), ROS 2 Humble |

IPs, outputs and limits are all in the package config files; adjust them for your cell.

## Setup

Requirements: Ubuntu 22.04 (native or WSL2), ROS 2 Humble, `direnv`, and the system packages
`python3-opencv` (4.5.4, includes `cv2.aruco`), `python3-pyqt5`, `python3-numpy`, `python3-yaml`.

```bash
sudo apt install direnv python3-opencv python3-pyqt5 python3-numpy python3-yaml python3-venv
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc     # once

git clone git@github.com:benn-uuaq/pp_ws.git ~/pp_ws
cd ~/pp_ws && direnv allow                         # sources ROS, creates .venv (system-site-packages)
pip install pyModbusTCP                            # for elite_robot_controller (into .venv)
```

Camera SDK (only needed for the real cameras; the simulators work without it): download the LUCID Arena SDK
(Linux x64) and the `arena_api` wheel and install them as described in
[`src/lucid_helios/README.md`](src/lucid_helios/README.md). On WSL, use mirrored networking and allow inbound UDP from
the camera subnet (see the calibration README, "When the camera does not connect").

Build:

```bash
cd ~/pp_ws
colcon build --symlink-install       # .envrc makes colcon run with the .venv Python
direnv reload                        # if "Package not found" right after the first build
```

`.envrc` puts a small `colcon` wrapper into `.venv/bin` so the installed executables use the `.venv` Python (where
`arena_api` lives) instead of `/usr/bin/python3`.

## Quick start (no hardware)

```bash
ros2 run robot_camera_calibration calib_gui --sim    # simulated robot + simulated Helios / Triton
ros2 run piece_picking picking_gui --sim             # simulated robot, simulated calibration if no result file
```

## Workflow on the real cell

1. **Calibrate** — `ros2 run robot_camera_calibration calib_gui`
   * connect the robot, tick the camera side and the camera (RGB or depth), connect the camera
   * enter the marker position (flange → marker centre), jog the marker into view, **[Save current pose]**
   * **[Start calibration]** → `~/pp_ws/calibration/calibration_rgb.yaml` / `calibration_depth.yaml`
2. **Pick** — `ros2 run piece_picking picking_gui`
   * tick the calibration result that matches the vision frame
   * teach **Home**, **Pick standby** (above the box) and **Place** with the pendant and save them
   * enter the gripper length (with the pendant TCP at the flange `[0,0,0,0,0,0]` it is required)
   * paste the vision result (**[Paste vision JSON]**: `position_m` + `normal`) and, for tilted products, tick
     **[Apply rotation]**
   * **[Pick → Place → Home]**:
     home → pick standby → approach (tool turns here) → product (straight along tool +Z) → grip →
     retreat 50 mm along the tool → lift 200 mm along base +Z → place approach → place → release → retreat → home

All motion is **servoj** with sine-shaped acceleration; the pendant speed ratio does **not** apply, so start with low
speeds and a final offset of 20–50 mm, and keep a hand on the emergency stop.

See [`COMMANDS.txt`](COMMANDS.txt) for a one-page command reference.

## Conventions

* Robot poses: `[x, y, z, rx, ry, rz]` in m / rad, RPY (`R = Rz·Ry·Rx`), base frame, relative to the active TCP.
* Camera frames: OpenCV convention (X right, Y down, Z forward), m (calibration) / vision `position_m` in m.
* Calibration result: `base_T_cam` with `p_base = R @ p_cam + t`; the normal is rotated only (`N_base = R @ N_cam`).
* Suction direction = tool +Z, pointing into the product surface; the vision normal sign is fixed automatically.
* Runtime data (not in git): `~/pp_ws/calibration/` (results, run logs, images), `~/pp_ws/picking/` (taught poses,
  last inputs).

## Tests

```bash
cd ~/pp_ws
python -m pytest src/robot_camera_calibration/test   # ~3 min, includes full simulated calibrations
python -m pytest src/piece_picking/test              # ~1 min
colcon test --packages-select elite_robot_controller && colcon test-result --verbose
```

## Status / TODO

* The servoj motion and the suction outputs have been tested in simulation only; verify on the real robot at low speed.
* Vision node trigger (`piece_picking/vision_client.py`) is a placeholder — the vision result is pasted for now.
* Approach path that avoids the box walls (the approach is a straight line along the tool axis).
* The RGB and depth calibrations should use the same, carefully measured marker offset.

## Repository layout

```
pp_ws/
├── .envrc                       direnv: ROS 2, .venv, colcon wrapper
├── COMMANDS.txt                 command reference
├── README.md
└── src/
    ├── robot_camera_calibration/   calib_gui, motion_tool, camera drivers, simulators, tests
    ├── piece_picking/              picking_gui, picking core, servoj motion, vision paste, tests
    ├── elite_robot_controller/     Elite driver + ROS 2 node, register map, robot simulator, tests
    └── lucid_helios/               README only (Arena SDK / arena_api go here, not in git)
```
