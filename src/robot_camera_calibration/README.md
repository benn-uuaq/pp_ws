# robot_camera_calibration

Eye-to-hand calibration package that finds where the Helios2+ Chroma camera fixed in the piece-picking
cell (**RGB = Triton**, **depth = Helios ToF**) sits relative to the robot base. An ArUco marker is
attached to the robot flange; the robot moves the marker through a 3×3×3 grid in front of the camera,
and at every pose the camera measurement is paired with the marker position reported by the robot to
solve `base_T_cam`.

```
p_base = R @ p_cam + t        (units: m)
```

Each camera is selected in the GUI and calibrated **separately**, and the results are saved separately.

| Camera | Method | Result file (`~/pp_ws/calibration/`) |
| --- | --- | --- |
| **RGB (Triton)** | **PnP** between marker pixels (2D) ↔ marker base coordinates known to the robot (3D). Uses the Chroma factory intrinsics stored in the camera. Independent of marker size and ToF | `calibration_rgb.yaml` |
| **Depth (Helios ToF)** | Rigid fit between marker 3D (ToF plane fit) ↔ marker base coordinates | `calibration_depth.yaml` |

The piece-picking program (`piece_picking`) uses one of the two. Use the result of the same camera
whose coordinates the vision system reports. (The older `aruco_rigid_result.yaml` from before
2026-10-28 is a depth result and was copied to `calibration_depth.yaml`.)

Origin: `D:\RND_WORK\PiecePicking\robot_camera_calibration` (RealSense version).

## Running

```bash
cd ~/pp_ws && colcon build --symlink-install     # direnv sets up the environment
ros2 run robot_camera_calibration calib_gui        # real robot + Helios
ros2 run robot_camera_calibration calib_gui --sim  # simulated robot + simulated camera (no hardware)
```

1. Enter the **robot IP** → [Connect robot]. Working when the TCP values update.
2. Tick the **camera position** (+X / −X / +Y / −Y): horizontal position of the camera seen from the
   robot base. The drawing and hints such as "Towards camera (depth) = base +Y" change accordingly.
3. Tick the **camera** (RGB (Triton) / Depth (Helios ToF)) → **camera IP** (empty = auto-discover; see
   `config/calibration.yaml`) → [Connect camera]. The live image shows the marker detection right away:
   green square = detected (with 3D for depth), orange = detected but no 3D points, grey = marker with a
   different id. The result file name for this camera and whether it exists are shown as well.
   **Sharpness** in the marker panel is the width (full resolution, px) over which the brightness of the
   marker's outer edge rises from 10 to 90 %. In focus it is 2–3 px; ≤ 3.5 px green, ≤ 6 px orange,
   above that red. Use it when cleaning the lens front or removing a protective film
   (Triton measured on 2026-10-28: about 8.5 px, blurred).
   Changing the camera selection while connected disconnects the camera.
4. Check the **marker position** (reference → marker centre, see below) and the **grid spacing**.
5. **Jog the robot** until the marker is inside the circle in the middle of the image, then **[Save current pose]**.
   * Mode: **continuous** (moves while pressed, `stopl` on release) or **step** (1 / 5 / 10 / 50 mm,
     0.5 / 1 / 2 / 5° per press). Speeds are next to it (mm/s, °/s).
   * Moves use camera axes (lateral · depth · vertical); the button labels show the actual base axis in
     brackets. Rotations turn about base RX / RY / RZ around the TCP.
   * Continuous jog sends one `speedl` on press (same approach as the robot node) and stops on release.
     If the release is missed, a 0.1 s watchdog stops it; failing that, the robot stops by itself after
     `jog.hold_s` (2 s).
   * The saved pose is `~/pp_ws/calibration/start_pose.yaml` (TCP pose, active TCP offset, joints).
     **[Go to saved pose]** returns there with a linear move (movel); the red **[Stop]** stops it immediately.
6. With "**Start calibration from the saved pose**" ticked, press [Start calibration]. The robot first
   moves to the saved pose and then visits 27 poses around it (untick to start from the current pose).
   Keep a hand on the emergency stop.
7. After returning to the start pose the result is shown (RMS, camera position, file path). [Stop]
   stops with `stopl`.

The IPs, camera position, marker position and spacing you enter are kept in
`~/pp_ws/calibration/ui_state.yaml` and come back at the next start (the config file is not modified).

## Marker position (reference → marker centre)

It is **to the marker centre**, not to a corner.

* Centre = intersection of the two diagonals of the black square. This is exactly the point the program
  measures in the image, so the input must point to the same point. Entering the distance to a corner
  shifts the camera position `t` by half a diagonal (about 42 mm for a 60 mm marker).
* It is a point **on the marker surface**. ToF measures the paper surface, so Z includes the bracket,
  plate and paper thickness.
* It is **three values X, Y, Z**, not one distance, and the axes are the **reference frame**
  (flange or TCP frame), not the base. Jog +X, +Y in the tool frame on the pendant to check directions.
* **Reference**
  * Flange (recommended): measured from the centre of the flange face. Whatever the active TCP on the
    pendant is, the program reads the TCP offset from the robot and compensates.
  * Active TCP: measured from the TCP set on the pendant. If the pendant TCP is registered at the marker
    centre, enter [0, 0, 0]. Do not change the TCP during calibration.

Example: marker plate 45 mm in front of the flange face, marker centre 12 mm along flange +X and 30 mm
along −Y → flange reference `[12, -30, 45]`.

### Marker rotation (RX / RY / RZ)

Marker rotation seen from the reference frame [deg], `R = Rz·Ry·Rx` (same order as the robot rpy).
Marker axes follow the ArUco standard: X = top-left → top-right corner, Y = bottom → top,
Z = out of the marker surface (towards the camera).

**It is not used to compute the camera pose (R, t).** Each pose pairs only the marker centre point, so as
long as the centre position (X/Y/Z) is right, a rotated marker gives the same result. Instead the program
measures the marker orientation from the 3D of its four corners at every pose, and after calibration

* stores the measured marker rotation in the result (`meta.marker.rotation_measured_deg`) and on screen,
* warns if it differs from the input by more than `rotation_tolerance_deg` (default 5°).

If you get a warning that the input rotation is wrong, also suspect the X/Y/Z measured in that frame. You
can use the measured value as the next input.

The tool orientation is the same at every pose, so a wrong X/Y/Z does **not** increase the RMS — it silently
shifts `t` by the same amount. Measure it accurately with CAD or calipers.

## Camera-relative move axes

The grid is defined in three directions seen from the camera, not in base X/Y/Z.

| Axis | Meaning |
| --- | --- |
| `lateral` (left/right) | + = right side as the camera looks at the robot |
| `depth` | + = horizontal direction towards the camera |
| `vertical` (up/down) | + = up (always base +Z) |

| Camera position | lateral | depth | vertical |
| --- | --- | --- | --- |
| `+x` | +Y | +X | +Z |
| `-x` | −Y | −X | +Z |
| `+y` | −X | +Y | +Z |
| `-y` | +X | −Y | +Z |

If all three spacings are equal, the grid points are the same whichever side the camera is on. The
difference shows up with different spacings per axis and in the logs / single moves. If the resulting
camera position differs from the ticked direction, a warning is shown (the fit itself is valid).

## 3D measurement (Helios)

The Helios gives XYZ for every pixel (camera frame: X right, Y down, Z in front of the lens). ToF is weak on
the black cells and produces jumping points on the marker border, so instead of a single centre point a
plane is fitted to the points inside the marker square (`sample_scale` 0.8) and the 3D of the centre pixel is
read from it (for a pinhole camera, X/Z, Y/Z and 1/Z of a plane are linear in the pixel coordinates). The
on-screen "N plane points, depth residual" shows its quality. Each pose uses the median of 10 frames.

## Helios setup (WSL) — installed 2026-10-27

| Item | Location / value |
| --- | --- |
| Arena SDK (Linux x64, AVMP 1.0.10.11) | `~/pp_ws/src/lucid_helios/ArenaSDK_Linux_x64` (excluded from the build with `COLCON_IGNORE`) |
| Library path registration | `/etc/ld.so.conf.d/Arena_SDK.conf` (run the official `Arena_SDK_Linux_x64_AVMP.conf`) |
| GigE receive buffer | `net.core.rmem_default/max = 33554432` in `/etc/sysctl.conf` |
| arena_api 2.9.5 | installed in `~/pp_ws/.venv` |
| Camera | Helios2+ `HTP003S-001`, S/N 263501855, FW 1.19.0.0 |

The Arena SDK and arena_api are not in this repository (LUCID licence, size). Download them from the LUCID
website; see `src/lucid_helios/README.md`. If the SDK is moved, run `sudo sh Arena_SDK_Linux_x64_AVMP.conf -r`
again in its folder. To remove it: `Uninstall_Arena_SDK_Linux_x64_AVMP.sh`.

### When the camera does not connect

* **"Another program is using the camera, so it cannot be controlled (PC in use …)"** — ArenaView or a similar
  program on another PC has the camera open. A GigE camera can be controlled by only one program at a time,
  so close that program.
* **Connects, but "No frames arriving"** — images (GVSP) are sent by the camera to the PC over UDP, so if WSL
  is in **NAT mode** the camera cannot reach WSL (discovery and settings still work via unicast). Switch WSL to
  mirrored mode and allow inbound UDP to WSL.

  1. `C:\Users\<user>\.wslconfig`
     ```ini
     [wsl2]
     networkingMode=mirrored
     ```
  2. Administrator PowerShell — allow inbound UDP from the camera subnet to WSL (Hyper-V firewall)
     ```powershell
     New-NetFirewallHyperVRule -Name LucidHeliosToWSL -DisplayName "LUCID Helios to WSL" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol UDP -RemoteAddresses 192.168.1.0/24 -Action Allow
     ```
  3. `wsl --shutdown`, then open the WSL terminal again (all running WSL programs are closed).

* **"Incomplete frame (packet loss)"** — the WSL path loses packets when they arrive in bursts. Increase
  `camera.packet_delay_ns` (gap between packets). Measured: 1500 B packets with gap 0 → most frames broken;
  20000 ns → about 17 fps with no loss (default); 40000 ns → about 10 fps.
* **`arena_api is not installed`** — `ros2 run` is using the system Python. Run `colcon build` again inside
  `~/pp_ws`. `.envrc` makes `colcon` run with the `.venv` Python, so the first line of the installed
  executables points to the `.venv` Python.

### Marker size

At work-table height about 1.5 m from the camera, one Helios pixel is about 3 mm. A 60 mm marker is only
about 20 px, so detection can be unstable. Use a marker of 100 mm or more, or raise the start pose towards
the camera (0.8 – 1.0 m).

## Results

```
~/pp_ws/calibration/
├── calibration_rgb.yaml         <- RGB (Triton) camera result (last one that passed)
├── calibration_depth.yaml       <- depth (Helios ToF) camera result
├── ui_state.yaml                <- last GUI inputs
└── runs/20261028_161230/        <- created for every run
    ├── result.yaml              <- this run's result (saved even if the quality check fails)
    ├── samples.yaml             <- robot pose, camera measurement (pixel / 3D) and residual per pose
    ├── run.log
    └── images/pose_01.png ...
```

Both files use the `base_T_cam: {R, t}` format with `meta.camera_kind` set to `rgb` / `depth`.
The RGB result also stores the intrinsics used (`meta.camera.camera_matrix`, `dist_coeffs`) and the
reprojection error (`meta.rms_reprojection_px`). For RGB, the RMS (mm) is the reprojection error converted
to a lateral distance at that range.

| RMS | Meaning |
| --- | --- |
| < 1.5 mm | good |
| 1.5 – 3 mm | enough for most picking |
| 3 – 5 mm | check the marker mounting / measurement conditions |
| > 5 mm | fail, the main result file is not overwritten |

## Files

| File | Role |
| --- | --- |
| `calib_gui.py` | GUI + calibration sequence |
| `calib_core.py` | marker detection, plane-fit 3D measurement, rigid fit, PnP fit (RGB), result file |
| `camera_helios.py` | Helios2 acquisition (Arena SDK, Coord3D_ABCY16) |
| `camera_triton.py` | Triton RGB acquisition (BayerRG8 → BGR), reads the Chroma factory intrinsics |
| `camera_base.py` | common frame format, latest-frame buffer |
| `motion_frame.py` | move axes depending on the camera position, grid |
| `robot_interface.py` | 30001 state polling, `movel`, reach check (uses the `elite_robot_controller` driver) |
| `pose_math.py` | pose ↔ matrix, marker position from the flange |
| `settings.py` | config file, saving UI inputs |
| `motion_tool.py` | command-line tool to check the moves without a camera (`axes`, `plan`, `step`, `run`) |
| `sim_robot.py`, `sim_camera.py` | simulated robot, simulated Helios / Triton |
| `config/calibration.yaml` | all settings |

## Tests

```bash
cd ~/pp_ws && python -m pytest src/robot_camera_calibration/test
```

Runs the full calibration with simulated hardware for each of the four camera positions and checks that the
true camera position is found within 3 mm (about 1 minute).
