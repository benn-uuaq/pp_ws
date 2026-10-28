# piece_picking

GUI that moves the robot **pick → place → home** using the calibration result and the **product position +
surface normal** reported by the vision system. It is a **separate program** from the calibration program
(`robot_camera_calibration`) and only reads its result files. The robot communication module
(`robot_interface`) and the simulated robot are taken from that package.
**All robot motion is servoj** (see "servoj motion" below).

## Running

```bash
cd ~/pp_ws && colcon build --symlink-install
ros2 run piece_picking picking_gui          # real robot
ros2 run piece_picking picking_gui --sim    # simulated robot
```

1. **Robot**: IP → [Connect robot].
2. **Calibration result**: both results are loaded automatically at start; **tick the one to use**.
   * **RGB camera result** — `~/pp_ws/calibration/calibration_rgb.yaml`
   * **Depth (ToF) result** — `~/pp_ws/calibration/calibration_depth.yaml`
3. **Work poses**: jog the robot with the pendant, then **[Save Home]**, **[Save Pick standby]**, **[Save Place]**
   (`~/pp_ws/picking/poses.yaml` — TCP pose, joints, active TCP offset). Check them with the [Go to ...] buttons.
   * **Pick standby** (optional): a taught pose above the box, visited right before the approach point.
     The tool turns towards the product only on the move from the standby pose to the approach point.
4. **Vision input mode**: **Manual entry** / **Vision node trigger** (not implemented yet, TODO).
   In **the same camera frame as the selected result**:
   * X, Y, Z [m] = vision `position_m` as is
   * NX, NY, NZ = vision `normal` as is (direction vector; sign and length are handled automatically)
   * **[Paste vision JSON]**: copy the vision result (`candidates.json`) and press the button to fill both.
     One entry, several entries, the whole file, or just the `"position_m"` / `"normal"` lines all work (trailing
     commas and cut-off fragments are fine). With several candidates a selection dialog opens (rank 1 is the
     default). If the clipboard has nothing usable, a paste dialog opens. Long decimals are cut to 0.01 mm
     (5 decimals in m) for the position and 6 decimals for the normal. A position above 5 m (mm values entered)
     or a Z of 0 or less is refused.
   * **[Apply rotation]**: tilt the tool along the normal. When off, or the normal is 0, the robot picks
     **vertically** with the tool orientation unchanged.
5. Check the **gripper (suction) length**; the picking target is computed and shown immediately (including how
   many degrees and in which base direction the normal tilts — check this before applying the rotation).
6. Motion
   * **[Pick → Place → Home]**: (home) → (pick standby) → approach → product → grip (DO4~7 OFF) →
     **retreat along the tool axis** (`Retreat (tool)`, default 50 mm) → **lift along base +Z** (`Lift (base +Z)`,
     default 200 mm, tool still tilted) → place approach → place → release (DO4~7 ON) → retreat (= place approach) → home.
     The retreat + lift take the held product out of the box **before** the tool turns back upright on the way to the
     place, so the product does not hit the box. The approach point is backed off **along the tool axis by the approach
     distance**; the last segment and the retreats run at the slow speed.
   * [Retreat up, then home] uses the same retreat (tool axis) + lift (base +Z) before going home.
   * [Move to approach] / [Move to product] / [Retreat up, then home] / [Stop]
   * With "Start picking from home" on, the robot goes home first if it is somewhere else.
   * Every move shows the sequence and coordinates and asks for confirmation.

Inputs are kept in `~/pp_ws/picking/ui_state.yaml` and come back at the next start.

## Coordinate conversion

```
Vision result (same camera frame as the selected result: position m, normal direction vector)
   │  selected calibration result base_T_cam
   │    position  p_base = R · p_cam + t
   │    normal    N_base = R · N_cam        (a direction, so rotation only, no translation)
   ▼
Robot base frame
```

* Flipping the sign of only one normal component (e.g. only NX) is not a rotation but a **mirror reflection**.
  If the position is used as is and the robot goes to the right place, enter the normal as is too (same camera frame).
* The normal sign (outward / inward) does not matter. The suction direction is automatically chosen to point away
  from the camera (into the surface).
* **Checking the tilt direction**: tilt a flat plate by 15–20° (e.g. so its face looks towards base +X), run the
  vision, enter the normal; the GUI must show "product face points towards base +X". If it shows the opposite,
  check the vision side's coordinate axes.

## Picking target

* **Tool orientation** (GUI [Apply rotation])
  | | Tool rotation |
  | --- | --- |
  | Apply rotation + normal given | Turn the reference tool orientation as little as possible so tool +Z (suction direction) matches the normal — no wrist (about-axis) rotation |
  | Apply rotation off / normal 0 | Reference tool orientation unchanged — vertical pick |

  Reference tool orientation = the pose right before the approach point: the **pick standby pose** if saved,
  else the **home pose** when "Start picking from home" is on, else the current pose.
* **Suction tip** target = product point − tool +Z × `final offset` (stops in front of the surface). Approach point =
  backed off from there along the tool axis by the `approach distance` (default 50 mm). The tool is tilted **before
  reaching** the approach point, and the approach point and the product use the same tool orientation, so
  **approach point → product is a straight move along tool +Z**. The retreat runs back along the same line —
  even for tilted picks the pad enters perpendicular to the surface.
* **Gripper (suction) length**: distance from the pendant's active TCP to the suction pad tip along tool +Z.
  TCP target sent to the robot = suction tip target − tool +Z × gripper length.
  * If the active TCP is the flange (0), use flange face → pad tip; if the pendant's active TCP already includes the
    tool length, use 0.
  * **Gripper length 0 + active TCP = flange `[0,0,0,0,0,0]`** blocks the move buttons (the flange would go down to the
    product and the gripper would collide). Entering both a pendant TCP and a gripper length shows an "applied twice" warning.
  * The gripper length is added **only to the vision-based pick target**. Home, pick standby and place are TCP poses
    taught with the pendant with the gripper mounted, so the gripper is already included and they are used as is.
    Teach the place pose at the height for placing **with a product held** (if taught with an empty pad on the floor,
    the product is pressed down by its thickness).
* The last segment before the product, placing and the retreats run at `pick.descent_speed_ratio` (30 %) of the speed.

* **After the pick**: retreat = final point backed off along the tool axis by `pick.retreat_mm` (50 mm), lift = that pose
  raised by `pick.lift_mm` (200 mm) along base +Z with the same tool orientation. Only the move from the lift pose to the
  place approach turns the tool back.

> **TODO**: the products are inside a box, so an approach path that avoids the box walls. Now the approach is straight
> along the tool axis, and lift pose → place approach is a straight line.

## servoj motion

All moves are servoj (`servo_motion.py`). Like dusan_v5 (`dus5_pass_r.script`), a `def ... end` script is sent to
port 30001 and a **loop on the robot** computes the next target every 4 ms and sends it with `servoj`.

* **Linear** (pick / place approach and retreat): `interpolate_pose` → `get_inverse_kin` → `servoj`.
  Checked with `get_inverse_kin_has_solution` before starting; without a solution it does not move and `halt`s.
* **Joint interpolation** (home, pick standby): joint-space interpolation to the saved joints → `servoj` (instead of movej).
* Velocity profile: a trapezoid whose **acceleration / deceleration phases are sine-shaped** — acceleration starts and
  ends at 0, so starts and stops do not jerk (the configured acceleration is the peak). Progress advances by the real
  elapsed time from `controller_timestamp()`; at the end it keeps sending for twice the lookahead, then `stopj`.
  The slower of position and rotation sets the timing. If it still jerks, lower the `servo` accelerations
  (e.g. `joint_acc_deg_s2: 40`).
* The first line of the script sets the active TCP currently reported by the robot again with `set_tcp`
  (30001 scripts run in a different context from the pendant task).
* **The pendant speed ratio does not apply to servoj.** The GUI speed and the `servo` settings are the real speeds.

| Setting (`servo`) | Default | |
| --- | --- | --- |
| `dt` · `lookahead` · `gain` | 0.004 s · 0.1 s · 300 | servoj arguments (longer lookahead = smoother) |
| `lin_acc` | 0.3 m/s² | linear acceleration / deceleration (peak) |
| `rot_speed_deg_s` · `rot_acc_deg_s2` | 30 °/s · 60 °/s² | tool rotation during linear moves |
| `joint_speed_deg_s` · `joint_acc_deg_s2` | 30 °/s · 60 °/s² | moves to saved poses (joint that turns the most) |

## Suction outputs

Suction switches **DO4, DO5, DO6, DO7 together** (`set_standard_digital_out` one after another in one `def` script).
They are **normally ON and OFF while gripping**.

| Step | Outputs | Wait |
| --- | --- | --- |
| Grip (after touching the product) | DO4~7 = OFF | `grip_wait_s` 0.5 s |
| Release (at the place pose) | DO4~7 = ON | `release_wait_s` 0.5 s |

* Settings: `gripper.suction_do: [4, 5, 6, 7]`, `gripper.grip_level: false` (output value while gripping). With null the
  grip / release steps are skipped.
* **[Grip (OFF)] / [Release (ON)]** in the gripper panel switch only the outputs (for testing, or to drop a product after
  stopping mid-cycle).
* The outputs are not changed at the start of a cycle. Start in the normal ON (released) state.

## Safety

| Setting (`config/picking.yaml`) | Default | Meaning |
| --- | --- | --- |
| `robot.velocity` | 0.05 m/s | default linear speed (changeable in the GUI) |
| `pick.final_offset_mm` | 0 | 20 ~ 50 mm recommended for the first tests |
| `safety.max_move_mm` | 800 | reject targets farther than this from the current TCP (catches unit / frame mistakes) |
| `safety.z_min_mm` | none | lower Z limit of the suction tip in the base frame. Measure the table / box bottom height |
| `safety.workspace_mm` | none | suction tip workspace `[[xmin,xmax],[ymin,ymax],[zmin,zmax]]` |
| `safety.max_tilt_deg` | 60 | reject if the suction direction tilts more than this from vertical |
| `safety.max_rotation_deg` | 90 | reject if the tool must rotate more than this from the reference pose |

## Vision node trigger (TODO)

`VisionClient.trigger()` in `vision_client.py` is only a placeholder. Plan:

* publish `vision.trigger_topic` (`/vision/trigger`, std_msgs/Empty)
* receive the product position (`position_m`) and surface normal on `vision.result_topic` (`/vision/result`) within
  `timeout_s` and fill the input fields

Once the vision interface is defined, only this class needs to be implemented.

## Tests

```bash
cd ~/pp_ws && python -m pytest src/piece_picking/test
```
