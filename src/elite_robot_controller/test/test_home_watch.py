"""While the task is not running, the home flag (276) is decided from the actual joints.

While the task runs, the task writes 276 every control cycle, but when it does not run nobody
writes it. Then right after RCS starts (register 0) or after moving the robot by hand with the pendant the value
differs from reality, and even with the robot at home ERUT sees deployed (if-0.5 evt/home).
"""

from test_pose_publishing import make_node


HOME = [0.1, -1.2, 1.5, -0.3, 1.57, 0.0]


class FakeDash:
    def __init__(self, value):
        self.value = value
        self.asked = 0

    def get_variable(self, name):
        self.asked += 1
        return self.value


class FakeParam:
    def __init__(self, value):
        self.value = value


def _node(joints_rad, flag, task_state=3, home=HOME, pose_src=0, regs_310=None,
          home_src=0, regs_322=None):
    regs = {276: [flag], 308: [pose_src], 328: [home_src]}
    if regs_310 is not None:
        regs[310] = regs_310
    if regs_322 is not None:
        regs[322] = regs_322
    node = make_node(regs, {
        'scale': {'position_per_count': 0.1, 'rotation_per_count': 1.0},
        'read': {'home_flag': {'address': 276, 'count': 1}},
        'write': {'home_flag': {'address': 276, 'count': 1},
                  'pose_src': {'address': 308, 'count': 1},
                  'standby_joint': {'address': 310, 'count': 6, 'kind': 'angle'},
                  'home_joint': {'address': 322, 'count': 6, 'kind': 'angle'},
                  'home_src': {'address': 328, 'count': 1}},
    })
    node.robot_dash = FakeDash(home)
    params = {'home_tol_j': 0.02, 'home_leave_tol_j': 0.05, 'home_refresh_s': 60.0}
    node.get_parameter = lambda name: FakeParam(params[name])
    node._last_codes = {'task_state': task_state}
    node._last_joint_mrad = [round(q * 1000) for q in joints_rad]
    node._joint_variables = {}
    return node


def test_robot_at_home_raises_a_stale_zero_flag():
    node = _node(HOME, flag=0)

    node._watch_home()

    assert node.robot_modbus.writes == [(276, 1)]


def test_robot_away_from_home_clears_a_stale_one():
    away = list(HOME)
    away[1] += 0.3
    node = _node(away, flag=1)

    node._watch_home()

    assert node.robot_modbus.writes == [(276, 0)]


def test_within_tolerance_counts_as_home():
    near = [q + 0.015 for q in HOME]
    node = _node(near, flag=1)

    node._watch_home()

    assert node.robot_modbus.writes == []


def test_running_task_owns_the_flag():
    """While the task runs, the task writes it — the node does not interfere."""
    node = _node(HOME, flag=0, task_state=1)

    node._watch_home()

    assert node.robot_modbus.writes == []


def test_unknown_home_joint_leaves_the_flag_alone():
    node = _node(HOME, flag=0, home='NOT_FOUND')

    node._watch_home()

    assert node.robot_modbus.writes == []


def test_saved_home_registers_win_when_home_src_is_set():
    """With 328 = 1 the home joints 322~327 are used like the task (RCS "Save home pose")."""
    other = [0.5, -1.0, 1.0, 0.0, 1.0, 0.0]
    node = _node(other, flag=0, home_src=1, regs_322=[round(q * 1000) for q in other])

    node._watch_home()

    assert node.robot_modbus.writes == [(276, 1)]
    assert node.robot_dash.asked == 0


def test_standby_registers_are_not_the_home():
    """310~315 (308) is the scan standby pose since 2026-10-16 — not used for the home check."""
    standby = [0.5, -1.0, 1.0, 0.0, 1.0, 0.0]
    node = _node(standby, flag=1, pose_src=1, regs_310=[round(q * 1000) for q in standby])

    node._watch_home()

    assert node.robot_modbus.writes == [(276, 0)]


def test_empty_saved_home_falls_back_to_the_robot_variable():
    """Even with 328 = 1, if 322~327 are all 0 nothing was ever saved."""
    node = _node(HOME, flag=0, home_src=1, regs_322=[0] * 6)

    node._watch_home()

    assert node.robot_modbus.writes == [(276, 1)]
    assert node.robot_dash.asked == 1


def test_home_joint_is_not_asked_every_cycle():
    node = _node(HOME, flag=1)
    for _ in range(5):
        node._watch_home()

    assert node.robot_dash.asked == 1


# ---- home move (cb_move_home): via the scan standby pose (2026-10-16) ------------------
STANDBY = [0.79941, -1.55029, -2.70026, 1.10839, 0.81203, -3.14139]


class FakeVars:
    """Robot variables by name. Unknown names return a string error like the robot."""

    def __init__(self, values):
        self.values = values
        self.last_error = ""

    def get_variable(self, name):
        return self.values.get(name, "NOT_FOUND")

    def robot_stop(self):
        return "Stopped"


class FakePrimary:
    def __init__(self):
        self.scripts = []

    def send_script(self, script):
        self.scripts.append(script)
        return True


class FakeResponse:
    success = None
    message = ""


def _home_script(variables, **regs):
    node = _node(HOME, flag=1, **regs)
    node.robot_dash = FakeVars(variables)
    node.robot_primary = FakePrimary()
    node.connected = True
    node._HOME_AFTER_STOP_S = 0
    node.cb_move_home(None, FakeResponse())
    assert len(node.robot_primary.scripts) == 1
    return node, node.robot_primary.scripts[0]


def test_move_home_goes_through_the_standby_then_home():
    node, script = _home_script({"Home_joint": HOME, "Standby_joint": STANDBY,
                                 "probe_dist": 100.0})
    assert f"mh_home = {HOME}" in script
    assert f"mh_sb = {STANDBY}" in script
    order = [script.index("movel(pose_trans(mh_cur"), script.index("movel(mh_up"),
             script.index("movej(mh_sb"), script.index("movej(mh_home"),
             script.index("write_port_register(276, 1)")]
    assert order == sorted(order)
    # after the retreat, line up with the standby pose — back the rest of the way in X, raise or lower the height.
    assert "mh_up[0] = mh_ref[0]" in script
    assert "mh_up[2] < mh_ref[2] - 0.001 or mh_up[2] > mh_ref[2] + 0.001" in script
    assert script.index("mh_up[0] = mh_ref[0]") < script.index("mh_up[2] = mh_ref[2]")
    # retreat by the smaller of the standby X and probe_dist, at the flange.
    assert "get_forward_kin(mh_sb)" in script and "get_actual_tool_flange_pose()" in script
    assert "set_tcp([0, 0, 0, 0, 0, 0])" in script and "0.1):" in script
    assert node.robot_modbus.writes[0] == (276, 0), "not home while moving"


def test_move_home_uses_saved_registers_for_home_and_standby():
    home = [0.5, -1.0, 1.0, 0.0, 1.0, 0.0]
    standby = [0.6, -1.1, 1.1, 0.1, 1.1, 0.1]
    _node_, script = _home_script(
        {"Home_joint": HOME, "Standby_joint": STANDBY, "probe_dist": 100.0},
        home_src=1, regs_322=[round(q * 1000) for q in home],
        pose_src=1, regs_310=[round(q * 1000) for q in standby])
    assert f"mh_home = {home}" in script
    assert f"mh_sb = {standby}" in script


def test_move_home_without_a_standby_variable_uses_the_home():
    """If the task is not loaded yet and there is no Standby_joint, home is used as the standby pose."""
    _node_, script = _home_script({"Home_joint": HOME, "probe_dist": 100.0})
    assert f"mh_sb = {HOME}" in script


# ---- sway while moving (2026-10-19) --------------------------------------------------
def _shifted(delta):
    return [q + delta for q in HOME]


def test_small_shake_keeps_the_home_flag():
    """Once home, sway within home_leave_tol_j (0.05 rad) still counts as home."""
    node = _node(_shifted(0.03), flag=1)
    node._watch_home()
    assert node.robot_modbus.writes == []


def test_leaving_beyond_the_wider_tolerance_clears_the_flag():
    node = _node(_shifted(0.06), flag=1)
    node._watch_home()
    assert node.robot_modbus.writes == [(276, 0)]


def test_entering_home_still_needs_the_tight_tolerance():
    """Entering from outside home still requires being within home_tol_j (0.02 rad) as before."""
    node = _node(_shifted(0.03), flag=0)
    node._watch_home()
    assert node.robot_modbus.writes == []
