# lucid_helios (not a ROS package)

Place for the LUCID Arena SDK and the `arena_api` Python wheel used by `robot_camera_calibration`
(Helios2+ Chroma: Helios ToF + Triton RGB). `COLCON_IGNORE` keeps colcon out of this folder.

The SDK and the wheel are **not in this repository** (LUCID licence, several GB). Download them from the LUCID Vision
Labs website (Arena SDK for Linux x64 and the Arena Python package):

| File (version used) | Purpose |
| --- | --- |
| `ArenaSDK_AVMP_v_1.0.10.11_Linux_x64.tar.gz` | Arena SDK (C/C++ libraries, GenTL) |
| `arena_api-2.9.5-py3-none-any.zip` | Python wheel `arena_api` |

## Install (Ubuntu 22.04 / WSL)

```bash
cd ~/pp_ws/src/lucid_helios
tar -xzf ArenaSDK_AVMP_v_1.0.10.11_Linux_x64.tar.gz          # -> ArenaSDK_Linux_x64/
cd ArenaSDK_Linux_x64 && sudo sh Arena_SDK_Linux_x64_AVMP.conf   # registers the library path
cd ~/pp_ws/src/lucid_helios
unzip arena_api-2.9.5-py3-none-any.zip -d arena_api_pkg
cd ~/pp_ws && pip install src/lucid_helios/arena_api_pkg/arena_api-2.9.5-py3-none-any.whl   # into .venv (direnv)
```

GigE receive buffers (recommended by LUCID), in `/etc/sysctl.conf`:

```
net.core.rmem_default=33554432
net.core.rmem_max=33554432
```

If the SDK folder is moved, run `sudo sh Arena_SDK_Linux_x64_AVMP.conf -r` again in its new location.

## WSL networking

The camera streams images to the PC over UDP, so WSL must use **mirrored** networking and the Hyper-V firewall must
allow inbound UDP from the camera subnet. See "When the camera does not connect" in
`src/robot_camera_calibration/README.md`.

Check that the cameras are found:

```bash
python3 -c "from arena_api.system import system; [print(i['model'], i['serial'], i['ip']) for i in system.device_infos]"
```
