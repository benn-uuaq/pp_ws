"""
camera_helios.py - LUCID Helios2 ToF camera (Arena SDK, arena_api).

Receives Coord3D_ABCY16 and builds per-pixel XYZ [m] and an intensity image.
Usage follows the Python examples in the Arena SDK docs (py_helios_min_and_max_depth,
py_helios_smooth_results, py_enumeration_unicast).

* Coordinates: X right, Y down, Z in front of the lens (camera frame).
* Invalid points have 65535 in the C channel; they become NaN.
* Unit conversion: value * Scan3dCoordinateScale + Scan3dCoordinateOffset  [mm]
* If an IP is given it is registered for unicast discovery. Needed when the camera is on a
  different subnet from the PC or broadcasts do not reach it (WSL etc.).

The arena_api wheel and the Arena SDK (Linux x64) must be downloaded from the LUCID website and installed.
"""

from __future__ import annotations

import ctypes
import threading
from typing import List, Optional

import numpy as np

from robot_camera_calibration.camera_base import FrameBuffer, intensity_to_bgr

PIXEL_FORMAT = "Coord3D_ABCY16"
INVALID_RAW = 65535
HELIOS_MODEL_PREFIXES = ("HL", "HT")     # HLT003S(Helios2), HTP003S(Helios2+), HTW003S ...


class HeliosCamera:
    def __init__(self, ip: str = "", operating_mode: str = "Distance3000mmSingleFreq",
                 exposure_time: str = "", image_accumulation: int = 1,
                 spatial_filter: bool = True, confidence_threshold: bool = True,
                 timeout_ms: int = 2000, packet_size: int = 1500, packet_delay_ns: int = 20000):
        self.ip = ip.strip()
        self.packet_size = int(packet_size)
        self.packet_delay_ns = int(packet_delay_ns)
        self.operating_mode = operating_mode
        self.exposure_time = exposure_time
        self.image_accumulation = int(image_accumulation)
        self.spatial_filter = bool(spatial_filter)
        self.confidence_threshold = bool(confidence_threshold)
        self.timeout_ms = int(timeout_ms)
        self.description = f"Helios ({self.ip or 'auto'})"
        self.last_error = ""
        self.warnings: List[str] = []
        self._system = None
        self._device = None
        self._scale = np.ones(3)
        self._offset = np.zeros(3)
        self._buf = FrameBuffer()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        try:
            from arena_api.system import system
        except ImportError as e:
            raise RuntimeError("arena_api is not installed. Install the LUCID Arena SDK (Linux x64) and "
                               "the arena_api wheel (see README).") from e
        self._system = system

        if self.ip:
            try:
                system.add_unicast_discovery_device(self.ip)
            except Exception as e:
                self.warnings.append(f"unicast discovery registration failed ({self.ip}): {e}")
        infos = system.device_infos
        if self.ip:
            infos = [i for i in infos if str(i.get("ip", "")) == self.ip]
        else:
            # the Helios2 Chroma also has a Triton RGB camera on the same LAN, so pick only Helios (HL*/HT*).
            infos = [i for i in infos if str(i.get("model", "")).upper().startswith(HELIOS_MODEL_PREFIXES)]
        if not infos:
            where = f"IP {self.ip}" if self.ip else "the network"
            raise RuntimeError(f"No Helios camera found on {where}. "
                               "Check power, cable and IP range.")
        info = infos[0]
        self._device = system.create_device(device_infos=[info])[0]
        nm = self._device.nodemap

        # if a program on another PC holds control, the camera opens read-only,
        # and setting changes / stream start fail with GC_ERR_RESOURCE_IN_USE.
        access = str(self._read(self._device.tl_device_nodemap, "DeviceAccessStatus") or "")
        if "ReadOnly" in access or access in ("NoAccess", "Busy"):
            owner = self._read(nm, "GevPrimaryApplicationIPAddress")
            owner_text = self._ip_text(owner) if owner else "unknown"
            self.stop()
            raise RuntimeError(f"Another program is using the camera, so it cannot be controlled "
                               f"(access {access}, PC in use {owner_text}). Close ArenaView or any other "
                               "camera program on that PC and connect again.")

        nm["PixelFormat"].value = PIXEL_FORMAT
        self._try_set(nm, "Scan3dOperatingMode", self.operating_mode)
        if self.exposure_time:
            self._try_set(nm, "ExposureTimeSelector", self.exposure_time)
        self._try_set(nm, "Scan3dImageAccumulation", self.image_accumulation)
        self._try_set(nm, "Scan3dSpatialFilterEnable", self.spatial_filter)
        self._try_set(nm, "Scan3dConfidenceThresholdEnable", self.confidence_threshold)

        for i, coord in enumerate(("CoordinateA", "CoordinateB", "CoordinateC")):
            nm["Scan3dCoordinateSelector"].value = coord
            self._scale[i] = float(nm["Scan3dCoordinateScale"].value)
            self._offset[i] = float(nm["Scan3dCoordinateOffset"].value)

        tl = self._device.tl_stream_nodemap
        tl["StreamBufferHandlingMode"].value = "NewestOnly"
        tl["StreamPacketResendEnable"].value = True
        # when packets arrive in bursts, the WSL (mirrored) path loses many and most frames break.
        # Fix the packet size and add a gap between packets (GevSCPD) (measured 2026-10-27: 1500 B,
        # 20 us gives about 17 fps with 0 broken frames). 0 = auto negotiation / no gap.
        if self.packet_size > 0:
            tl["StreamAutoNegotiatePacketSize"].value = False
            self._try_set(nm, "DeviceStreamChannelPacketSize", self.packet_size)
        else:
            tl["StreamAutoNegotiatePacketSize"].value = True
        self._try_set(nm, "DeviceLinkThroughputLimitMode", "Off")
        self._try_set(nm, "GevSCPD", self.packet_delay_ns)

        model = info.get("model", "Helios")
        serial = info.get("serial", "")
        self.description = (f"{model} {serial} ({info.get('ip', '')}), "
                            f"{nm['Scan3dOperatingMode'].value}")
        self._device.start_stream()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="helios", daemon=True)
        self._thread.start()

    @staticmethod
    def _read(nm, name: str):
        try:
            return nm[name].value
        except Exception:
            return None

    @staticmethod
    def _ip_text(value) -> str:
        v = int(value)
        return ".".join(str((v >> s) & 0xFF) for s in (24, 16, 8, 0))

    def _try_set(self, nm, name: str, value) -> None:
        """Some nodes do not exist on some camera models / firmware, so failures are only kept as warnings."""
        try:
            nm[name].value = value
        except Exception as e:
            self.warnings.append(f"failed to set {name} = {value}: {e}")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                buf = self._device.get_buffer(timeout=self.timeout_ms)
            except TimeoutError:
                # if control works but no images arrive, usually the stream UDP does not reach the PC
                # (WSL NAT mode, firewall). See README "Helios setup".
                self.last_error = "No frames arriving (stream UDP blocked? check WSL networking / firewall)"
                continue
            except Exception as e:
                self.last_error = f"Frame receive failed: {e}"
                continue
            try:
                if getattr(buf, "is_incomplete", False):
                    self.last_error = "Incomplete frame (packet loss)"
                    continue
                h, w = int(buf.height), int(buf.width)
                ptr = ctypes.cast(buf.pdata, ctypes.POINTER(ctypes.c_uint16))
                raw = np.ctypeslib.as_array(ptr, shape=(h * w * 4,)).reshape(h, w, 4).copy()
            finally:
                self._device.requeue_buffer(buf)
            image, xyz = self.convert(raw, self._scale, self._offset)
            self._buf.publish(image, xyz, raw[..., 3])
            self.last_error = ""

    @staticmethod
    def convert(raw: np.ndarray, scale: np.ndarray, offset: np.ndarray):
        """ABCY16 raw values (HxWx4 uint16) -> (BGR intensity image, XYZ [m])."""
        invalid = raw[..., 2] == INVALID_RAW
        xyz = (raw[..., :3].astype(np.float32) * scale.astype(np.float32)
               + offset.astype(np.float32)) / 1000.0
        xyz[invalid] = np.nan
        return intensity_to_bgr(raw[..., 3]), xyz

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._system is not None and self._device is not None:
            try:
                self._device.stop_stream()
            except Exception:
                pass
            try:
                self._system.destroy_device(self._device)
            except Exception:
                pass
        self._device = None

    def get_latest(self):
        return self._buf.latest()

    def wait_for_new_frame(self, after_seq: int, timeout: float = 2.0):
        return self._buf.wait_newer(after_seq, timeout)
