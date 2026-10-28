"""
camera_triton.py - LUCID Triton RGB camera (the RGB side of the Helios2 Chroma, Arena SDK).

* Image: received as BayerRG8 and converted to BGR (1/3 of the network load of RGB8).
* Intrinsics: the Helios2 Chroma stores the factory calibration in camera nodes
  (CalibFocalLengthX/Y, CalibOpticalCenterX/Y, CalibLensDistortionValue0..7 —
  same approach as the arena_api example py_helios_chroma_overlay.py). If intrinsics_file is given,
  that file (OpenCV FileStorage: cameraMatrix, distCoeffs) is used instead.
* There is no depth, so Frame.xyz is None.
"""

from __future__ import annotations

import ctypes
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

from robot_camera_calibration.camera_base import FrameBuffer

TRITON_MODEL_PREFIXES = ("TRI",)
# GenICam pixel format -> OpenCV conversion code. OpenCV's Bayer names are offset from GenICam
# (GenICam BayerRG = R at top-left  ->  OpenCV COLOR_BayerBG2BGR).
BAYER_CODES = {"BayerRG8": cv.COLOR_BayerBG2BGR, "BayerGR8": cv.COLOR_BayerGB2BGR,
               "BayerBG8": cv.COLOR_BayerRG2BGR, "BayerGB8": cv.COLOR_BayerGR2BGR}


def load_intrinsics_file(path) -> Tuple[np.ndarray, np.ndarray]:
    """cameraMatrix / distCoeffs from an OpenCV FileStorage (.yml) (LUCID tritoncalibration.yml format)."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {path}")
    fs = cv.FileStorage(str(path), cv.FileStorage_READ)
    try:
        K = fs.getNode("cameraMatrix").mat()
        D = fs.getNode("distCoeffs").mat()
    finally:
        fs.release()
    if K is None:
        raise ValueError(f"No cameraMatrix in: {path}")
    return np.asarray(K, float).reshape(3, 3), (np.zeros(5) if D is None else np.asarray(D, float).reshape(-1))


class TritonCamera:
    def __init__(self, ip: str = "", pixel_format: str = "BayerRG8", intrinsics_file: str = "",
                 timeout_ms: int = 3000, packet_size: int = 1500, packet_delay_ns: int = 20000):
        self.ip = ip.strip()
        self.pixel_format = pixel_format
        self.intrinsics_file = intrinsics_file
        self.timeout_ms = int(timeout_ms)
        self.packet_size = int(packet_size)
        self.packet_delay_ns = int(packet_delay_ns)
        self.description = f"Triton ({self.ip or 'auto'})"
        self.last_error = ""
        self.warnings: List[str] = []
        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self.intrinsics_source = ""
        self._system = None
        self._device = None
        self._buf = FrameBuffer()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        try:
            from arena_api.system import system
        except ImportError as e:
            raise RuntimeError("arena_api is not installed (see README).") from e
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
            infos = [i for i in infos if str(i.get("model", "")).upper().startswith(TRITON_MODEL_PREFIXES)]
        if not infos:
            where = f"IP {self.ip}" if self.ip else "the network"
            raise RuntimeError(f"No Triton RGB camera found on {where}.")
        info = infos[0]
        self._device = system.create_device(device_infos=[info])[0]
        nm = self._device.nodemap

        access = str(self._read(self._device.tl_device_nodemap, "DeviceAccessStatus") or "")
        if "ReadOnly" in access or access in ("NoAccess", "Busy"):
            self.stop()
            raise RuntimeError(f"Another program is using the camera, so it cannot be controlled (access {access}). "
                               "Close ArenaView or any other camera program and connect again.")

        self._load_intrinsics(nm)
        if self.pixel_format not in BAYER_CODES and self.pixel_format not in ("BGR8", "RGB8", "Mono8"):
            raise ValueError(f"Unsupported pixel format: {self.pixel_format}")
        nm["PixelFormat"].value = self.pixel_format

        tl = self._device.tl_stream_nodemap
        tl["StreamBufferHandlingMode"].value = "NewestOnly"
        tl["StreamPacketResendEnable"].value = True
        if self.packet_size > 0:
            tl["StreamAutoNegotiatePacketSize"].value = False
            self._try_set(nm, "DeviceStreamChannelPacketSize", self.packet_size)
        else:
            tl["StreamAutoNegotiatePacketSize"].value = True
        self._try_set(nm, "DeviceLinkThroughputLimitMode", "Off")
        self._try_set(nm, "GevSCPD", self.packet_delay_ns)

        w, h = self._read(nm, "Width"), self._read(nm, "Height")
        self.description = f"{info.get('model', 'Triton')} {info.get('serial', '')} ({info.get('ip', '')}), {w}x{h}"
        self._device.start_stream()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="triton", daemon=True)
        self._thread.start()

    def _load_intrinsics(self, nm) -> None:
        if self.intrinsics_file:
            self.K, self.dist = load_intrinsics_file(self.intrinsics_file)
            self.intrinsics_source = f"file {Path(self.intrinsics_file).expanduser()}"
            return
        fx, fy = self._read(nm, "CalibFocalLengthX"), self._read(nm, "CalibFocalLengthY")
        cx, cy = self._read(nm, "CalibOpticalCenterX"), self._read(nm, "CalibOpticalCenterY")
        if None in (fx, fy, cx, cy) or float(fx) <= 0:
            raise RuntimeError("The camera has no factory-calibrated intrinsics (CalibFocalLength etc.). "
                               "Set an intrinsics file in camera_rgb.intrinsics_file.")
        dist = []
        for i in range(8):
            try:
                nm["CalibLensDistortionValueSelector"].value = f"Value{i}"
                dist.append(float(nm["CalibLensDistortionValue"].value))
            except Exception:
                break
        self.K = np.array([[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]])
        self.dist = np.array(dist if dist else [0.0] * 5)
        model = self._read(nm, "CalibLensDistortionModel")
        self.intrinsics_source = f"camera factory calibration ({model or 'distortion model ?'}, {len(dist)} coefficients)"

    @staticmethod
    def _read(nm, name: str):
        try:
            return nm[name].value
        except Exception:
            return None

    def _try_set(self, nm, name: str, value) -> None:
        try:
            nm[name].value = value
        except Exception as e:
            self.warnings.append(f"failed to set {name} = {value}: {e}")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                buf = self._device.get_buffer(timeout=self.timeout_ms)
            except TimeoutError:
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
                channels = 1 if (self.pixel_format in BAYER_CODES or self.pixel_format == "Mono8") else 3
                ptr = ctypes.cast(buf.pdata, ctypes.POINTER(ctypes.c_uint8))
                raw = np.ctypeslib.as_array(ptr, shape=(h * w * channels,)).reshape(h, w, channels).copy()
            finally:
                self._device.requeue_buffer(buf)
            self._buf.publish(self.convert(raw, self.pixel_format), None)
            self.last_error = ""

    @staticmethod
    def convert(raw: np.ndarray, pixel_format: str) -> np.ndarray:
        """Raw buffer -> BGR uint8."""
        if pixel_format in BAYER_CODES:
            return cv.cvtColor(raw.reshape(raw.shape[0], raw.shape[1]), BAYER_CODES[pixel_format])
        if pixel_format == "RGB8":
            return cv.cvtColor(raw, cv.COLOR_RGB2BGR)
        if pixel_format == "Mono8":
            return cv.cvtColor(raw.reshape(raw.shape[0], raw.shape[1]), cv.COLOR_GRAY2BGR)
        return raw

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
