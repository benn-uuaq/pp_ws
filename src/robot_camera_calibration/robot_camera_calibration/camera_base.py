"""
camera_base.py - common frame format for cameras and a latest-frame buffer.

Each camera has a background thread holding the latest frame; the screen (live view) and the
calibration job read the same buffer. A camera object provides:

    start(), stop(), get_latest(), wait_for_new_frame(after_seq, timeout),
    description, last_error
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2 as cv
import numpy as np


@dataclass
class Frame:
    image: np.ndarray        # BGR uint8 (HxWx3) for display / marker detection
    xyz: Optional[np.ndarray]  # per-pixel 3D points [m], camera frame, invalid = NaN (HxWx3 float32). None for the RGB camera
    seq: int                 # increases by 1 for every new frame
    timestamp: float         # time.time() when received
    intensity: Optional[np.ndarray] = None   # raw intensity (uint16, HxW). Used for marker detection


class FrameBuffer:
    """Holds the latest frame and lets callers wait for a newer one."""

    def __init__(self):
        self._cond = threading.Condition()
        self._latest: Optional[Frame] = None
        self._seq = 0

    def publish(self, image: np.ndarray, xyz: Optional[np.ndarray],
                intensity: Optional[np.ndarray] = None) -> None:
        with self._cond:
            self._seq += 1
            self._latest = Frame(image, xyz, self._seq, time.time(), intensity)
            self._cond.notify_all()

    def latest(self) -> Optional[Frame]:
        with self._cond:
            return self._latest

    def wait_newer(self, after_seq: int, timeout: float) -> Optional[Frame]:
        deadline = time.time() + timeout
        with self._cond:
            while self._latest is None or self._latest.seq <= after_seq:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._latest


def intensity_to_gray(intensity: np.ndarray, percentile: float = 99.5, gamma: float = 1.0) -> np.ndarray:
    """16-bit intensity -> 8-bit. The brightest (100 - percentile) % saturates; gamma < 1 lifts the dark side."""
    sample = intensity[::4, ::4]
    sample = sample[sample > 0]
    top = float(np.percentile(sample, percentile)) if sample.size else 1.0
    x = np.clip(intensity.astype(np.float32) / max(top, 1.0), 0.0, 1.0)
    if gamma != 1.0:
        x = np.power(x, gamma)
    return (x * 255.0).astype(np.uint8)


def intensity_to_bgr(intensity: np.ndarray) -> np.ndarray:
    """For display. Compressed with a near-maximum reference + gamma (0.5) so the very bright centre (ToF head-on
    reflection) does not saturate. Marker detection is done separately on the raw intensity by calib_core.MarkerDetector."""
    return cv.cvtColor(intensity_to_gray(intensity, 99.95, 0.5), cv.COLOR_GRAY2BGR)
