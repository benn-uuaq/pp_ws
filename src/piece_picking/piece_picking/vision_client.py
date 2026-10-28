"""
vision_client.py - receiving vision results: reading pasted JSON + vision node trigger (TODO: not implemented yet).

Paste: reads one entry, several entries or the whole vision output (candidates.json), or just the "position_m" / "normal"
lines (fragments with trailing commas etc. are fine). Long decimals are cut to 0.01 mm for position and 1e-6 for the normal.

Trigger: the [Capture with camera] button of the picking GUI calls trigger(). Integration plan (to change once the vision interface is defined)
  * publish the ROS 2 topic trigger_topic (std_msgs/Empty)
  * receive one product position (position_m) and surface normal (direction in the camera frame) on result_topic within timeout_s
    (current vision output: position_m, normal, uv ... in candidates.json)
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

POSITION_DECIMALS = 5          # [m] -> 0.01 mm
NORMAL_DECIMALS = 6
MAX_POSITION_M = 5.0           # above this the unit (m) is assumed to be wrong


@dataclass
class VisionResult:
    position_m: np.ndarray          # product position [m], camera frame
    frame: Optional[str] = None     # "rgb" | "tof" | None (unknown — pasted)
    stamp: float = field(default_factory=time.time)
    info: str = ""                  # extra info from the vision system (product id, rank etc.)
    normal: Optional[np.ndarray] = None         # surface normal (direction in the camera frame)
    rank: Optional[int] = None


# ======================================================================
# Pasted JSON
# ======================================================================
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _load_json_loose(text: str):
    """Also reads JSON fragments: trailing commas, several objects separated by commas, "key": value lines without braces."""
    s = re.sub(r",\s*([}\]])", r"\1", text.strip()).rstrip(",").strip()
    for candidate in (s, f"[{s}]", "{" + s + "}"):
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return None


def _candidates_from(obj) -> List[dict]:
    """All dicts that have position_m (in order of appearance)."""
    if isinstance(obj, dict):
        if "position_m" in obj:
            return [obj]
        obj = list(obj.values())
    if isinstance(obj, list):
        return [c for v in obj for c in _candidates_from(v)]
    return []


def _candidate_from_text(text: str) -> Optional[dict]:
    """Fragments that are not valid JSON: find "position_m": [...] and "normal": [...] as text."""
    def array(key):
        m = re.search(r'(?<!\w)"?' + key + r'"?\s*:\s*\[([^\]]*)\]', text)
        return None if m is None else re.findall(_NUMBER, m.group(1))
    pos = array("position_m")
    if pos is None:
        return None
    out = {"position_m": pos}
    normal = array("normal")
    if normal is not None:
        out["normal"] = normal
    return out


def _vector(values, name: str) -> np.ndarray:
    try:
        v = [float(x) for x in values]
    except (TypeError, ValueError):
        raise ValueError(f"Cannot read the {name} values as numbers: {values}")
    if len(v) != 3 or not all(math.isfinite(x) for x in v):
        raise ValueError(f"{name} must be three numbers: {values}")
    return np.array(v)


def parse_vision_candidates(text: str) -> List[VisionResult]:
    """Pasted vision result -> list of candidates (in order). Empty list if none found, ValueError for bad values."""
    data = _load_json_loose(text)
    raw = _candidates_from(data) if data is not None else []
    if not raw:
        one = _candidate_from_text(text)
        raw = [one] if one is not None else []
    results = []
    for c in raw:
        p = np.round(_vector(c["position_m"], "position_m"), POSITION_DECIMALS)
        if np.max(np.abs(p)) > MAX_POSITION_M:
            raise ValueError(f"position_m {p.tolist()} exceeds {MAX_POSITION_M:g} m. Check that the values are in m.")
        if p[2] <= 0.0:
            raise ValueError(f"The Z of position_m ({p[2]:g} m) is 0 or less (not in front of the camera).")
        n = c.get("normal")
        n = None if n is None else np.round(_vector(n, "normal"), NORMAL_DECIMALS)
        rank = c.get("rank")
        rank = int(rank) if isinstance(rank, (int, float)) and not isinstance(rank, bool) else None
        info = ", ".join(f"{k} {c[k]}" for k in ("object_id", "rank") if k in c)
        results.append(VisionResult(p, info=info, normal=n, rank=rank))
    return results


# ======================================================================
# Vision node trigger (TODO)
# ======================================================================
class VisionClient:
    implemented = False

    def __init__(self, trigger_topic: str = "/vision/trigger", result_topic: str = "/vision/result",
                 timeout_s: float = 5.0):
        self.trigger_topic = trigger_topic
        self.result_topic = result_topic
        self.timeout_s = float(timeout_s)

    def describe(self) -> str:
        return f"trigger {self.trigger_topic} -> result {self.result_topic} (timeout {self.timeout_s:g} s)"

    def trigger(self) -> Optional[VisionResult]:
        # TODO: vision node integration. Publish trigger_topic from an rclpy node, then wait for result_topic.
        raise NotImplementedError("The vision node trigger is not implemented yet (TODO). "
                                  "For now enter the product position with 'Manual entry' or [Paste vision JSON].")
