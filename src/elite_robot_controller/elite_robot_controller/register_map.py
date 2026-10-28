"""Reads the Modbus register map.

Addresses are not kept in code but managed in one place, config/modbus_registers.json.
Entries whose address is not decided yet have address null, and the corresponding function is
refused so nothing is written to a wrong register.
"""

from __future__ import annotations

import json
import os


DEFAULT_FILENAME = "modbus_registers.json"


class RegisterEntry:
    """One register entry. Unusable without an address."""

    __slots__ = ("name", "address", "count", "kind")

    def __init__(self, name: str, address: int | None, count: int, kind: str = "pose") -> None:
        self.name = name
        self.address = address
        self.count = count
        # for pose the first 3 are position and the last 3 rotation. For angle all 6 are rotations.
        self.kind = kind

    @property
    def available(self) -> bool:
        """Whether the address is set and the entry can actually be read / written."""
        return self.address is not None

    def __repr__(self) -> str:  # pragma: no cover - for debugging convenience.
        return f"RegisterEntry({self.name!r}, address={self.address}, count={self.count})"


class RegisterMap:
    """Holds the read / write register entries and the unit conversion factors."""

    def __init__(self, data: dict) -> None:
        scale = data.get("scale", {})
        self.position_scale = float(scale.get("position_per_count", 0.1))
        self.rotation_scale = float(scale.get("rotation_per_count", 1.0))
        self.read = self._entries(data.get("read", {}))
        self.write = self._entries(data.get("write", {}))

    @staticmethod
    def _entries(section: dict) -> dict[str, RegisterEntry]:
        entries: dict[str, RegisterEntry] = {}
        for name, value in section.items():
            if name.startswith("_") or not isinstance(value, dict):
                continue
            address = value.get("address")
            entries[name] = RegisterEntry(
                name,
                None if address is None else int(address),
                int(value.get("count", 1)),
                str(value.get("kind", "pose")),
            )
        return entries

    def read_entry(self, name: str) -> RegisterEntry:
        return self.read.get(name) or RegisterEntry(name, None, 1)

    def write_entry(self, name: str) -> RegisterEntry:
        return self.write.get(name) or RegisterEntry(name, None, 1)

    def scales_for(self, entry: "RegisterEntry") -> list[float]:
        """Return the per-component conversion factors of an entry in order."""
        if entry.kind == "raw":
            # values already stored as integers (mm, counts, ...). No sub-unit decimals.
            return [1.0] * entry.count
        if entry.kind == "angle":
            return [self.rotation_scale] * entry.count
        # pose splits into 3 positions and 3 rotations.
        half = entry.count // 2
        return [self.position_scale] * half + [self.rotation_scale] * (entry.count - half)

    def available_writes(self) -> list[str]:
        """Names of write entries with an address. Used to enable UI buttons."""
        return sorted(name for name, entry in self.write.items() if entry.available)


def default_path() -> str:
    """Find the config file path of the installed package.

    Falls back to the source tree path for running before a build or without ROS.
    """
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("elite_robot_controller")
        candidate = os.path.join(share, "config", DEFAULT_FILENAME)
        if os.path.isfile(candidate):
            return candidate
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "config", DEFAULT_FILENAME)


def load(path: str | None = None) -> RegisterMap:
    """Read the register map. Without a path the default location is used."""
    with open(path or default_path(), encoding="utf-8") as handle:
        return RegisterMap(json.load(handle))
