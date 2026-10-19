"""Device backends that report what Hailo accelerator the agent server is attached to."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from hailo_apps.python.core.common.defines import (
    HAILO8_ARCH,
    HAILO8_ARCH_CAPS,
    HAILO8L_ARCH,
    HAILO8L_ARCH_CAPS,
    HAILO10H_ARCH,
    HAILO10H_ARCH_CAPS,
    HAILO15H_ARCH_CAPS,
    HAILO_FW_CONTROL_CMD,
)
from hailo_apps.python.core.common.installation_utils import detect_host_arch

IDENTIFY_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class DeviceInfo:
    """Snapshot of the attached Hailo device.

    Attributes:
        arch: Hailo-Apps architecture name (``hailo8``, ``hailo8l``, ``hailo10h``), or None
            when no device could be identified.
        host_arch: Host platform as reported by ``detect_host_arch`` (``x86``, ``rpi``, ...).
        board_name: Board name reported by the firmware.
        firmware_version: Firmware version string.
        serial_number: Device serial number.
        product_name: Product name reported by the firmware.
        error: Why identification failed, when it did.
    """

    arch: str | None
    host_arch: str
    board_name: str | None = None
    firmware_version: str | None = None
    serial_number: str | None = None
    product_name: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the snapshot as a JSON-serialisable dict."""
        return asdict(self)


class DeviceBackend(Protocol):
    """Source of device information for the agent server."""

    def device_info(self) -> DeviceInfo:
        """Identify the attached device.

        Returns:
            The device snapshot; never raises for a missing or broken device.
        """
        ...


def parse_identify_output(stdout: str) -> dict[str, str]:
    """Parse ``hailortcli fw-control identify`` output into a key/value dict.

    Args:
        stdout: Raw command output.

    Returns:
        Mapping of each ``Key: Value`` line, e.g. ``{"Board Name": "Hailo-8", ...}``.
    """
    fields: dict[str, str] = {}
    for line in stdout.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() and value.strip():
            fields[key.strip()] = value.strip()
    return fields


def arch_from_identify(fields: dict[str, str]) -> str | None:
    """Map identify fields to a Hailo-Apps architecture name.

    Args:
        fields: Output of :func:`parse_identify_output`.

    Returns:
        The architecture name, or None if the device architecture is not recognised.
    """
    device_arch = fields.get("Device Architecture", "").upper()
    # HAILO8L must be tested before its prefix HAILO8.
    if HAILO8L_ARCH_CAPS in device_arch:
        return HAILO8L_ARCH
    if HAILO8_ARCH_CAPS in device_arch:
        return HAILO8_ARCH
    if HAILO10H_ARCH_CAPS in device_arch or HAILO15H_ARCH_CAPS in device_arch:
        return HAILO10H_ARCH
    return None


class HailoCliBackend:
    """Identifies the device through ``hailortcli``."""

    def __init__(self, timeout_s: float = IDENTIFY_TIMEOUT_S) -> None:
        """Create the backend.

        Args:
            timeout_s: Upper bound on the identify call, so a wedged driver cannot hang a tool.
        """
        self._timeout_s = timeout_s

    def device_info(self) -> DeviceInfo:
        """Identify the attached device with ``hailortcli fw-control identify``.

        Returns:
            The device snapshot, with ``error`` set when identification failed.
        """
        host_arch = detect_host_arch()
        try:
            res = subprocess.run(
                shlex.split(HAILO_FW_CONTROL_CMD),
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                check=False,
            )
        except FileNotFoundError:
            return DeviceInfo(arch=None, host_arch=host_arch, error="hailortcli not found")
        except subprocess.TimeoutExpired:
            return DeviceInfo(
                arch=None,
                host_arch=host_arch,
                error=f"hailortcli timed out after {self._timeout_s:g} s",
            )
        if res.returncode != 0:
            return DeviceInfo(
                arch=None,
                host_arch=host_arch,
                error=f"hailortcli exited with {res.returncode}: {res.stderr.strip()[-300:]}",
            )
        fields = parse_identify_output(res.stdout)
        arch = arch_from_identify(fields)
        return DeviceInfo(
            arch=arch,
            host_arch=host_arch,
            board_name=fields.get("Board Name"),
            firmware_version=fields.get("Firmware Version"),
            serial_number=fields.get("Serial Number"),
            product_name=fields.get("Product Name"),
            error=None if arch else "unrecognised device architecture",
        )


class MockBackend:
    """Pretends to be attached to a device; for tests and hardware-free demos."""

    def __init__(self, arch: str = HAILO10H_ARCH) -> None:
        """Create the backend.

        Args:
            arch: Architecture to report.
        """
        self._arch = arch

    def device_info(self) -> DeviceInfo:
        """Return a fixed snapshot for the configured architecture.

        Returns:
            The mock device snapshot.
        """
        return DeviceInfo(
            arch=self._arch,
            host_arch="mock",
            board_name=f"mock-{self._arch}",
            firmware_version="mock",
            serial_number="MOCK0000",
            product_name=f"Mock {self._arch} device",
        )
