"""Device backends: identify the attached Hailo accelerator, run pipelines and the VLM on it."""

from __future__ import annotations

import re
import shlex
import subprocess
import tempfile
import threading
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from hailo_apps.config.config_manager import get_default_model_name
from hailo_apps.python.agent_server.pipelines import (
    MockPipeline,
    PipelineHandle,
    SubprocessPipeline,
    runner_command,
)
from hailo_apps.python.core.common.defines import (
    HAILO8_ARCH,
    HAILO8_ARCH_CAPS,
    HAILO8L_ARCH,
    HAILO8L_ARCH_CAPS,
    HAILO10H_ARCH,
    HAILO10H_ARCH_CAPS,
    HAILO15H_ARCH_CAPS,
    HAILO_FILE_EXTENSION,
    HAILO_FW_CONTROL_CMD,
    RESOURCES_MODELS_DIR_NAME,
    RESOURCES_ROOT_PATH_DEFAULT,
    VLM_CHAT_APP,
)
from hailo_apps.python.core.common.installation_utils import detect_host_arch

IDENTIFY_TIMEOUT_S = 10.0
CAMERA_LIST_TIMEOUT_S = 10.0
# "0 : ov5647 [2592x1944 10-bit GBRG] (/base/axi/...)" in `rpicam-hello --list-cameras` output.
_RPI_CAMERA_LINE = re.compile(r"^\s*(\d+)\s*:\s*(\S+)")
# The first request also loads the VLM onto the device, which takes far longer than a caption.
VLM_LOAD_TIMEOUT_S = 180
VLM_TIMEOUT_S = 60
VLM_MAX_TOKENS = 150
VLM_SYSTEM_PROMPT = (
    "You describe camera frames for a monitoring agent. Answer in two or three factual "
    "sentences about what is visible; do not speculate beyond the image."
)
# vlm_chat's Backend reports failures as answers with these prefixes instead of raising.
_VLM_FAILURE_PREFIXES = ("Error:", "Request timed out", "Queue error")


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

    def cameras(self) -> dict[str, list[str]]:
        """List the cameras a pipeline could open.

        Returns:
            ``{"usb": [...], "rpi": [...]}``: USB capture devices and Raspberry Pi CSI cameras.
        """
        ...

    def start_pipeline(
        self, app: str, source: str, labels: tuple[str, ...], fps: int | None
    ) -> PipelineHandle:
        """Start a pipeline app.

        Args:
            app: Key of ``PIPELINE_TARGETS``.
            source: Validated ``--input`` value.
            labels: Label filter recorded on the handle.
            fps: Requested frame rate, or None for the app default.

        Returns:
            The running pipeline.
        """
        ...

    def describe(self, image: np.ndarray, prompt: str, objects: Iterable[dict[str, Any]]) -> str:
        """Caption an image with the on-device VLM.

        Args:
            image: Frame in BGR.
            prompt: What to ask about the image.
            objects: Detections in the frame; only the mock uses them.

        Returns:
            The VLM's answer.

        Raises:
            RuntimeError: If the VLM is unavailable or the request fails.
        """
        ...

    def close(self) -> None:
        """Release device resources held by the backend, such as a loaded VLM."""
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


def parse_rpi_camera_list(stdout: str) -> list[str]:
    """Extract camera sensor names from ``rpicam-hello --list-cameras`` output.

    Args:
        stdout: Command output.

    Returns:
        One sensor name per detected camera, e.g. ``["ov5647"]``.
    """
    return [m.group(2) for line in stdout.splitlines() if (m := _RPI_CAMERA_LINE.match(line))]


def list_rpi_cameras(timeout_s: float = CAMERA_LIST_TIMEOUT_S) -> list[str]:
    """List Raspberry Pi CSI cameras without opening them.

    Uses ``rpicam-hello --list-cameras`` rather than Hailo-Apps' ``is_rpi_camera_available``,
    which streams from the camera for seconds and would collide with a running pipeline.

    Args:
        timeout_s: Upper bound on the listing call.

    Returns:
        Sensor names; empty when the tool is missing (not a Pi) or the call fails.
    """
    try:
        res = subprocess.run(
            ["rpicam-hello", "--list-cameras"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return parse_rpi_camera_list(res.stdout + res.stderr)


def installed_vlm_hef(models_root: Path | None = None) -> Path:
    """Locate the installed default VLM model without downloading it.

    ``resolve_hef_path`` would fetch a missing model, which for the VLM is several GB; an agent
    tool must not start that on a device that may not have the space.

    Args:
        models_root: Directory holding per-architecture model folders; the Hailo-Apps
            resources ``models`` directory when None.

    Returns:
        Path of the installed HEF.

    Raises:
        RuntimeError: If no default VLM is configured or its HEF is not installed.
    """
    name = get_default_model_name(VLM_CHAT_APP, HAILO10H_ARCH)
    if name is None:
        raise RuntimeError("no default VLM model is configured for hailo10h")
    root = models_root or Path(RESOURCES_ROOT_PATH_DEFAULT) / RESOURCES_MODELS_DIR_NAME
    hef_path = root / HAILO10H_ARCH / f"{name}{HAILO_FILE_EXTENSION}"
    if not hef_path.is_file():
        raise RuntimeError(
            f"VLM model {name} is not installed at {hef_path}; install it with "
            f"hailo-download-resources before using describe_scene"
        )
    return hef_path


class HailoCliBackend:
    """Identifies the device through ``hailortcli`` and runs pipelines as child processes."""

    def __init__(self, timeout_s: float = IDENTIFY_TIMEOUT_S) -> None:
        """Create the backend.

        Args:
            timeout_s: Upper bound on the identify call, so a wedged driver cannot hang a tool.
        """
        self._timeout_s = timeout_s
        self._vlm: Any = None
        self._vlm_lock = threading.Lock()

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

    def cameras(self) -> dict[str, list[str]]:
        """List connected USB and Raspberry Pi CSI cameras.

        Returns:
            ``{"usb": [...], "rpi": [...]}``.
        """
        from hailo_apps.python.core.common.camera_utils import get_usb_video_devices

        return {
            "usb": [str(dev) for dev in get_usb_video_devices()],
            "rpi": list_rpi_cameras(),
        }

    def start_pipeline(
        self, app: str, source: str, labels: tuple[str, ...], fps: int | None
    ) -> PipelineHandle:
        """Run the app in a child process through ``pipeline_runner``.

        Args:
            app: Key of ``PIPELINE_TARGETS``.
            source: Validated ``--input`` value.
            labels: Label filter recorded on the handle.
            fps: Requested frame rate, or None for the app default.

        Returns:
            The running pipeline.

        Raises:
            ValueError: If ``source`` is a video file that is not installed.
        """
        if Path(source).is_absolute() and not Path(source).is_file():
            raise ValueError(f"video not found: {source} (run hailo-download-resources)")
        frame_dir = Path(tempfile.mkdtemp(prefix=f"hailo-agent-{app}-"))
        return SubprocessPipeline(
            app=app,
            source=source,
            labels=labels,
            command=runner_command(app, source, fps, frame_dir),
            frame_dir=frame_dir,
        ).start()

    def describe(self, image: np.ndarray, prompt: str, objects: Iterable[dict[str, Any]]) -> str:
        """Caption an image with ``vlm_chat``'s VLM, loading it on the first call.

        Requests are serialised: the VLM holds one conversation at a time.

        Args:
            image: Frame in BGR.
            prompt: What to ask about the image.
            objects: Unused; the VLM looks at the pixels.

        Returns:
            The VLM's answer.

        Raises:
            RuntimeError: If the VLM model is missing or the request fails or times out.
        """
        with self._vlm_lock:
            timeout = VLM_TIMEOUT_S
            if self._vlm is None:
                self._vlm = self._load_vlm()
                timeout = VLM_LOAD_TIMEOUT_S
            answer = str(self._vlm.vlm_inference(image, prompt, timeout=timeout)["answer"])
        if answer.startswith(_VLM_FAILURE_PREFIXES):
            raise RuntimeError(f"VLM request failed: {answer}")
        return answer

    @staticmethod
    def _load_vlm() -> Any:
        from hailo_apps.python.gen_ai_apps.vlm_chat.backend import Backend

        hef_path = installed_vlm_hef()
        return Backend(
            hef_path=str(hef_path), max_tokens=VLM_MAX_TOKENS, system_prompt=VLM_SYSTEM_PROMPT
        )

    def close(self) -> None:
        """Shut down the VLM worker if it was started."""
        with self._vlm_lock:
            if self._vlm is not None:
                self._vlm.close()
                self._vlm = None


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

    def cameras(self) -> dict[str, list[str]]:
        """Report one fake CSI camera so camera sources can be exercised.

        Returns:
            ``{"usb": [], "rpi": ["mock"]}``.
        """
        return {"usb": [], "rpi": ["mock"]}

    def start_pipeline(
        self, app: str, source: str, labels: tuple[str, ...], fps: int | None
    ) -> PipelineHandle:
        """Start a synthetic scene in place of the app.

        Args:
            app: Key of ``PIPELINE_TARGETS``.
            source: Validated ``--input`` value; recorded but not opened.
            labels: Label filter recorded on the handle.
            fps: Frame rate of the synthetic scene, or None for 10.

        Returns:
            The running mock pipeline.
        """
        return MockPipeline(app=app, source=source, labels=labels, fps=fps or 10).start()

    def describe(self, image: np.ndarray, prompt: str, objects: Iterable[dict[str, Any]]) -> str:
        """Build a caption from the frame's detections instead of running a VLM.

        Args:
            image: Unused.
            prompt: Echoed back so callers can see what was asked.
            objects: Detections in the frame.

        Returns:
            A deterministic caption naming the objects.
        """
        counts = Counter(obj["label"] for obj in objects)
        seen = ", ".join(
            f"{n} {label}{'s' if n > 1 else ''}" for label, n in sorted(counts.items())
        )
        return f"[mock VLM] Asked: {prompt!r}. A flat synthetic scene showing {seen or 'nothing'}."

    def close(self) -> None:
        """Nothing to release."""
