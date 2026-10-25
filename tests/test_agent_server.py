"""Tests for hailo-agent-server, the MCP server that exposes a Hailo device to remote agents.

Contract tests (``sanity``) run the server against the mock backend and need no hardware.
The ``requires_device`` smoke test runs one ``watch_scene`` against the attached device.
"""

import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import anyio
import pytest

pytest.importorskip("mcp")

from mcp.server.mcpserver.exceptions import ToolError

from hailo_apps.python.agent_server import backend as backend_mod, pipelines
from hailo_apps.python.agent_server.backend import (
    HailoCliBackend,
    MockBackend,
    arch_from_identify,
    parse_identify_output,
)
from hailo_apps.python.agent_server.pipelines import SubprocessPipeline
from hailo_apps.python.agent_server.server import BearerTokenMiddleware, build_server

IDENTIFY_H8 = """\
Executing on device: 0000:01:00.0
Identifying board
Firmware Version: 4.22.0 (release,app,extended context switch buffer)
Board Name: Hailo-8
Device Architecture: HAILO8
Serial Number: HLLWM2B225000123
"""

TOOLS = {
    "hailo_status",
    "list_apps",
    "start_pipeline",
    "stop_pipeline",
    "get_detections",
    "snapshot",
    "describe_scene",
    "watch_scene",
}
MUTATING_TOOLS = {"start_pipeline", "stop_pipeline", "watch_scene"}

# A runner stand-in: prints detection lines until SIGINT, then exits cleanly.
FAKE_RUNNER = textwrap.dedent(
    f"""
    import json, sys, time
    frame = 0
    try:
        while True:
            frame += 1
            print({pipelines.DETECTIONS_MARKER!r} + json.dumps({{"frame": frame, "objects": [
                {{"label": "person", "confidence": 0.9, "bbox": [0, 0, 1, 1]}}]}}), flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        sys.exit(0)
    """
)
STUBBORN_RUNNER = (
    "import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\n"
    "print('ready', flush=True)\ntime.sleep(60)\n"
)


def _call(server: Any, tool: str, **arguments: Any) -> Any:
    return anyio.run(server.call_tool, tool, arguments)


def _json(result: Any) -> dict[str, Any]:
    if result.structured_content:
        return result.structured_content
    return json.loads(next(c for c in result.content if c.type == "text").text)


def _wait_for(predicate: Any, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.02)


@pytest.fixture
def mock_server() -> Any:
    return build_server(MockBackend("hailo10h"))


# --- contract tests (no hardware) ------------------------------------------------------------


@pytest.mark.sanity
def test_server_registers_tools_with_read_only_hints(mock_server: Any) -> None:
    tools = {tool.name: tool for tool in anyio.run(mock_server.list_tools)}

    assert set(tools) == TOOLS
    assert {n for n, t in tools.items() if not t.annotations.read_only_hint} == MUTATING_TOOLS


@pytest.mark.sanity
def test_device_identification_from_hailortcli() -> None:
    fields = parse_identify_output(IDENTIFY_H8)

    assert fields["Board Name"] == "Hailo-8"
    assert arch_from_identify(fields) == "hailo8"
    assert arch_from_identify({"Device Architecture": "HAILO8L"}) == "hailo8l"
    assert arch_from_identify({"Device Architecture": "HAILO10H"}) == "hailo10h"
    assert arch_from_identify({"Device Architecture": "UNKNOWN"}) is None


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("authorization", "expected_status"),
    [(None, 401), (b"Bearer wrong", 401), (b"secret", 401), (b"Bearer secret", 200)],
)
def test_bearer_token_is_required(authorization: bytes | None, expected_status: int) -> None:
    sent: list[dict[str, Any]] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    headers = [(b"authorization", authorization)] if authorization else []
    scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": headers}
    anyio.run(BearerTokenMiddleware(inner, "secret"), scope, receive, send)

    assert sent[0]["status"] == expected_status


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("source", "accepted"),
    [
        ("usb", True),
        ("rpi", True),
        ("rtsp://cam.local/stream", True),
        ("example.mp4", True),
        ("/etc/passwd", False),
        ("../secret.mp4", False),
        ("notes.txt", False),
    ],
)
def test_pipeline_sources_are_restricted(source: str, accepted: bool) -> None:
    if accepted:
        assert pipelines.resolve_source(source).endswith(source)
    else:
        with pytest.raises(ValueError):
            pipelines.resolve_source(source)


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"app": "vlm_chat"}, "cannot be started"),
        ({"app": "detection", "source": "usb"}, "no usb camera is connected"),
        ({"app": "detection", "fps": 500}, "fps must be"),
    ],
)
def test_start_pipeline_rejects_bad_arguments(
    mock_server: Any, arguments: dict[str, Any], message: str
) -> None:
    with pytest.raises(ToolError, match=message):
        _call(mock_server, "start_pipeline", **arguments)


@pytest.mark.sanity
def test_pipeline_tool_lifecycle_on_mock(mock_server: Any) -> None:
    pid = _json(_call(mock_server, "start_pipeline", app="detection", fps=50))["pipeline_id"]
    _wait_for(lambda: _json(_call(mock_server, "get_detections", pipeline_id=pid))["latest"])

    with pytest.raises(ToolError, match="already running"):
        _call(mock_server, "start_pipeline", app="pose_estimation")
    detections = _json(_call(mock_server, "get_detections", pipeline_id=pid, window_s=2))
    snapshot = _call(mock_server, "snapshot", pipeline_id=pid, crop=[0.0, 0.0, 0.5, 0.5])
    caption = _json(_call(mock_server, "describe_scene", pipeline_id=pid))["caption"]
    status = _json(_call(mock_server, "hailo_status"))
    stopped = _json(_call(mock_server, "stop_pipeline", pipeline_id=pid))

    assert "person" in detections["counts"]
    assert any(c.type == "image" for c in snapshot.content)
    assert "person" in caption
    assert [p["pipeline_id"] for p in status["pipelines"]] == [pid]
    assert stopped["stopped"] is True
    assert _json(_call(mock_server, "hailo_status"))["pipelines"] == []


@pytest.mark.sanity
def test_watch_scene_observes_and_cleans_up(mock_server: Any) -> None:
    result = _call(mock_server, "watch_scene", duration_s=1, labels=["person"], describe=True)

    report = _json(result)
    assert set(report["counts"]) == {"person"}
    assert any(c.type == "image" for c in result.content)
    assert report["vlm"]["caption"]
    assert report["stopped"]["stopped"] is True
    assert _json(_call(mock_server, "hailo_status"))["pipelines"] == []


@pytest.mark.sanity
def test_describe_scene_requires_hailo10h() -> None:
    server = build_server(MockBackend("hailo8"))

    with pytest.raises(ToolError, match="Hailo-10H"):
        _call(server, "watch_scene", describe=True)
    assert _json(_call(server, "hailo_status"))["pipelines"] == []


@pytest.mark.sanity
@pytest.mark.parametrize(
    ("script", "timeout_s", "forced"),
    [(FAKE_RUNNER, 5.0, False), (STUBBORN_RUNNER, 0.5, True)],
    ids=["graceful-sigint", "escalates"],
)
def test_subprocess_pipeline_stops_child(
    script: str, timeout_s: float, forced: bool, tmp_path: Path
) -> None:
    handle = SubprocessPipeline(
        app="detection",
        source="x",
        labels=(),
        command=[sys.executable, "-u", "-c", script],
        frame_dir=tmp_path / "frames",
    ).start()
    _wait_for(lambda: handle.buffer.summary(5.0)["frames_in_window"] or handle._log)

    report = handle.stop(timeout_s=timeout_s)

    assert report["forced"] is forced
    assert not handle.is_running()


@pytest.mark.sanity
def test_vlm_model_is_never_auto_downloaded(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not installed"):
        backend_mod.installed_vlm_hef(models_root=tmp_path)


# --- device smoke test -----------------------------------------------------------------------


@pytest.mark.requires_device
@pytest.mark.pipeline
def test_watch_scene_on_device() -> None:
    pytest.importorskip("gi")
    if HailoCliBackend().device_info().arch is None:
        pytest.skip("no Hailo device attached")
    video = Path(pipelines.resolve_source(pipelines.DEFAULT_SOURCE))
    if not video.is_file():
        pytest.skip(f"{video} not installed; run hailo-download-resources")
    server = build_server(HailoCliBackend())

    report = _json(_call(server, "watch_scene", duration_s=3, snapshot=False))

    assert report["frames_in_window"] > 0
    assert report["counts"]
    assert report["stopped"]["forced"] is False
