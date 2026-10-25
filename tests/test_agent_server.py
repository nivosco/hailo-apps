"""Contract tests for hailo-agent-server; no Hailo device or GStreamer needed."""

import json
import subprocess
import sys
import textwrap
import time
from typing import Any

import anyio
import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp.server.mcpserver.exceptions import ToolError

from hailo_apps.python.agent_server import apps, backend as backend_mod, imaging, pipelines
from hailo_apps.python.agent_server.backend import (
    HailoCliBackend,
    MockBackend,
    arch_from_identify,
    parse_identify_output,
)
from hailo_apps.python.agent_server.pipelines import (
    DetectionBuffer,
    FrameDetections,
    MockPipeline,
    SubprocessPipeline,
)
from hailo_apps.python.agent_server.server import BearerTokenMiddleware, build_app, build_server

IDENTIFY_H8 = """\
Executing on device: 0000:01:00.0
Identifying board
Control Protocol Version: 2
Firmware Version: 4.22.0 (release,app,extended context switch buffer)
Logger Version: 0
Board Name: Hailo-8
Device Architecture: HAILO8
Serial Number: HLLWM2B225000123
Part Number: HM218B1C2LAE
Product Name: HAILO-8 AI ACC M.2 M KEY MODULE EXT TEMP
"""

GENAI_APPS = {"vlm_chat", "llm_chat", "whisper_chat"}


def _call_tool(server: Any, name: str, **arguments: Any) -> dict[str, Any]:
    result = anyio.run(server.call_tool, name, arguments)
    return result.structured_content or json.loads(result.content[0].text)


def _wait_for(predicate: Any, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.02)


@pytest.fixture
def h10_server() -> Any:
    return build_server(MockBackend("hailo10h"))


def test_parse_identify_output_reads_key_value_lines() -> None:
    fields = parse_identify_output(IDENTIFY_H8)

    assert fields["Board Name"] == "Hailo-8"
    assert fields["Firmware Version"].startswith("4.22.0")
    assert "Identifying board" not in fields


@pytest.mark.parametrize(
    ("device_arch", "expected"),
    [
        ("HAILO8", "hailo8"),
        ("HAILO8L", "hailo8l"),
        ("HAILO10H", "hailo10h"),
        ("HAILO15H", "hailo10h"),
        ("SOMETHING_ELSE", None),
    ],
)
def test_arch_from_identify_maps_device_architecture(
    device_arch: str, expected: str | None
) -> None:
    assert arch_from_identify({"Device Architecture": device_arch}) == expected


def test_cli_backend_reports_device_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=IDENTIFY_H8, stderr="")
    monkeypatch.setattr(backend_mod.subprocess, "run", lambda *a, **k: completed)

    info = HailoCliBackend().device_info()

    assert info.arch == "hailo8"
    assert info.serial_number == "HLLWM2B225000123"
    assert info.error is None


def test_cli_backend_reports_missing_hailortcli(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: Any, **_k: Any) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(backend_mod.subprocess, "run", _raise)

    info = HailoCliBackend().device_info()

    assert info.arch is None
    assert info.error == "hailortcli not found"


def test_cli_backend_reports_identify_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: Any, **_k: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="hailortcli", timeout=1)

    monkeypatch.setattr(backend_mod.subprocess, "run", _raise)

    info = HailoCliBackend(timeout_s=1).device_info()

    assert info.arch is None
    assert "timed out" in info.error


def test_list_apps_hides_genai_apps_on_hailo8() -> None:
    names = {app["name"] for app in apps.list_apps("hailo8")}

    assert "detection" in names
    assert not names & GENAI_APPS


def test_list_apps_includes_genai_apps_on_hailo10h() -> None:
    by_name = {app["name"]: app for app in apps.list_apps("hailo10h")}

    assert GENAI_APPS <= set(by_name)
    assert by_name["detection"]["default_model"]
    assert "command" not in by_name["detection"]


def test_list_apps_without_device_lists_whole_catalog() -> None:
    listed = apps.list_apps(None)

    assert len(listed) == len(apps.APP_CATALOG)
    assert all("default_model" not in app for app in listed)


def test_server_marks_only_device_control_tools_as_mutating(h10_server: Any) -> None:
    tools = {tool.name: tool for tool in anyio.run(h10_server.list_tools)}

    assert set(tools) == {
        "hailo_status",
        "list_apps",
        "start_pipeline",
        "stop_pipeline",
        "get_detections",
        "snapshot",
        "describe_scene",
        "watch_scene",
    }
    mutating = {name for name, tool in tools.items() if not tool.annotations.read_only_hint}
    assert mutating == {"start_pipeline", "stop_pipeline", "watch_scene"}


def test_hailo_status_tool_returns_device_snapshot(h10_server: Any) -> None:
    status = _call_tool(h10_server, "hailo_status")

    assert status["arch"] == "hailo10h"
    assert status["error"] is None


def test_list_apps_tool_matches_device_arch() -> None:
    result = _call_tool(build_server(MockBackend("hailo8l")), "list_apps")

    assert result["arch"] == "hailo8l"
    assert not {app["name"] for app in result["apps"]} & GENAI_APPS


def _asgi_get(app: Any, headers: list[tuple[bytes, bytes]]) -> tuple[int, bool]:
    """Send one GET through an ASGI app; return (status, whether the inner app ran)."""
    sent: list[dict[str, Any]] = []
    scope = {"type": "http", "method": "GET", "path": "/mcp", "headers": headers}

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(message: dict[str, Any]) -> None:
        sent.append(message)

    anyio.run(app, scope, _receive, _send)
    return sent[0]["status"], sent[0].get("inner", False)


async def _inner_app(scope: Any, receive: Any, send: Any) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": [], "inner": True})
    await send({"type": "http.response.body", "body": b""})


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(b"authorization", b"Bearer wrong")],
        [(b"authorization", b"secret")],
    ],
)
def test_bearer_middleware_rejects_bad_credentials(headers: list[tuple[bytes, bytes]]) -> None:
    status, inner_ran = _asgi_get(BearerTokenMiddleware(_inner_app, "secret"), headers)

    assert status == 401
    assert not inner_ran


def test_bearer_middleware_forwards_valid_token() -> None:
    app = BearerTokenMiddleware(_inner_app, "secret")

    status, inner_ran = _asgi_get(app, [(b"authorization", b"Bearer secret")])

    assert status == 200
    assert inner_ran


def test_build_app_refuses_empty_token() -> None:
    with pytest.raises(ValueError):
        build_app(MockBackend(), token="", allowed_hosts=["127.0.0.1:*"])


# --- pipelines -----------------------------------------------------------------------------


def _frame(t: float, *objects: dict[str, Any], frame: int = 0) -> FrameDetections:
    return FrameDetections(timestamp=t, frame=frame, objects=objects)


def _obj(label: str, track_id: int | None = None) -> dict[str, Any]:
    obj: dict[str, Any] = {"label": label, "confidence": 0.9, "bbox": [0.1, 0.1, 0.2, 0.2]}
    if track_id is not None:
        obj["track_id"] = track_id
    return obj


@pytest.mark.parametrize("source", ["usb", "rpi", "rtsp://cam.local/stream", "example.mp4"])
def test_resolve_source_accepts_cameras_streams_and_bundled_videos(source: str) -> None:
    resolved = pipelines.resolve_source(source)

    assert resolved == source or resolved.endswith(f"/videos/{source}")


@pytest.mark.parametrize(
    "source", ["/etc/passwd", "../secret.mp4", "videos/example.mp4", ".hidden.mp4", "notes.txt"]
)
def test_resolve_source_rejects_paths_and_non_videos(source: str) -> None:
    with pytest.raises(ValueError):
        pipelines.resolve_source(source)


def test_parse_detections_line_ignores_app_output() -> None:
    assert pipelines.parse_detections_line("Frame count: 3", 1.0) is None
    assert pipelines.parse_detections_line(f"{pipelines.DETECTIONS_MARKER}{{bad", 1.0) is None


def test_parse_detections_line_reads_marked_frame() -> None:
    line = pipelines.DETECTIONS_MARKER + json.dumps({"frame": 7, "objects": [_obj("person", 2)]})

    frame = pipelines.parse_detections_line(line, 1.5)

    assert frame == _frame(1.5, _obj("person", 2), frame=7)


def test_detection_buffer_summary_counts_within_window() -> None:
    buf = DetectionBuffer()
    buf.add(_frame(0.0, _obj("person", 1), _obj("person", 2), _obj("person", 3)))
    buf.add(_frame(9.0, _obj("person", 1), _obj("person", 4), _obj("car", 9)))
    buf.add(_frame(10.0, _obj("person", 4), _obj("car", 9)))

    summary = buf.summary(window_s=2.0, now=10.0)

    assert summary["frames_in_window"] == 2
    assert summary["counts"]["person"] == {"latest": 1, "max_in_frame": 2, "unique_tracks": 2}
    assert summary["latest"]["age_s"] == 0.0


def test_detection_buffer_summary_applies_label_filter() -> None:
    buf = DetectionBuffer()
    buf.add(_frame(1.0, _obj("person", 1), _obj("car", 9)))

    summary = buf.summary(window_s=5.0, labels=["car"], now=1.0)

    assert set(summary["counts"]) == {"car"}
    assert [o["label"] for o in summary["latest"]["objects"]] == ["car"]


def test_detection_buffer_drops_frames_older_than_max_age() -> None:
    buf = DetectionBuffer(max_age_s=10.0)
    buf.add(_frame(0.0, _obj("person")))
    buf.add(_frame(30.0, _obj("car")))

    summary = buf.summary(window_s=100.0, now=30.0)

    assert summary["frames_in_window"] == 1


def test_mock_scene_has_stable_person_tracks() -> None:
    first = MockPipeline.scene(1.0)
    later = MockPipeline.scene(1.5)

    track_ids = {o["track_id"] for o in first if o["label"] == "person"}
    assert track_ids == {o["track_id"] for o in later if o["label"] == "person"} == {1}
    assert any(o["label"] == "car" for o in first)


def test_runner_command_passes_input_and_fps() -> None:
    cmd = pipelines.runner_command("detection", "/videos/example.mp4", 15)

    assert cmd[:3] == [sys.executable, "-m", "hailo_apps.python.agent_server.pipeline_runner"]
    assert cmd[cmd.index("--input") + 1] == "/videos/example.mp4"
    assert cmd[cmd.index("--frame-rate") + 1] == "15"


FAKE_RUNNER = textwrap.dedent(
    f"""
    import json, sys, time
    frame = 0
    try:
        while True:
            frame += 1
            print("app noise", flush=True)
            print({pipelines.DETECTIONS_MARKER!r} + json.dumps(
                {{"frame": frame, "objects": [{{"label": "person", "confidence": 0.9,
                  "bbox": [0, 0, 1, 1], "track_id": 1}}]}}), flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        sys.exit(0)
    """
)

STUBBORN_RUNNER = "import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\n" + (
    "print('ready', flush=True)\ntime.sleep(60)\n"
)


def test_subprocess_pipeline_collects_frames_and_stops_on_sigint() -> None:
    handle = SubprocessPipeline(
        app="detection", source="x", labels=(), command=[sys.executable, "-c", FAKE_RUNNER]
    ).start()
    _wait_for(lambda: handle.buffer.summary(5.0)["frames_in_window"] >= 3)

    report = handle.stop(timeout_s=5)

    assert report == {"stopped": True, "forced": False, "exit_code": 0}
    assert not handle.is_running()
    assert handle.error() is None


def test_subprocess_pipeline_escalates_when_sigint_is_ignored() -> None:
    handle = SubprocessPipeline(
        app="detection",
        source="x",
        labels=(),
        command=[sys.executable, "-u", "-c", STUBBORN_RUNNER],
    ).start()
    _wait_for(lambda: handle._log and handle._log[-1] == "ready")

    report = handle.stop(timeout_s=0.5)

    assert report["forced"] is True
    assert not handle.is_running()


def test_subprocess_pipeline_reports_crash_output() -> None:
    handle = SubprocessPipeline(
        app="detection",
        source="x",
        labels=(),
        command=[sys.executable, "-c", "print('no device found'); raise SystemExit(3)"],
    ).start()
    _wait_for(lambda: not handle.is_running())
    handle._reader.join(timeout=2)

    assert "exited with 3" in handle.error()
    assert "no device found" in handle.error()


def test_cli_backend_refuses_missing_video(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="video not found"):
        HailoCliBackend().start_pipeline("detection", str(tmp_path / "nope.mp4"), (), None)


def test_list_apps_marks_detection_pipelines_startable() -> None:
    by_name = {app["name"]: app for app in apps.list_apps("hailo10h")}

    assert by_name["detection"]["startable"] is True
    assert by_name["vlm_chat"]["startable"] is False


def test_pipeline_tools_round_trip_on_mock(h10_server: Any) -> None:
    started = _call_tool(h10_server, "start_pipeline", app="detection", fps=50, labels=["person"])
    pid = started["pipeline_id"]
    _wait_for(
        lambda: _call_tool(h10_server, "get_detections", pipeline_id=pid)["frames_in_window"] > 0
    )

    detections = _call_tool(h10_server, "get_detections", pipeline_id=pid, window_s=2)
    status = _call_tool(h10_server, "hailo_status")
    stopped = _call_tool(h10_server, "stop_pipeline", pipeline_id=pid)

    assert set(detections["counts"]) == {"person"}
    assert [p["pipeline_id"] for p in status["pipelines"]] == [pid]
    assert stopped["stopped"] is True
    assert _call_tool(h10_server, "hailo_status")["pipelines"] == []


def test_start_pipeline_allows_one_pipeline_at_a_time(h10_server: Any) -> None:
    pid = _call_tool(h10_server, "start_pipeline", app="detection")["pipeline_id"]
    try:
        with pytest.raises(ToolError, match="already running"):
            _call_tool(h10_server, "start_pipeline", app="pose_estimation")
    finally:
        _call_tool(h10_server, "stop_pipeline", pipeline_id=pid)

    second = _call_tool(h10_server, "start_pipeline", app="pose_estimation")
    _call_tool(h10_server, "stop_pipeline", pipeline_id=second["pipeline_id"])


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"app": "vlm_chat"}, "cannot be started"),
        ({"app": "detection", "source": "/etc/passwd"}, "not a path"),
        ({"app": "detection", "fps": 500}, "fps must be"),
        ({"app": "detection", "labels": [str(i) for i in range(30)]}, "at most"),
    ],
)
def test_start_pipeline_rejects_bad_arguments(
    h10_server: Any, arguments: dict[str, Any], message: str
) -> None:
    with pytest.raises(ToolError, match=message):
        _call_tool(h10_server, "start_pipeline", **arguments)


def test_tools_reject_unknown_pipeline_id(h10_server: Any) -> None:
    for tool in ("get_detections", "stop_pipeline"):
        with pytest.raises(ToolError, match="unknown pipeline_id"):
            _call_tool(h10_server, tool, pipeline_id="nope")


# --- snapshot / describe_scene ---------------------------------------------------------------


def _content(result: Any, kind: str) -> Any:
    return next(c for c in result.content if c.type == kind)


@pytest.fixture
def running_mock(h10_server: Any) -> Any:
    pid = _call_tool(h10_server, "start_pipeline", app="detection", fps=50)["pipeline_id"]
    _wait_for(lambda: _call_tool(h10_server, "get_detections", pipeline_id=pid)["latest"])
    yield h10_server, pid
    _call_tool(h10_server, "stop_pipeline", pipeline_id=pid)


@pytest.mark.parametrize(
    "bbox", [[0.1, 0.1, 0.5], [0.5, 0.1, 0.4, 0.9], [-0.1, 0.0, 0.5, 0.5], [0.0, 0.0, 1.2, 1.0]]
)
def test_validate_bbox_rejects_malformed_boxes(bbox: list[float]) -> None:
    with pytest.raises(ValueError):
        imaging.validate_bbox(bbox)


def test_crop_keeps_margin_inside_frame() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    out = imaging.crop(image, (0.0, 0.0, 0.5, 0.5), margin=0.1)

    assert out.shape[:2] == (60, 120)


def test_fit_downscales_but_never_upscales() -> None:
    big = np.zeros((1000, 2000, 3), dtype=np.uint8)
    small = np.zeros((10, 20, 3), dtype=np.uint8)

    assert imaging.fit(big, 500).shape[:2] == (250, 500)
    assert imaging.fit(small, 500).shape[:2] == (10, 20)


def test_annotate_does_not_modify_the_input() -> None:
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    out = imaging.annotate(image, [_obj("person", 1)])

    assert not image.any()
    assert out.any()


def test_subprocess_pipeline_reads_frame_written_by_runner(tmp_path: Any) -> None:
    jpeg = imaging.encode_jpeg(np.full((40, 60, 3), 128, dtype=np.uint8))
    (tmp_path / pipelines.FRAME_IMAGE_NAME).write_bytes(jpeg)
    meta = {"frame": 5, "objects": [_obj("car", 9)], "captured_at": time.time()}
    (tmp_path / pipelines.FRAME_META_NAME).write_text(json.dumps(meta))
    handle = SubprocessPipeline(
        app="detection", source="x", labels=(), command=[], frame_dir=tmp_path
    )

    frame = handle.latest_frame()

    assert frame.frame == 5
    assert frame.image.shape == (40, 60, 3)
    assert frame.objects[0]["label"] == "car"


def test_subprocess_pipeline_has_no_frame_before_runner_writes_one(tmp_path: Any) -> None:
    handle = SubprocessPipeline(
        app="detection", source="x", labels=(), command=[], frame_dir=tmp_path
    )

    assert handle.latest_frame() is None


def test_subprocess_pipeline_removes_frame_dir_on_stop(tmp_path: Any) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    handle = SubprocessPipeline(
        app="detection",
        source="x",
        labels=(),
        command=[sys.executable, "-c", FAKE_RUNNER],
        frame_dir=frame_dir,
    ).start()

    handle.stop(timeout_s=5)

    assert not frame_dir.exists()


def test_runner_command_passes_frame_dir_before_app_args(tmp_path: Any) -> None:
    cmd = pipelines.runner_command("detection", "usb", None, tmp_path)

    assert cmd.index("--frame-dir") < cmd.index("--")
    assert cmd[cmd.index("--frame-dir") + 1] == str(tmp_path)


def test_snapshot_returns_jpeg_and_frame_metadata(running_mock: Any) -> None:
    server, pid = running_mock

    result = anyio.run(server.call_tool, "snapshot", {"pipeline_id": pid})

    image = _content(result, "image")
    meta = json.loads(_content(result, "text").text)
    assert image.mime_type == "image/jpeg"
    assert (meta["width"], meta["height"]) == imaging.MOCK_FRAME_SIZE
    assert meta["objects"]


def test_snapshot_crops_and_limits_size(running_mock: Any) -> None:
    server, pid = running_mock

    result = anyio.run(
        server.call_tool,
        "snapshot",
        {"pipeline_id": pid, "crop": [0.0, 0.0, 0.5, 0.5], "max_side": 200},
    )

    meta = json.loads(_content(result, "text").text)
    assert max(meta["width"], meta["height"]) == 200


def test_snapshot_rejects_bad_crop(running_mock: Any) -> None:
    server, pid = running_mock

    with pytest.raises(ToolError, match="crop"):
        anyio.run(server.call_tool, "snapshot", {"pipeline_id": pid, "crop": [0.5, 0.5, 0.1, 0.9]})


def test_describe_scene_returns_caption_for_latest_frame(running_mock: Any) -> None:
    server, pid = running_mock

    result = _call_tool(server, "describe_scene", pipeline_id=pid, prompt="Who is there?")

    assert "Who is there?" in result["caption"]
    assert "person" in result["caption"]
    assert result["frame"] > 0


@pytest.mark.parametrize("prompt", ["", "   ", "x" * 501])
def test_describe_scene_rejects_bad_prompt(running_mock: Any, prompt: str) -> None:
    server, pid = running_mock

    with pytest.raises(ToolError, match="prompt"):
        _call_tool(server, "describe_scene", pipeline_id=pid, prompt=prompt)


def test_describe_scene_requires_hailo10h() -> None:
    server = build_server(MockBackend("hailo8"))
    pid = _call_tool(server, "start_pipeline", app="detection")["pipeline_id"]
    try:
        with pytest.raises(ToolError, match="Hailo-10H"):
            _call_tool(server, "describe_scene", pipeline_id=pid)
    finally:
        _call_tool(server, "stop_pipeline", pipeline_id=pid)


class _FakeVlm:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.timeouts: list[int] = []

    def vlm_inference(self, image: Any, prompt: str, timeout: int) -> dict[str, str]:
        self.timeouts.append(timeout)
        return {"answer": self.answer, "time": "0.1 seconds"}


def test_cli_backend_loads_vlm_once_with_a_longer_first_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeVlm("A person at a door.")
    loads: list[int] = []
    monkeypatch.setattr(HailoCliBackend, "_load_vlm", staticmethod(lambda: loads.append(1) or fake))
    backend = HailoCliBackend()
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    first = backend.describe(image, "What?", ())
    backend.describe(image, "What?", ())

    assert first == "A person at a door."
    assert loads == [1]
    assert fake.timeouts == [backend_mod.VLM_LOAD_TIMEOUT_S, backend_mod.VLM_TIMEOUT_S]


def test_cli_backend_turns_vlm_error_answers_into_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        HailoCliBackend,
        "_load_vlm",
        staticmethod(lambda: _FakeVlm("Request timed out after 60 seconds")),
    )

    with pytest.raises(RuntimeError, match="timed out"):
        HailoCliBackend().describe(np.zeros((8, 8, 3), dtype=np.uint8), "What?", ())


def test_installed_vlm_hef_refuses_to_download_missing_model(tmp_path: Any) -> None:
    with pytest.raises(RuntimeError, match="not installed"):
        backend_mod.installed_vlm_hef(models_root=tmp_path)


def test_installed_vlm_hef_finds_installed_model(tmp_path: Any) -> None:
    name = apps.get_default_model_name("vlm_chat", "hailo10h")
    hef = tmp_path / "hailo10h" / f"{name}.hef"
    hef.parent.mkdir()
    hef.write_bytes(b"")

    assert backend_mod.installed_vlm_hef(models_root=tmp_path) == hef


# --- cameras ---------------------------------------------------------------------------------

RPICAM_LIST = """\
Available cameras
-----------------
0 : ov5647 [2592x1944 10-bit GBRG] (/base/axi/pcie@1000120000/rp1/i2c@88000/ov5647@36)
    Modes: 'SGBRG10_CSI2P' : 640x480 [62.50 fps - (16, 0)/2560x1920 crop]
                             1296x972 [46.34 fps - (0, 0)/2592x1944 crop]
1 : imx708 [4608x2592 10-bit RGGB] (/base/axi/pcie@1000120000/rp1/i2c@80000/imx708@1a)
"""


def test_parse_rpi_camera_list_reads_sensor_names() -> None:
    assert backend_mod.parse_rpi_camera_list(RPICAM_LIST) == ["ov5647", "imx708"]


def test_parse_rpi_camera_list_handles_no_cameras() -> None:
    assert backend_mod.parse_rpi_camera_list("No cameras available!") == []


def test_list_rpi_cameras_is_empty_off_a_pi(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: Any, **_k: Any) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(backend_mod.subprocess, "run", _raise)

    assert backend_mod.list_rpi_cameras() == []


def test_hailo_status_reports_cameras(h10_server: Any) -> None:
    status = _call_tool(h10_server, "hailo_status")

    assert status["cameras"] == {"usb": [], "rpi": ["mock"]}


def test_start_pipeline_rejects_unconnected_camera(h10_server: Any) -> None:
    with pytest.raises(ToolError, match=r"no usb camera is connected.*\['rpi'\]"):
        _call_tool(h10_server, "start_pipeline", app="detection", source="usb")


def test_start_pipeline_accepts_connected_camera(h10_server: Any) -> None:
    pid = _call_tool(h10_server, "start_pipeline", app="detection", source="rpi")["pipeline_id"]

    _call_tool(h10_server, "stop_pipeline", pipeline_id=pid)


# --- compact outputs / watch_scene -----------------------------------------------------------


def test_compact_objects_keeps_most_confident_and_reports_total() -> None:
    objects = [
        {"label": "person", "confidence": c / 100, "bbox": [0.12345, 0.2, 0.3, 0.4]}
        for c in range(30)
    ]

    out = pipelines.compact_objects(objects, limit=5)

    assert [o["confidence"] for o in out["objects"]] == [0.29, 0.28, 0.27, 0.26, 0.25]
    assert out["objects"][0]["bbox"][0] == 0.123
    assert out["objects_total"] == 30


def test_compact_objects_omits_total_when_nothing_dropped() -> None:
    assert "objects_total" not in pipelines.compact_objects([_obj("car")])


def test_describe_shows_video_name_and_omits_empty_fields() -> None:
    handle = MockPipeline(app="detection", source="/x/videos/example.mp4", labels=())

    out = pipelines.describe(handle)

    assert out["source"] == "example.mp4"
    assert "labels" not in out and "error" not in out


def test_watch_scene_reports_counts_snapshot_and_stops(h10_server: Any) -> None:
    result = anyio.run(h10_server.call_tool, "watch_scene", {"duration_s": 1, "labels": ["person"]})

    meta = json.loads(_content(result, "text").text)
    assert _content(result, "image").mime_type == "image/jpeg"
    assert set(meta["counts"]) == {"person"}
    assert meta["stopped"]["stopped"] is True
    assert _call_tool(h10_server, "hailo_status")["pipelines"] == []


def test_watch_scene_adds_vlm_caption_when_asked(h10_server: Any) -> None:
    result = anyio.run(
        h10_server.call_tool,
        "watch_scene",
        {"duration_s": 1, "snapshot": False, "describe": True, "prompt": "Who?"},
    )

    meta = json.loads(_content(result, "text").text)
    assert not [c for c in result.content if c.type == "image"]
    assert "Who?" in meta["vlm"]["caption"]


def test_watch_scene_keep_running_leaves_pipeline_for_follow_up(h10_server: Any) -> None:
    result = anyio.run(h10_server.call_tool, "watch_scene", {"duration_s": 1, "keep_running": True})
    pid = json.loads(_content(result, "text").text)["pipeline_id"]

    again = anyio.run(h10_server.call_tool, "watch_scene", {"duration_s": 1, "pipeline_id": pid})

    assert json.loads(_content(again, "text").text)["running"] is True
    _call_tool(h10_server, "stop_pipeline", pipeline_id=pid)


def test_watch_scene_rejects_describe_off_hailo10h_before_starting() -> None:
    server = build_server(MockBackend("hailo8"))

    with pytest.raises(ToolError, match="Hailo-10H"):
        anyio.run(server.call_tool, "watch_scene", {"describe": True})
    assert _call_tool(server, "hailo_status")["pipelines"] == []


def test_watch_scene_stops_pipeline_that_never_produces_frames(
    h10_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MockPipeline, "_run", lambda self: None)

    with pytest.raises(ToolError, match="no frames"):
        anyio.run(h10_server.call_tool, "watch_scene", {"duration_s": 1})
    assert _call_tool(h10_server, "hailo_status")["pipelines"] == []


def test_watch_scene_defaults_to_short_observation(h10_server: Any) -> None:
    started = time.monotonic()

    anyio.run(h10_server.call_tool, "watch_scene", {"snapshot": False})

    assert time.monotonic() - started < 3.0 + 2.0
