"""Contract tests for hailo-agent-server; no Hailo device or GStreamer needed."""

import json
import subprocess
from typing import Any

import anyio
import pytest

pytest.importorskip("mcp")

from hailo_apps.python.agent_server import apps, backend as backend_mod
from hailo_apps.python.agent_server.backend import (
    HailoCliBackend,
    MockBackend,
    arch_from_identify,
    parse_identify_output,
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


def _call_tool(server: Any, name: str) -> dict[str, Any]:
    async def _run() -> Any:
        return await server.call_tool(name, {})

    result = anyio.run(_run)
    return result.structured_content or json.loads(result.content[0].text)


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
    assert by_name["detection"]["command"] == "hailo-detect"
    assert by_name["detection"]["default_model"]


def test_list_apps_without_device_lists_whole_catalog() -> None:
    listed = apps.list_apps(None)

    assert len(listed) == len(apps.APP_CATALOG)
    assert all("default_model" not in app for app in listed)


def test_server_exposes_read_only_tools(h10_server: Any) -> None:
    tools = anyio.run(h10_server.list_tools)

    assert {tool.name for tool in tools} == {"hailo_status", "list_apps"}
    assert all(tool.annotations.read_only_hint for tool in tools)


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
