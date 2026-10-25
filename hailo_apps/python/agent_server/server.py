"""MCP server (streamable HTTP) that exposes a Hailo device to a remote agent as typed tools."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import time
from collections.abc import Sequence
from typing import Any

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.types import ASGIApp, Receive, Scope, Send

from hailo_apps.python.agent_server import apps, imaging, pipelines
from hailo_apps.python.agent_server.backend import DeviceBackend, HailoCliBackend, MockBackend
from hailo_apps.python.agent_server.pipelines import PipelineManager
from hailo_apps.python.core.common.defines import (
    HAILO8_ARCH,
    HAILO8L_ARCH,
    HAILO10H_ARCH,
)

SERVER_NAME = "hailo-agent-server"
TOKEN_ENV_VAR = "HAILO_AGENT_TOKEN"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MCP_PATH = "/mcp"

INSTRUCTIONS = (
    "Tools for a Hailo AI accelerator attached to an edge device. Call hailo_status to learn "
    "which accelerator and cameras are attached. For a one-off look at a scene, prefer a single "
    "watch_scene call: it starts a pipeline, observes it, and can return a snapshot and a VLM "
    "caption. Use start_pipeline, get_detections, snapshot, describe_scene and stop_pipeline "
    "only for longer monitoring; one pipeline runs at a time. Raw video never leaves the device."
)
WATCH_MIN_S, WATCH_MAX_S = 1.0, 30.0
WATCH_DEFAULT_S = 3.0
FIRST_FRAME_TIMEOUT_S = 15.0
WATCH_SNAPSHOT_SIDE_PX = 640
DEFAULT_DESCRIBE_PROMPT = "Describe what is happening in this scene."
MAX_PROMPT_CHARS = 500

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
DEVICE_CONTROL = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False
)


def build_server(backend: DeviceBackend, manager: PipelineManager | None = None) -> MCPServer:
    """Create the MCP server with its tools bound to a device backend.

    Args:
        backend: Source of device information and pipelines.
        manager: Tracks running pipelines; a fresh one when None.

    Returns:
        The configured server.
    """
    server = MCPServer(name=SERVER_NAME, instructions=INSTRUCTIONS)
    manager = manager if manager is not None else PipelineManager()

    def _start(app: str, source: str, labels: list[str] | None, fps: int | None) -> Any:
        if app not in pipelines.PIPELINE_TARGETS:
            raise ValueError(
                f"app {app!r} cannot be started; startable apps: "
                f"{sorted(pipelines.PIPELINE_TARGETS)}"
            )
        arch = backend.device_info().arch
        if arch is None:
            raise ValueError("no Hailo device is attached; see hailo_status")
        if arch not in apps.supported_archs(app):
            raise ValueError(f"app {app!r} has no model for {arch}")
        resolved = pipelines.resolve_source(source)
        if source in pipelines.CAMERA_SOURCES:
            connected = backend.cameras()
            if not connected.get(source):
                available = [kind for kind, found in connected.items() if found]
                raise ValueError(
                    f"no {source} camera is connected; connected cameras: "
                    f"{available or 'none'}; a bundled video such as "
                    f"{pipelines.DEFAULT_SOURCE!r} also works"
                )
        label_filter = pipelines.validate_labels(labels)
        manager.check_capacity()
        return manager.add(
            backend.start_pipeline(app, resolved, label_filter, pipelines.validate_fps(fps))
        )

    def _check_describe(prompt: str) -> None:
        if backend.device_info().arch != HAILO10H_ARCH:
            raise ToolError("describe_scene needs a Hailo-10H; see hailo_status")
        if not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            raise ToolError(f"prompt must be 1-{MAX_PROMPT_CHARS} characters")

    def _describe(frame: Any, prompt: str, bbox: Any) -> dict[str, Any]:
        image = frame.image if bbox is None else imaging.crop(frame.image, bbox)
        started = time.monotonic()
        try:
            caption = backend.describe(image, prompt, frame.objects)
        except RuntimeError as exc:
            raise ToolError(str(exc)) from exc
        return {"caption": caption, "latency_s": round(time.monotonic() - started, 2)}

    def _snapshot_image(frame: Any, annotated: bool, bbox: Any, max_side: int) -> Any:
        image = imaging.annotate(frame.image, frame.objects) if annotated else frame.image
        if bbox is not None:
            image = imaging.crop(image, bbox)
        return imaging.fit(image, min(max(max_side, 64), imaging.MAX_SIDE_PX))

    @server.tool(annotations=READ_ONLY)
    def hailo_status() -> dict[str, Any]:
        """Report the attached Hailo accelerator and what is running on it.

        Gives architecture, board, firmware and host type, the connected cameras (`cameras.usb`
        and `cameras.rpi`; use "usb" or "rpi" as a start_pipeline source only when its list is
        non-empty), and the pipelines started through this server. `arch` is null and `error`
        explains why when no device is reachable.
        """
        status = backend.device_info().to_dict()
        status["cameras"] = backend.cameras()
        status["pipelines"] = [pipelines.describe(h) for h in manager.list()]
        return status

    @server.tool(annotations=READ_ONLY)
    def list_apps() -> dict[str, Any]:
        """List the Hailo-Apps vision pipelines and GenAI apps the attached accelerator can run.

        Each app has a name, kind (pipeline or genai), description, default model and whether
        start_pipeline can run it (`startable`).
        """
        arch = backend.device_info().arch
        return {"arch": arch, "apps": apps.list_apps(arch)}

    @server.tool(annotations=DEVICE_CONTROL)
    def start_pipeline(
        app: str,
        source: str = pipelines.DEFAULT_SOURCE,
        labels: list[str] | None = None,
        fps: int | None = None,
    ) -> dict[str, Any]:
        """Start a vision pipeline on the device and begin collecting its detections.

        Args:
            app: A startable app from list_apps, e.g. "detection" or "pose_estimation".
            source: "usb" or "rpi" for a camera that hailo_status lists under `cameras`, an
                rtsp:// URL, or the name of a bundled video such as "example.mp4".
            labels: Only report these labels in get_detections, e.g. ["person", "car"].
            fps: Frame rate to request (1-60); the app default when omitted.

        Returns the pipeline_id to pass to get_detections and stop_pipeline.
        """
        try:
            handle = _start(app, source, labels, fps)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return pipelines.describe(handle)

    @server.tool(annotations=DEVICE_CONTROL)
    def stop_pipeline(pipeline_id: str) -> dict[str, Any]:
        """Stop a running pipeline and free the accelerator.

        Args:
            pipeline_id: Id returned by start_pipeline.
        """
        try:
            return manager.stop(pipeline_id)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(annotations=READ_ONLY)
    def get_detections(pipeline_id: str, window_s: float = 5.0) -> dict[str, Any]:
        """Summarise what a running pipeline has seen recently.

        Args:
            pipeline_id: Id returned by start_pipeline.
            window_s: How many seconds back to look (max 120).

        Returns the objects in the latest frame (label, confidence, normalised bbox
        [xmin, ymin, xmax, ymax], track_id) and per-label counts: `latest`, `max_in_frame`
        and `unique_tracks` over the window.
        """
        try:
            handle = manager.get(pipeline_id)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        window_s = min(max(window_s, 0.1), pipelines.MAX_WINDOW_S)
        return {
            **pipelines.describe(handle),
            **handle.buffer.summary(window_s, handle.labels),
        }

    def _latest_frame(pipeline_id: str, crop: list[float] | None) -> tuple[Any, Any]:
        try:
            handle = manager.get(pipeline_id)
            bbox = imaging.validate_bbox(crop)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        frame = handle.latest_frame()
        if frame is None:
            raise ToolError("no frame captured yet; wait a second and retry")
        return frame, bbox

    # Unstructured: the result is an image content block plus a JSON text block.
    @server.tool(annotations=READ_ONLY, structured_output=False)
    def snapshot(
        pipeline_id: str,
        annotated: bool = True,
        crop: list[float] | None = None,
        max_side: int = imaging.MAX_SIDE_PX,
    ) -> list[Any]:
        """Grab the newest frame of a running pipeline as a JPEG.

        Args:
            pipeline_id: Id returned by start_pipeline.
            annotated: Draw the frame's detection boxes, labels and track ids.
            crop: Normalised [xmin, ymin, xmax, ymax] region to cut out, e.g. a detection's
                bbox from get_detections; the whole frame when omitted.
            max_side: Longest side of the returned image in pixels (64-1280).

        Returns the image plus the frame number, its age and its detections.
        """
        frame, bbox = _latest_frame(pipeline_id, crop)
        image = _snapshot_image(frame, annotated, bbox, max_side)
        meta = {
            "pipeline_id": pipeline_id,
            "frame": frame.frame,
            "frame_age_s": frame.age_s,
            "width": image.shape[1],
            "height": image.shape[0],
            **pipelines.compact_objects(frame.objects),
        }
        return [Image(data=imaging.encode_jpeg(image), format="jpeg"), meta]

    @server.tool(annotations=READ_ONLY)
    def describe_scene(
        pipeline_id: str,
        prompt: str = DEFAULT_DESCRIBE_PROMPT,
        crop: list[float] | None = None,
    ) -> dict[str, Any]:
        """Ask the on-device vision-language model about the newest frame (Hailo-10H only).

        Takes several seconds, longer on the first call while the model loads.

        Args:
            pipeline_id: Id returned by start_pipeline.
            prompt: Question about the frame, e.g. "What is the person holding?".
            crop: Normalised [xmin, ymin, xmax, ymax] region to ask about, e.g. a detection's
                bbox; the whole frame when omitted.

        Returns the model's caption with the frame number and the time it took.
        """
        _check_describe(prompt)
        frame, bbox = _latest_frame(pipeline_id, crop)
        return {
            "pipeline_id": pipeline_id,
            "frame": frame.frame,
            "frame_age_s": frame.age_s,
            "prompt": prompt,
            **_describe(frame, prompt, bbox),
        }

    # Unstructured: the result is an optional image content block plus a JSON text block.
    @server.tool(annotations=DEVICE_CONTROL, structured_output=False)
    def watch_scene(
        app: str = "detection",
        source: str = pipelines.DEFAULT_SOURCE,
        duration_s: float = WATCH_DEFAULT_S,
        labels: list[str] | None = None,
        snapshot: bool = True,
        describe: bool = False,
        prompt: str = DEFAULT_DESCRIBE_PROMPT,
        pipeline_id: str | None = None,
        keep_running: bool = False,
    ) -> list[Any]:
        """Watch a scene in one call: start a pipeline, observe it, report, and stop it.

        Prefer this over separate start_pipeline / get_detections / snapshot / stop_pipeline
        calls for a one-off look.

        Args:
            app: A startable app from list_apps; ignored when pipeline_id is given.
            source: "usb"/"rpi" camera from hailo_status, an rtsp:// URL, or a bundled video
                such as "example.mp4"; ignored when pipeline_id is given.
            duration_s: Seconds to observe after the first frame (1-30, default 3).
            labels: Only count these labels, e.g. ["person"].
            snapshot: Also return the newest frame, annotated, at most 640 px.
            describe: Also caption the newest frame with the on-device VLM (Hailo-10H only;
                adds about 10 s).
            prompt: Question for the VLM when describe is true.
            pipeline_id: Watch this already running pipeline instead of starting one; it is
                left running.
            keep_running: Leave a pipeline started here running and return its pipeline_id.

        Returns per-label counts over the window (`latest`, `max_in_frame`, `unique_tracks`),
        the latest frame's main objects, and the snapshot and caption when requested.
        """
        if describe:
            _check_describe(prompt)
        try:
            if pipeline_id is not None:
                handle, started_here = manager.get(pipeline_id), False
            else:
                handle, started_here = _start(app, source, labels, None), True
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        try:
            deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_S
            while handle.buffer.summary(pipelines.MAX_WINDOW_S)["latest"] is None:
                if not handle.is_running() or time.monotonic() > deadline:
                    reason = handle.error() or f"no frames after {FIRST_FRAME_TIMEOUT_S:g} s"
                    raise ToolError(f"pipeline produced no frames: {reason}")
                time.sleep(0.1)
            window = min(max(duration_s, WATCH_MIN_S), WATCH_MAX_S)
            time.sleep(window)
            result: dict[str, Any] = {
                **pipelines.describe(handle),
                **handle.buffer.summary(window, handle.labels),
            }
            content: list[Any] = []
            frame = handle.latest_frame() if (snapshot or describe) else None
            if (snapshot or describe) and frame is None:
                result["snapshot_error"] = "no frame captured yet"
            if frame is not None and snapshot:
                image = _snapshot_image(frame, True, None, WATCH_SNAPSHOT_SIDE_PX)
                content.append(Image(data=imaging.encode_jpeg(image), format="jpeg"))
            if frame is not None and describe:
                result["vlm"] = {"prompt": prompt, **_describe(frame, prompt, None)}
        finally:
            if started_here and not keep_running:
                result_stop = manager.stop(handle.pipeline_id)
        if started_here and not keep_running:
            result["stopped"] = {k: v for k, v in result_stop.items() if k != "pipeline_id"}
        return [*content, result]

    return server


class BearerTokenMiddleware:
    """Rejects HTTP requests that lack ``Authorization: Bearer <token>``."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        """Wrap an ASGI app.

        Args:
            app: The app to protect.
            token: The shared secret clients must present.
        """
        self._app = app
        self._expected = f"Bearer {token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Forward authorised requests and answer 401 to the rest.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        # Lifespan events must pass through: the MCP session manager starts on them.
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        presented = dict(scope.get("headers") or []).get(b"authorization", b"")
        if hmac.compare_digest(presented, self._expected):
            await self._app(scope, receive, send)
            return
        body = json.dumps({"error": "unauthorized"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b"Bearer"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def build_app(
    backend: DeviceBackend,
    token: str,
    allowed_hosts: Sequence[str],
    manager: PipelineManager | None = None,
) -> ASGIApp:
    """Build the token-protected streamable-HTTP ASGI app.

    Args:
        backend: Source of device information and pipelines.
        token: Shared secret for the bearer check; must be non-empty.
        allowed_hosts: ``Host`` header values accepted by the DNS-rebinding check, e.g.
            ``"10.0.0.5:8765"`` or ``"10.0.0.5:*"``.
        manager: Tracks running pipelines; a fresh one when None.

    Returns:
        The ASGI app to serve.

    Raises:
        ValueError: If ``token`` is empty.
    """
    if not token:
        raise ValueError("an auth token is required")
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=list(allowed_hosts)
    )
    app = build_server(backend, manager).streamable_http_app(
        streamable_http_path=MCP_PATH, transport_security=security
    )
    return BearerTokenMiddleware(app, token)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=SERVER_NAME,
        description=f"Serve Hailo device tools over MCP (streamable HTTP at {MCP_PATH}). "
        f"The bearer token is read from ${TOKEN_ENV_VAR}.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Interface to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind.")
    parser.add_argument(
        "--backend",
        choices=["hailo", "mock"],
        default="hailo",
        help="'hailo' uses the attached device; 'mock' fakes the device and a synthetic scene.",
    )
    parser.add_argument(
        "--mock-arch",
        choices=[HAILO8_ARCH, HAILO8L_ARCH, HAILO10H_ARCH],
        default=HAILO10H_ARCH,
        help="Architecture the mock backend reports.",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="Extra Host header value clients may use (repeatable), e.g. 172.30.49.5:8765. "
        "localhost and the bind address are always allowed.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the server.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    import uvicorn

    args = _parse_args(argv)
    token = os.environ.get(TOKEN_ENV_VAR, "")
    if not token:
        sys.exit(f"{SERVER_NAME}: set ${TOKEN_ENV_VAR} to the bearer token clients will use")
    backend: DeviceBackend = (
        MockBackend(args.mock_arch) if args.backend == "mock" else HailoCliBackend()
    )
    allowed_hosts = [
        "127.0.0.1:*",
        "localhost:*",
        f"{args.host}:{args.port}",
        *args.allowed_host,
    ]
    manager = PipelineManager()
    try:
        uvicorn.run(
            build_app(backend, token, allowed_hosts, manager), host=args.host, port=args.port
        )
    finally:
        manager.stop_all()
        backend.close()


if __name__ == "__main__":
    main()
