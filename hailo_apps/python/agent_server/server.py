"""MCP server (streamable HTTP) that exposes a Hailo device to a remote agent as typed tools."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.types import ASGIApp, Receive, Scope, Send

from hailo_apps.python.agent_server import apps
from hailo_apps.python.agent_server.backend import DeviceBackend, HailoCliBackend, MockBackend
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
    "Tools for a Hailo AI accelerator attached to an edge device. Call hailo_status first to "
    "learn which accelerator is attached, then list_apps to see which vision and GenAI apps it "
    "can run. All video stays on the device; tools return compact JSON."
)

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)


def build_server(backend: DeviceBackend) -> MCPServer:
    """Create the MCP server with its tools bound to a device backend.

    Args:
        backend: Source of device information.

    Returns:
        The configured server.
    """
    server = MCPServer(name=SERVER_NAME, instructions=INSTRUCTIONS)

    @server.tool(annotations=READ_ONLY)
    def hailo_status() -> dict[str, Any]:
        """Report the attached Hailo accelerator: architecture, board, firmware and host type.

        `arch` is null and `error` explains why when no device is reachable.
        """
        return backend.device_info().to_dict()

    @server.tool(annotations=READ_ONLY)
    def list_apps() -> dict[str, Any]:
        """List the Hailo-Apps vision pipelines and GenAI apps the attached accelerator can run.

        Each app has a name, kind (pipeline or genai), description and default model.
        """
        arch = backend.device_info().arch
        return {"arch": arch, "apps": apps.list_apps(arch)}

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


def build_app(backend: DeviceBackend, token: str, allowed_hosts: Sequence[str]) -> ASGIApp:
    """Build the token-protected streamable-HTTP ASGI app.

    Args:
        backend: Source of device information.
        token: Shared secret for the bearer check; must be non-empty.
        allowed_hosts: ``Host`` header values accepted by the DNS-rebinding check, e.g.
            ``"10.0.0.5:8765"`` or ``"10.0.0.5:*"``.

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
    app = build_server(backend).streamable_http_app(
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
        help="'hailo' queries the attached device; 'mock' needs no hardware.",
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
    uvicorn.run(build_app(backend, token, allowed_hosts), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
