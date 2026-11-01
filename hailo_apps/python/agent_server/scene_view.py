"""MCP Apps view that shows snapshot and watch_scene results in the agent host's chat.

Hosts strip images from tool output before showing it to the user, so a snapshot is otherwise
visible only to the model. Tools that point ``_meta.ui.resourceUri`` at this view are rendered
by MCP Apps hosts (e.g. the OpenClaw dashboard) as an inline panel that receives the tool result.
"""

from __future__ import annotations

from mcp.server.apps import Apps

SCENE_VIEW_URI = "ui://hailo/scene-view.html"
# MCP Apps protocol version the view speaks to its host.
APPS_PROTOCOL_VERSION = "2026-01-26"

SCENE_VIEW_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --fg: #1d2330; --muted: #5d6675; --bg: #ffffff; --chip: #eef1f5; }
  @media (prefers-color-scheme: dark) {
    :root { --fg: #e6e9ef; --muted: #9aa3b2; --bg: #15181e; --chip: #232832; }
  }
  body { margin: 0; padding: 8px; font: 14px/1.4 system-ui, sans-serif;
         color: var(--fg); background: var(--bg); }
  img { display: block; max-width: 100%; height: auto; border-radius: 6px; }
  .muted { color: var(--muted); font-size: 12px; margin: 6px 0; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
  .chip { background: var(--chip); border-radius: 12px; padding: 2px 10px; }
  .caption { margin-top: 8px; }
  .error { color: #c0392b; }
</style>
</head>
<body>
<div id="status" class="muted">Waiting for the Hailo device result...</div>
<img id="frame" alt="" hidden>
<div id="info" class="muted"></div>
<div id="counts" class="chips"></div>
<div id="caption" class="caption"></div>
<script>
const PROTOCOL_VERSION = "__PROTOCOL_VERSION__";
let nextId = 0;
const pending = new Map();

function send(message) { window.parent.postMessage(message, "*"); }
function notify(method, params) { send({ jsonrpc: "2.0", method, params }); }
function request(method, params) {
  const id = ++nextId;
  send({ jsonrpc: "2.0", id, method, params });
  return new Promise((resolve) => pending.set(id, resolve));
}
function reportSize() {
  notify("ui/notifications/size-changed", {
    width: document.documentElement.scrollWidth,
    height: document.documentElement.scrollHeight,
  });
}
function el(id) { return document.getElementById(id); }

function render(result) {
  const content = (result && result.content) || [];
  const image = content.find((c) => c.type === "image");
  const text = content.find((c) => c.type === "text");
  let data = (result && result.structuredContent) || null;
  if (!data && text) { try { data = JSON.parse(text.text); } catch (e) { data = null; } }

  el("status").hidden = true;
  if (result && result.isError) {
    el("caption").className = "caption error";
    el("caption").textContent = text ? text.text : "The tool reported an error.";
    reportSize();
    return;
  }
  if (image) {
    const img = el("frame");
    img.onload = reportSize;
    img.src = "data:" + (image.mimeType || "image/jpeg") + ";base64," + image.data;
    img.hidden = false;
  }
  if (data) {
    const parts = [];
    if (data.app) parts.push(data.app);
    if (data.source) parts.push(data.source);
    if (data.frame !== undefined) parts.push("frame " + data.frame);
    else if (data.latest && data.latest.frame !== undefined) parts.push("frame " + data.latest.frame);
    if (data.window_s !== undefined) parts.push(data.window_s + " s window");
    el("info").textContent = parts.join(" · ");
    const counts = el("counts");
    counts.replaceChildren();
    for (const [label, c] of Object.entries(data.counts || {})) {
      const chip = document.createElement("span");
      chip.className = "chip";
      let line = label + ": " + c.latest + " now, " + c.max_in_frame + " peak";
      if (c.unique_tracks !== undefined) line += ", " + c.unique_tracks + " tracked";
      chip.textContent = line;
      counts.appendChild(chip);
    }
    if (!data.counts && data.objects) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = (data.objects_total || data.objects.length) + " objects";
      counts.appendChild(chip);
    }
    if (data.vlm && data.vlm.caption) el("caption").textContent = data.vlm.caption;
  }
  reportSize();
}

window.addEventListener("message", (event) => {
  const message = event.data;
  if (!message || message.jsonrpc !== "2.0") return;
  if (message.id !== undefined && pending.has(message.id) && !message.method) {
    pending.get(message.id)(message.result);
    pending.delete(message.id);
    return;
  }
  if (message.method === "ui/notifications/tool-result") render(message.params);
  else if ((message.method === "ping" || message.method === "ui/resource-teardown")
           && message.id !== undefined) send({ jsonrpc: "2.0", id: message.id, result: {} });
});

request("ui/initialize", {
  protocolVersion: PROTOCOL_VERSION,
  appInfo: { name: "hailo-scene-view", version: "1.0.0" },
  appCapabilities: {},
}).then(() => { notify("ui/notifications/initialized", {}); reportSize(); });
</script>
</body>
</html>
""".replace("__PROTOCOL_VERSION__", APPS_PROTOCOL_VERSION)

SCENE_VIEW_META = {"ui": {"resourceUri": SCENE_VIEW_URI}}


def scene_view_extension() -> Apps:
    """Build the MCP Apps extension that serves the scene view.

    Returns:
        The extension, to pass to ``MCPServer(extensions=[...])``.
    """
    apps = Apps()
    apps.add_html_resource(
        SCENE_VIEW_URI,
        SCENE_VIEW_HTML,
        name="Hailo scene view",
        description="Shows a Hailo snapshot with its detection counts and VLM caption.",
        prefers_border=True,
    )
    return apps
