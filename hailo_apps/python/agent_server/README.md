# Hailo Agent Server

An [MCP](https://modelcontextprotocol.io) server that lets a large LLM agent, such as one running in
[OpenClaw](https://openclaw.ai), see through a camera on a Hailo device and run Hailo vision analytics
on it. All pixel work stays on the Hailo accelerator; the agent only receives detection counts, small
JPEG snapshots and short VLM captions, never a video stream.

```
 ┌──────────── GPU host ────────────┐               ┌──────── Edge device ────────┐
 │ OpenClaw gateway + dashboard     │  MCP (HTTP,   │ hailo-agent-server          │
 │ Ollama  ──  qwen3.5:27b (brain)  │ ─────────────►│   ├─ Hailo-Apps pipelines   │
 └──────────────────────────────────┘  bearer token)│   └─ on-device VLM (H10)    │
                                                     │ Hailo-8 / 8L / 10H + camera │
                                                     └─────────────────────────────┘
```

## Features

- **Typed tools, no shell access** - the agent can only call the tools listed below
- **One-call scene checks** - `watch_scene` starts a pipeline, observes, reports and stops
- **On-device VLM captions** on Hailo-10H via the `vlm_chat` model
- **Camera, video file and RTSP inputs**; bundled videos work without a camera
- **Mock backend** to develop and test the agent side without a Hailo device

## Tools

| Tool | What it returns |
|------|-----------------|
| `hailo_status` | Accelerator, firmware, host type, connected cameras (`usb` / `rpi`), running pipelines |
| `list_apps` | Vision pipelines and GenAI apps the accelerator can run, with default models |
| `watch_scene` | One call: start a pipeline, observe for N seconds, return counts + optional snapshot and VLM caption, stop |
| `start_pipeline` / `stop_pipeline` | Start and stop a detection-family pipeline (detection, simple_detection, pose_estimation, instance_segmentation, tiling) for longer monitoring |
| `get_detections` | Per-label counts over a time window and the latest frame's main objects |
| `snapshot` | The newest frame as a JPEG (at most 1280 px), optionally annotated or cropped to a box |
| `describe_scene` | A caption of the newest frame, or a crop, from the on-device VLM (Hailo-10H only) |

Only one pipeline runs at a time. Inputs are limited to cameras (`usb`, `rpi`), `rtsp://` streams and
videos in the Hailo-Apps resources directory, never arbitrary file paths.

## Requirements

### Edge device

- A supported Hailo-Apps host running **Linux** (Raspberry Pi 5 with AI HAT+ / AI HAT+ 2, or x86_64)
  with a Hailo-8, Hailo-8L or Hailo-10H, set up for **GStreamer pipeline apps** as in the
  [Prerequisites](../../../doc/user_guide/prerequisites.md) guide
- Python >=3.10
- A camera, or the bundled example videos
- For `describe_scene`: a Hailo-10H and **about 3.5 GB of free disk** for the VLM model

> Windows is not supported as an edge device yet: Hailo-Apps GStreamer pipelines do not run on
> Windows. Use the mock backend there for agent-side development.

### Agent host

The agent brain is a large open-weight model with tool calling and vision input.

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA with **at least 24 GB VRAM** (e.g. RTX A5000); 32 GB or more recommended for long context. `qwen3.5:27b` (4-bit) uses about 19 GB with a 32k context |
| NVIDIA driver | Recent enough for current Ollama CUDA builds (555 or newer was used in testing) |
| Software | Docker (or a native install) for [Ollama](https://ollama.com) and Node.js 24.16+ for OpenClaw |

Any OpenAI- or Ollama-compatible endpoint works; the model only needs tool calling, and vision
input if the agent should look at snapshots itself.

## Installation

### Edge device

Install Hailo-Apps as described in the [Installation Guide](../../../doc/user_guide/installation.md),
then add the agent-server extra and prepare the resources:

```bash
# From the repository root directory
pip install -e ".[agent-server]"

hailo-set-env                               # writes resources/.env (required by the pipelines)
hailo-compile-postprocess                   # builds the post-process libraries pipelines load
hailo-download-resources --group detection  # detection model and example videos
hailo-download-resources --group vlm_chat   # Hailo-10H only: VLM for describe_scene (3.3 GB)
```

Download any other pipeline you want the agent to use the same way (`--group pose_estimation`, ...).
The server never downloads models itself: a missing model is reported as an error.

### Agent host

1. **Ollama and the model**:
   ```bash
   docker run -d --name ollama --gpus '"device=0"' -p 127.0.0.1:11434:11434 \
     -v ollama:/root/.ollama -e OLLAMA_CONTEXT_LENGTH=32768 ollama/ollama
   docker exec ollama ollama pull qwen3.5:27b
   ```
2. **OpenClaw** (pin a tested version; its configuration format changes between releases):
   ```bash
   npm install -g openclaw@2026.9.33
   ```

## Usage

### 1. Start the server on the edge device

```bash
export HAILO_AGENT_TOKEN="$(openssl rand -hex 16)"   # share this value with the agent host
hailo-agent-server --host 0.0.0.0 --port 8765 --allowed-host <device-ip>:8765
```

- `--allowed-host` lists the `Host` header values clients may use; requests with any other host are
  rejected (DNS-rebinding protection). Add the address the agent host connects to.
- `--backend mock` runs without a Hailo device, serving a synthetic scene.
- Requests without `Authorization: Bearer $HAILO_AGENT_TOKEN` get `401`.

### 2. Configure OpenClaw

Add the model, the MCP server and the tool settings to `~/.openclaw/openclaw.json`:

```json5
{
  models: {
    providers: {
      ollama: {
        baseUrl: "http://127.0.0.1:11434",   // native API, not /v1
        api: "ollama",
        apiKey: "ollama-local",
        // input "image" lets the model see snapshot results
        models: [{ id: "qwen3.5:27b", name: "qwen3.5:27b", input: ["text", "image"] }],
      },
    },
  },
  agents: { defaults: { model: { primary: "ollama/qwen3.5:27b" } } },
  mcp: {
    servers: {
      hailo: {
        url: "http://<device-ip>:8765/mcp",
        transport: "streamable-http",
        headers: { Authorization: "Bearer <HAILO_AGENT_TOKEN>" },
        requestTimeoutMs: 120000,   // watch_scene with a VLM caption can take ~1 min
      },
    },
  },
  tools: {
    // Direct tool schemas: the default tool-search bridge turns image results into base64 text.
    toolSearch: false,
    // Optional, recommended: shell/file/web tools are not needed and shrink the prompt.
    deny: ["group:runtime", "group:fs", "group:web", "group:ui", "group:sessions"],
  },
  skills: { allowBundled: [] },
}
```

Then check the connection and open the dashboard:

```bash
openclaw mcp probe hailo        # expect: hailo: 8 tools
openclaw gateway                # dashboard at http://localhost:18789
```

If the device is not directly reachable from the agent host, forward its port over SSH, e.g.
`ssh -N -R 8765:<device-ip>:8765 <agent-host>` from a machine that can reach both, and point
`url` at the forwarded port.

### 3. Ask the agent

- *"Which Hailo device is attached and what can it run?"*
- *"How many people are in front of the camera right now?"*
- *"Take a quick look at example.mp4 and describe what you see."*
- *"Watch the camera for 10 seconds and tell me if anyone carries a backpack."*

## Performance

Measured on a Raspberry Pi 5 + Hailo-10H with `qwen3.5:27b` on an RTX 5000 Ada:

| Step | Time |
|------|------|
| `get_detections` / `snapshot` on the device | 3 ms / 20-25 ms |
| Network and tunnel per call (VPN) | 70-130 ms |
| Pipeline start to first detections | ~1.8 s |
| `describe_scene` with the VLM loaded | ~10 s (the first call after boot also loads the model) |
| One LLM step of the agent | 2-7 s (about 19 tokens/s generation) |

Most of an agent turn is LLM time, and every tool call is one more LLM step, so prefer
`watch_scene` for one-off questions.

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| Pipelines run but report no detections | `/usr/local/hailo/resources/.env` is missing; run `hailo-set-env` |
| `No USB camera found for '--input usb'` | Check `cameras` in `hailo_status`; a Raspberry Pi ribbon camera is `rpi`, not `usb` |
| The agent describes snapshots wrongly or says it got base64 | Set `input: ["text", "image"]` on the model and `tools.toolSearch: false` |
| `describe_scene`: VLM model not installed | Install the `vlm_chat` model; the server does not download it |
| OpenClaw: `failed to start server "hailo"` | The server or the network path to it is down; check `openclaw mcp probe hailo` |
| Camera images are very dark or blue | Low light confuses auto white balance; check with `rpicam-still` in a lit scene |

## Files

- `server.py` - MCP tools, bearer-token check and the `hailo-agent-server` command
- `backend.py` - device identification, camera listing, pipelines and VLM per backend (Hailo / mock)
- `pipelines.py` - pipeline lifecycle, input validation and detection history
- `pipeline_runner.py` - child process that runs a Hailo-Apps pipeline headless and prints detections
- `imaging.py` - crop, annotate, resize and encode frames
- `apps.py` - the app catalogue shown by `list_apps`
