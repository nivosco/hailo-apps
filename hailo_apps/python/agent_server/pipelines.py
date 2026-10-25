"""Running vision pipelines for the agent: lifecycle, input validation and detection history."""

from __future__ import annotations

import json
import math
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from hailo_apps.python.agent_server import imaging
from hailo_apps.python.core.common.defines import (
    DETECTION_PIPELINE,
    INSTANCE_SEGMENTATION_PIPELINE,
    POSE_ESTIMATION_PIPELINE,
    RESOURCES_ROOT_PATH_DEFAULT,
    RESOURCES_VIDEOS_DIR_NAME,
    SIMPLE_DETECTION_PIPELINE,
    TILING_PIPELINE,
)

# Apps whose callback yields HAILO_DETECTION objects, mapped to (module, app class).
PIPELINE_TARGETS: dict[str, tuple[str, str]] = {
    DETECTION_PIPELINE: (
        "hailo_apps.python.pipeline_apps.detection.detection_pipeline",
        "GStreamerDetectionApp",
    ),
    SIMPLE_DETECTION_PIPELINE: (
        "hailo_apps.python.pipeline_apps.detection_simple.detection_simple_pipeline",
        "GStreamerDetectionSimpleApp",
    ),
    POSE_ESTIMATION_PIPELINE: (
        "hailo_apps.python.pipeline_apps.pose_estimation.pose_estimation_pipeline",
        "GStreamerPoseEstimationApp",
    ),
    INSTANCE_SEGMENTATION_PIPELINE: (
        "hailo_apps.python.pipeline_apps.instance_segmentation.instance_segmentation_pipeline",
        "GStreamerInstanceSegmentationApp",
    ),
    TILING_PIPELINE: (
        "hailo_apps.python.pipeline_apps.tiling.tiling_pipeline",
        "GStreamerTilingApp",
    ),
}

DETECTIONS_MARKER = "@@hailo-dets "
# The runner keeps the newest frame on disk: the JPEG, then a sidecar with that frame's
# detections, each replaced atomically.
FRAME_IMAGE_NAME = "latest.jpg"
FRAME_META_NAME = "latest.json"
SNAPSHOT_INTERVAL_S = 1.0
CAMERA_SOURCES = ("usb", "rpi")
STREAM_PREFIXES = ("rtsp://", "rtsps://")
VIDEO_SUFFIXES = (".mp4", ".mkv", ".avi", ".mov")
DEFAULT_SOURCE = "example.mp4"
MAX_WINDOW_S = 120.0
MAX_LABELS = 20
# Tool results are re-read by the agent's LLM on every later step, so per-frame object lists
# are capped and rounded; the counts still cover every object.
MAX_REPORTED_OBJECTS = 15
MIN_FPS, MAX_FPS = 1, 60
# Generous on purpose: the app's own SIGINT shutdown drains the device, and killing a process
# that still has DMA in flight can fault the host driver.
STOP_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class FrameDetections:
    """Objects seen in one frame.

    Attributes:
        timestamp: ``time.monotonic()`` when the frame's detections arrived.
        frame: Frame counter reported by the pipeline.
        objects: One dict per object: ``label``, ``confidence``, normalised
            ``bbox`` ``[xmin, ymin, xmax, ymax]`` and, when tracked, ``track_id``.
    """

    timestamp: float
    frame: int
    objects: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FrameImage:
    """The newest captured frame of a pipeline.

    Attributes:
        image: The frame in BGR.
        frame: Frame counter reported by the pipeline.
        objects: Detections in this frame, in the :class:`FrameDetections` format.
        captured_at: Wall-clock ``time.time()`` when the frame was captured.
    """

    image: np.ndarray
    frame: int
    objects: tuple[dict[str, Any], ...]
    captured_at: float

    @property
    def age_s(self) -> float:
        """Seconds since the frame was captured."""
        return round(time.time() - self.captured_at, 2)


class DetectionBuffer:
    """Thread-safe history of recent frames, bounded by age and by frame count."""

    def __init__(self, max_age_s: float = MAX_WINDOW_S, max_frames: int = 5000) -> None:
        """Create an empty buffer.

        Args:
            max_age_s: Frames older than this are dropped.
            max_frames: Hard cap on stored frames, whatever their age.
        """
        self._max_age_s = max_age_s
        self._frames: deque[FrameDetections] = deque(maxlen=max_frames)
        self._lock = threading.Lock()

    def add(self, frame: FrameDetections) -> None:
        """Append a frame and drop the ones that aged out.

        Args:
            frame: The frame to store.
        """
        with self._lock:
            self._frames.append(frame)
            horizon = frame.timestamp - self._max_age_s
            while self._frames and self._frames[0].timestamp < horizon:
                self._frames.popleft()

    def summary(
        self, window_s: float, labels: Iterable[str] | None = None, now: float | None = None
    ) -> dict[str, Any]:
        """Summarise the frames seen within a time window.

        Args:
            window_s: How far back to look, in seconds.
            labels: Only count objects with these labels; all labels when None or empty.
            now: Reference ``time.monotonic()`` value; defaults to the current time.

        Returns:
            ``frames_in_window``, the ``latest`` frame (its objects and age) and per-label
            ``counts``: objects in the latest frame, the peak in any single frame, and the
            number of distinct track ids.
        """
        now = time.monotonic() if now is None else now
        wanted = set(labels or ())
        with self._lock:
            frames = [f for f in self._frames if f.timestamp >= now - window_s]

        def _keep(obj: dict[str, Any]) -> bool:
            return not wanted or obj["label"] in wanted

        counts: dict[str, dict[str, int]] = {}
        tracks: dict[str, set[int]] = {}
        for f in frames:
            per_frame = Counter(o["label"] for o in f.objects if _keep(o))
            for label, n in per_frame.items():
                entry = counts.setdefault(label, {"latest": 0, "max_in_frame": 0})
                entry["max_in_frame"] = max(entry["max_in_frame"], n)
            for o in f.objects:
                if _keep(o) and "track_id" in o:
                    tracks.setdefault(o["label"], set()).add(o["track_id"])
        latest = frames[-1] if frames else None
        if latest is not None:
            for label, n in Counter(o["label"] for o in latest.objects if _keep(o)).items():
                counts[label]["latest"] = n
        for label, ids in tracks.items():
            counts[label]["unique_tracks"] = len(ids)
        return {
            "window_s": window_s,
            "frames_in_window": len(frames),
            "latest": None
            if latest is None
            else {
                "frame": latest.frame,
                "age_s": round(now - latest.timestamp, 2),
                **compact_objects([o for o in latest.objects if _keep(o)]),
            },
            "counts": counts,
        }


class PipelineHandle(Protocol):
    """A started pipeline."""

    pipeline_id: str
    app: str
    source: str
    labels: tuple[str, ...]
    started_at: float
    buffer: DetectionBuffer

    def is_running(self) -> bool:
        """Return whether the pipeline is still producing frames."""
        ...

    def error(self) -> str | None:
        """Return why the pipeline stopped on its own, if it did."""
        ...

    def latest_frame(self) -> FrameImage | None:
        """Return the newest captured frame, or None before the first one."""
        ...

    def stop(self, timeout_s: float = STOP_TIMEOUT_S) -> dict[str, Any]:
        """Stop the pipeline and report how.

        Args:
            timeout_s: How long to wait for a graceful stop.

        Returns:
            At least ``stopped`` (bool) and ``forced`` (bool).
        """
        ...


def compact_objects(
    objects: Sequence[dict[str, Any]], limit: int = MAX_REPORTED_OBJECTS
) -> dict[str, Any]:
    """Shrink a frame's object list for tool output.

    Keeps the ``limit`` most confident objects, with confidence rounded to two decimals and the
    bbox to three.

    Args:
        objects: Detections in the frame.
        limit: Most objects to keep.

    Returns:
        ``objects`` and, when some were dropped, ``objects_total``.
    """
    ranked = sorted(objects, key=lambda o: o["confidence"], reverse=True)[:limit]
    kept = [
        {
            **o,
            "confidence": round(o["confidence"], 2),
            "bbox": [round(v, 3) for v in o["bbox"]],
        }
        for o in ranked
    ]
    out: dict[str, Any] = {"objects": kept}
    if len(objects) > limit:
        out["objects_total"] = len(objects)
    return out


def describe(handle: PipelineHandle) -> dict[str, Any]:
    """Describe a pipeline for tool output.

    Args:
        handle: The pipeline.

    Returns:
        Its id, app, source (a video's file name rather than its path), uptime and running
        state, plus the label filter and error when there are any.
    """
    out: dict[str, Any] = {
        "pipeline_id": handle.pipeline_id,
        "app": handle.app,
        "source": Path(handle.source).name if handle.source.startswith("/") else handle.source,
        "uptime_s": round(time.monotonic() - handle.started_at, 1),
        "running": handle.is_running(),
    }
    if handle.labels:
        out["labels"] = list(handle.labels)
    if (error := handle.error()) is not None:
        out["error"] = error
    return out


def resolve_source(source: str) -> str:
    """Validate an input source and turn a bundled video name into its path.

    Only cameras, RTSP streams and videos in the Hailo-Apps resources directory are accepted,
    so the agent cannot point a pipeline at arbitrary files.

    Args:
        source: ``usb``, ``rpi``, an ``rtsp://`` URL, or a bare video file name such as
            ``example.mp4``.

    Returns:
        The value to pass as the pipeline's ``--input``.

    Raises:
        ValueError: If the source is not one of the accepted forms.
    """
    if source in CAMERA_SOURCES or source.startswith(STREAM_PREFIXES):
        return source
    if Path(source).name != source or source.startswith("."):
        raise ValueError(f"source must be a bare video file name, not a path: {source!r}")
    if not source.lower().endswith(VIDEO_SUFFIXES):
        raise ValueError(
            f"unsupported source {source!r}; use one of {CAMERA_SOURCES}, an rtsp:// URL, "
            f"or a video file name ending in {VIDEO_SUFFIXES}"
        )
    return str(Path(RESOURCES_ROOT_PATH_DEFAULT) / RESOURCES_VIDEOS_DIR_NAME / source)


def parse_detections_line(line: str, timestamp: float) -> FrameDetections | None:
    """Parse one line of runner output.

    Args:
        line: A line from the runner's stdout.
        timestamp: Arrival time to record for the frame.

    Returns:
        The frame, or None if the line is ordinary app output or malformed.
    """
    if not line.startswith(DETECTIONS_MARKER):
        return None
    try:
        payload = json.loads(line[len(DETECTIONS_MARKER) :])
        return FrameDetections(
            timestamp=timestamp,
            frame=int(payload["frame"]),
            objects=tuple(payload["objects"]),
        )
    except (ValueError, KeyError, TypeError):
        return None


@dataclass
class SubprocessPipeline:
    """A Hailo-Apps pipeline running in a child process via ``pipeline_runner``."""

    app: str
    source: str
    labels: tuple[str, ...]
    command: Sequence[str]
    frame_dir: Path | None = None
    pipeline_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    started_at: float = field(default_factory=time.monotonic)
    buffer: DetectionBuffer = field(default_factory=DetectionBuffer)
    _log: deque[str] = field(default_factory=lambda: deque(maxlen=50))
    _proc: subprocess.Popen[str] | None = None
    _reader: threading.Thread | None = None

    def start(self) -> SubprocessPipeline:
        """Launch the child process and start collecting its detections.

        Returns:
            self, for chaining.
        """
        self._proc = subprocess.Popen(
            list(self.command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        return self

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            frame = parse_detections_line(line, time.monotonic())
            if frame is not None:
                self.buffer.add(frame)
            elif line.strip():
                self._log.append(line.rstrip())

    def is_running(self) -> bool:
        """Return whether the child process is alive."""
        return self._proc is not None and self._proc.poll() is None

    def error(self) -> str | None:
        """Return the exit code and last output lines if the child exited by itself."""
        if self._proc is None or self._proc.returncode in (None, 0):
            return None
        tail = " | ".join(list(self._log)[-5:])
        return f"exited with {self._proc.returncode}: {tail}"

    def latest_frame(self) -> FrameImage | None:
        """Read the newest frame the runner wrote to ``frame_dir``."""
        if self.frame_dir is None:
            return None
        try:
            meta = json.loads((self.frame_dir / FRAME_META_NAME).read_text())
        except (OSError, ValueError):
            return None
        image = cv2.imread(str(self.frame_dir / FRAME_IMAGE_NAME))
        if image is None:
            return None
        return FrameImage(
            image=image,
            frame=int(meta["frame"]),
            objects=tuple(meta["objects"]),
            captured_at=float(meta["captured_at"]),
        )

    def stop(self, timeout_s: float = STOP_TIMEOUT_S) -> dict[str, Any]:
        """Stop the child with SIGINT, escalating only if it does not exit in time.

        Args:
            timeout_s: How long to wait after SIGINT before escalating.

        Returns:
            ``stopped``, ``forced`` and the child's ``exit_code``.
        """
        if self._proc is None:
            return {"stopped": True, "forced": False, "exit_code": None}
        forced = False
        if self._proc.poll() is None:
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                forced = True
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
        if self._reader is not None:
            self._reader.join(timeout=2)
        if self.frame_dir is not None:
            shutil.rmtree(self.frame_dir, ignore_errors=True)
        return {"stopped": True, "forced": forced, "exit_code": self._proc.returncode}


def runner_command(
    app: str, source: str, fps: int | None, frame_dir: Path | None = None
) -> list[str]:
    """Build the command that runs an app under ``pipeline_runner``.

    Args:
        app: Key of :data:`PIPELINE_TARGETS`.
        source: Resolved ``--input`` value.
        fps: Frame rate to request, or None for the app default.
        frame_dir: Where the runner keeps the newest frame; no frames are saved when None.

    Returns:
        The argv for the child process.
    """
    cmd = [
        sys.executable,
        "-m",
        "hailo_apps.python.agent_server.pipeline_runner",
        "--app",
        app,
        *(["--frame-dir", str(frame_dir)] if frame_dir is not None else []),
        "--",
        "--input",
        source,
    ]
    if fps is not None:
        cmd += ["--frame-rate", str(fps)]
    return cmd


@dataclass
class MockPipeline:
    """Synthetic scene for hardware-free runs: people walking through, a car now and then."""

    app: str
    source: str
    labels: tuple[str, ...]
    fps: int = 10
    pipeline_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    started_at: float = field(default_factory=time.monotonic)
    buffer: DetectionBuffer = field(default_factory=DetectionBuffer)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> MockPipeline:
        """Start generating frames.

        Returns:
            self, for chaining.
        """
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    @staticmethod
    def scene(t: float) -> tuple[dict[str, Any], ...]:
        """Return the objects visible ``t`` seconds into the scene.

        One to three people walk across the frame, each with a stable track id, and a car is
        parked for the first 8 s of every 20 s.

        Args:
            t: Seconds since the pipeline started.

        Returns:
            The objects in that frame.
        """
        objects: list[dict[str, Any]] = []
        people = 1 + int(t // 10) % 3
        for i in range(people):
            x = (0.1 + 0.25 * i + 0.02 * t) % 0.8
            y = 0.3 + 0.1 * math.sin(t / 3 + i)
            objects.append(
                {
                    "label": "person",
                    "confidence": 0.9,
                    "bbox": [round(x, 3), round(y, 3), round(x + 0.12, 3), round(y + 0.4, 3)],
                    "track_id": i + 1,
                }
            )
        if t % 20 < 8:
            objects.append(
                {
                    "label": "car",
                    "confidence": 0.85,
                    "bbox": [0.55, 0.55, 0.9, 0.85],
                    "track_id": 100 + int(t // 20),
                }
            )
        return tuple(objects)

    def _run(self) -> None:
        frame = 0
        while not self._stop.wait(1 / self.fps):
            now = time.monotonic()
            frame += 1
            self.buffer.add(FrameDetections(now, frame, self.scene(now - self.started_at)))

    def is_running(self) -> bool:
        """Return whether frames are still being generated."""
        return self._thread is not None and self._thread.is_alive()

    def error(self) -> str | None:
        """Return None; the mock never fails on its own."""
        return None

    def latest_frame(self) -> FrameImage | None:
        """Render the newest frame of the synthetic scene."""
        summary = self.buffer.summary(window_s=MAX_WINDOW_S)
        latest = summary["latest"]
        if latest is None:
            return None
        objects = tuple(latest["objects"])
        return FrameImage(
            image=imaging.render_mock_frame(objects),
            frame=latest["frame"],
            objects=objects,
            captured_at=time.time() - latest["age_s"],
        )

    def stop(self, timeout_s: float = STOP_TIMEOUT_S) -> dict[str, Any]:
        """Stop generating frames.

        Args:
            timeout_s: How long to wait for the generator thread.

        Returns:
            ``stopped`` and ``forced`` (always False).
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        return {"stopped": True, "forced": False}


class PipelineManager:
    """Starts, tracks and stops pipelines, enforcing a cap on how many run at once."""

    def __init__(self, max_running: int = 1) -> None:
        """Create the manager.

        Args:
            max_running: Pipelines allowed at the same time; one accelerator rarely runs more.
        """
        self._max_running = max_running
        self._pipelines: dict[str, PipelineHandle] = {}
        self._lock = threading.Lock()

    def add(self, handle: PipelineHandle) -> PipelineHandle:
        """Register a started pipeline.

        Args:
            handle: The pipeline.

        Returns:
            The same handle.
        """
        with self._lock:
            self._pipelines[handle.pipeline_id] = handle
        return handle

    def check_capacity(self) -> None:
        """Raise if another pipeline may not be started now.

        Raises:
            ValueError: If the running cap is reached; names the running pipelines.
        """
        running = [h.pipeline_id for h in self.list() if h.is_running()]
        if len(running) >= self._max_running:
            raise ValueError(
                f"pipeline(s) {running} already running; stop_pipeline first "
                f"(limit {self._max_running})"
            )

    def get(self, pipeline_id: str) -> PipelineHandle:
        """Look up a pipeline.

        Args:
            pipeline_id: Id returned by start.

        Returns:
            The pipeline.

        Raises:
            ValueError: If no pipeline has that id.
        """
        with self._lock:
            handle = self._pipelines.get(pipeline_id)
        if handle is None:
            raise ValueError(f"unknown pipeline_id {pipeline_id!r}")
        return handle

    def list(self) -> list[PipelineHandle]:
        """Return every tracked pipeline, running or not."""
        with self._lock:
            return list(self._pipelines.values())

    def stop(self, pipeline_id: str) -> dict[str, Any]:
        """Stop a pipeline and forget it.

        Args:
            pipeline_id: Id returned by start.

        Returns:
            The pipeline's stop report plus its id.

        Raises:
            ValueError: If no pipeline has that id.
        """
        handle = self.get(pipeline_id)
        report = handle.stop()
        with self._lock:
            self._pipelines.pop(pipeline_id, None)
        return {"pipeline_id": pipeline_id, **report}

    def stop_all(self) -> None:
        """Stop every tracked pipeline."""
        for handle in self.list():
            self.stop(handle.pipeline_id)


def validate_labels(labels: Sequence[str] | None) -> tuple[str, ...]:
    """Normalise a label filter.

    Args:
        labels: Labels to keep, or None for all.

    Returns:
        The labels as a tuple.

    Raises:
        ValueError: If more than :data:`MAX_LABELS` labels are given.
    """
    if not labels:
        return ()
    if len(labels) > MAX_LABELS:
        raise ValueError(f"at most {MAX_LABELS} labels may be given")
    return tuple(str(label) for label in labels)


def validate_fps(fps: int | None) -> int | None:
    """Check a requested frame rate.

    Args:
        fps: Frames per second, or None for the app default.

    Returns:
        The same value.

    Raises:
        ValueError: If outside :data:`MIN_FPS`..:data:`MAX_FPS`.
    """
    if fps is not None and not MIN_FPS <= fps <= MAX_FPS:
        raise ValueError(f"fps must be between {MIN_FPS} and {MAX_FPS}")
    return fps
