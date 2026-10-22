"""Child-process entry point: runs one Hailo-Apps pipeline and prints its detections as JSON.

Usage: ``python -m hailo_apps.python.agent_server.pipeline_runner --app detection
[--frame-dir DIR] -- <app args>``. Each frame becomes one stdout line, ``DETECTIONS_MARKER``
followed by ``{"frame": N, "objects": [...]}``; everything else the app prints is ignored by the
server. With ``--frame-dir``, the newest frame is also saved there about once a second.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hailo_apps.python.agent_server.pipelines import (
    DETECTIONS_MARKER,
    FRAME_IMAGE_NAME,
    FRAME_META_NAME,
    PIPELINE_TARGETS,
    SNAPSHOT_INTERVAL_S,
)
from hailo_apps.python.core.common.defines import (
    HAILO_NV12_VIDEO_FORMAT,
    HAILO_RGB_VIDEO_FORMAT,
    HAILO_YUYV_VIDEO_FORMAT,
)

HEADLESS_VIDEO_SINK = "fakesink"


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _frame_to_bgr(buffer: Any, pad: Any) -> Any:
    """Copy the buffer's pixels into a BGR array, or None for an unsupported format."""
    import cv2
    import numpy as np

    from hailo_apps.python.core.common.buffer_utils import (
        get_caps_from_pad,
        get_numpy_from_buffer,
    )

    fmt, width, height = get_caps_from_pad(pad)
    if fmt is None:
        return None
    pixels = get_numpy_from_buffer(buffer, fmt, width, height)
    if fmt == HAILO_RGB_VIDEO_FORMAT:
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if fmt == HAILO_YUYV_VIDEO_FORMAT:
        return cv2.cvtColor(pixels, cv2.COLOR_YUV2BGR_YUYV)
    if fmt == HAILO_NV12_VIDEO_FORMAT:
        y_plane, uv_plane = pixels
        nv12 = np.concatenate([y_plane, uv_plane.reshape(height // 2, width)])
        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    return None


class _Snapshotter:
    """Saves the newest frame and its detections to a directory at a bounded rate."""

    def __init__(self, frame_dir: Path, interval_s: float = SNAPSHOT_INTERVAL_S) -> None:
        self._dir = frame_dir
        self._interval_s = interval_s
        self._last = 0.0

    def maybe_save(self, element: Any, buffer: Any, frame: int, objects: list[dict]) -> None:
        now = time.monotonic()
        if now - self._last < self._interval_s:
            return
        import cv2

        image = _frame_to_bgr(buffer, element.get_static_pad("src"))
        if image is None:
            return
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return
        self._last = now
        _write_atomic(self._dir / FRAME_IMAGE_NAME, jpeg.tobytes())
        meta = {"frame": frame, "objects": objects, "captured_at": time.time()}
        _write_atomic(self._dir / FRAME_META_NAME, json.dumps(meta).encode())


def _make_callback(snapshotter: _Snapshotter | None) -> Any:
    def _detections_callback(element: Any, buffer: Any, user_data: Any) -> None:
        import hailo

        if buffer is None:
            return
        objects = []
        for det in hailo.get_roi_from_buffer(buffer).get_objects_typed(hailo.HAILO_DETECTION):
            bbox = det.get_bbox()
            obj: dict[str, Any] = {
                "label": det.get_label(),
                "confidence": round(det.get_confidence(), 3),
                "bbox": [
                    round(bbox.xmin(), 4),
                    round(bbox.ymin(), 4),
                    round(bbox.xmin() + bbox.width(), 4),
                    round(bbox.ymin() + bbox.height(), 4),
                ],
            }
            track = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if track:
                obj["track_id"] = track[0].get_id()
            objects.append(obj)
        frame = user_data.get_count()
        sys.stdout.write(f"{DETECTIONS_MARKER}{json.dumps({'frame': frame, 'objects': objects})}\n")
        sys.stdout.flush()
        if snapshotter is not None:
            snapshotter.maybe_save(element, buffer, frame, objects)

    return _detections_callback


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested pipeline until it is interrupted.

    Args:
        argv: ``--app <name> [--frame-dir DIR] -- <app args>``; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(prog="hailo-agent-pipeline-runner")
    parser.add_argument("--app", required=True, choices=sorted(PIPELINE_TARGETS))
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("app_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    app_args = args.app_args[1:] if args.app_args[:1] == ["--"] else args.app_args

    from hailo_apps.python.core.gstreamer import gstreamer_app

    # The pipeline CLI has no headless option; every app reads its sink from this module constant
    # when it is constructed, so swapping it here keeps the device free of a display window.
    gstreamer_app.GST_VIDEO_SINK = HEADLESS_VIDEO_SINK

    module_name, class_name = PIPELINE_TARGETS[args.app]
    app_class = getattr(importlib.import_module(module_name), class_name)
    snapshotter = _Snapshotter(args.frame_dir) if args.frame_dir is not None else None
    # The app classes parse their options from sys.argv.
    sys.argv = [f"hailo-agent-{args.app}", *app_args]
    app_class(_make_callback(snapshotter), gstreamer_app.app_callback_class()).run()


if __name__ == "__main__":
    main()
