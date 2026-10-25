"""Child-process entry point: runs one Hailo-Apps standalone vision app and prints detections.

Usage: ``python -m hailo_apps.python.agent_server.pipeline_runner --app detection --input SRC
--arch ARCH --stop-file PATH [--parent-pid PID] [--fps N] [--frame-dir DIR]``.

Each processed frame becomes one stdout line, ``DETECTIONS_MARKER`` followed by
``{"frame": N, "objects": [...]}`` with boxes normalised to the frame. With ``--frame-dir``, the
newest frame and its detections are also saved there about once a second. The runner stops when
the stop file appears or the parent process is gone, then releases the camera and the device in
order. A file is used rather than a signal or stdin: Windows cannot send SIGINT to a child, and a
thread blocked reading a stdin pipe there stalls the child's own start-up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from hailo_apps.python.agent_server.pipelines import (
    DETECTIONS_MARKER,
    FRAME_IMAGE_NAME,
    FRAME_META_NAME,
    PIPELINE_TARGETS,
    SNAPSHOT_INTERVAL_S,
)

INFER_TIMEOUT_MS = 10000
# Pose models report a confidence per person; Hailo-Apps' pose app draws people above this.
POSE_SCORE_THRESHOLD = 0.5
PARENT_CHECK_INTERVAL_S = 1.0


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _normalise(box: Sequence[float], width: int, height: int) -> list[float]:
    xmin, ymin, xmax, ymax = box
    return [
        round(min(max(xmin / width, 0.0), 1.0), 4),
        round(min(max(ymin / height, 0.0), 1.0), 4),
        round(min(max(xmax / width, 0.0), 1.0), 4),
        round(min(max(ymax / height, 0.0), 1.0), 4),
    ]


class _Detector:
    """Object detection with ByteTrack ids, on Hailo-Apps' standalone detection code."""

    def __init__(self) -> None:
        from hailo_apps.python.core.common.toolbox import get_labels, load_json_file
        from hailo_apps.python.core.tracker.byte_tracker import BYTETracker
        from hailo_apps.python.standalone_apps.object_detection import (
            object_detection_post_process as post,
        )

        config_path = Path(post.__file__).with_name("config.json")
        self._config = load_json_file(str(config_path))
        self._labels = get_labels(None)
        self._post = post
        tracker_cfg = self._config.get("visualization_params", {}).get("tracker", {})
        self._tracker = BYTETracker(SimpleNamespace(**tracker_cfg))

    def objects(self, frame: np.ndarray, result: Any) -> list[dict[str, Any]]:
        if isinstance(result, dict):
            dets = self._post.decode_multioutput_detections(frame, result, self._config)
        else:
            dets = self._post.extract_detections(frame, result, self._config)
        height, width = frame.shape[:2]
        boxes = dets["detection_boxes"]
        objects = [
            {
                "label": self._labels[int(cls)],
                "confidence": round(float(score), 3),
                "bbox": _normalise(box, width, height),
            }
            for box, cls, score in zip(
                boxes, dets["detection_classes"], dets["detection_scores"], strict=True
            )
        ]
        if boxes:
            scores = dets["detection_scores"]
            tracks = self._tracker.update(
                np.array([[*box, score] for box, score in zip(boxes, scores, strict=True)])
            )
            for track in tracks:
                idx = self._post.find_best_matching_detection_index(track.tlbr, boxes)
                if idx is not None and idx >= 0:
                    objects[idx]["track_id"] = int(track.track_id)
        return objects


class _PoseEstimator:
    """Person boxes from Hailo-Apps' standalone pose-estimation post-processing."""

    def __init__(self, model_h: int, model_w: int) -> None:
        from hailo_apps.python.standalone_apps.pose_estimation.pose_estimation_utils import (
            PoseEstPostProcessing,
        )

        # Same settings as Hailo-Apps' pose_estimation app.
        self._post = PoseEstPostProcessing(
            max_detections=300,
            score_threshold=0.001,
            nms_iou_thresh=0.7,
            regression_length=15,
            strides=[8, 16, 32],
        )
        self._model_h, self._model_w = model_h, model_w

    def objects(self, frame: np.ndarray, result: Any) -> list[dict[str, Any]]:
        out = self._post.post_process(result, self._model_h, self._model_w, 1)
        height, width = frame.shape[:2]
        objects = []
        for box, score in zip(out["bboxes"][0], out["scores"][0], strict=True):
            if float(score[0]) < POSE_SCORE_THRESHOLD:
                continue
            mapped = self._post.map_box_to_original_coords(
                box, width, height, self._model_w, self._model_h
            )
            objects.append(
                {
                    "label": "person",
                    "confidence": round(float(score[0]), 3),
                    "bbox": _normalise(mapped, width, height),
                }
            )
        return objects


def _make_postprocess(app: str, model_h: int, model_w: int) -> Any:
    if app == "pose_estimation":
        return _PoseEstimator(model_h, model_w)
    return _Detector()


def _process_alive(pid: int) -> bool:
    """Return whether a process exists, without signalling it."""
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_condition(stop_file: Path, parent_pid: int | None) -> Callable[[], bool]:
    """Build the check the capture loop runs once per frame.

    Args:
        stop_file: Its appearance asks the runner to stop.
        parent_pid: Stop when this process is gone, so a crashed server does not leave the
            device busy; not checked when None.

    Returns:
        A function returning True once the runner should stop.
    """
    last_parent_check = 0.0

    def should_stop() -> bool:
        nonlocal last_parent_check
        if stop_file.exists():
            return True
        now = time.monotonic()
        if parent_pid is not None and now - last_parent_check >= PARENT_CHECK_INTERVAL_S:
            last_parent_check = now
            return not _process_alive(parent_pid)
        return False

    return should_stop


def _infer_once(hailo_infer: Any, model_input: np.ndarray) -> Any:
    """Run one synchronous inference and return the raw output (array or dict of arrays)."""
    box: dict[str, Any] = {}

    def _done(completion_info: Any, bindings_list: list) -> None:
        if completion_info.exception:
            box["error"] = completion_info.exception
            return
        bindings = bindings_list[0]
        names = bindings._output_names
        if len(names) == 1:
            box["result"] = bindings.output().get_buffer()
        else:
            box["result"] = {
                name: np.expand_dims(bindings.output(name).get_buffer(), axis=0) for name in names
            }

    hailo_infer.run([model_input], _done).wait(INFER_TIMEOUT_MS)
    if "error" in box:
        raise RuntimeError(f"inference failed: {box['error']}")
    return box["result"]


def run(
    app: str,
    source: str,
    arch: str,
    fps: float | None,
    frame_dir: Path | None,
    should_stop: Callable[[], bool],
    emit: Callable[[str], None],
) -> None:
    """Capture, infer and report until ``should_stop`` returns True.

    Args:
        app: Key of ``PIPELINE_TARGETS``.
        source: ``usb``, ``rpi``, a camera index, an ``rtsp://`` URL or a video path.
        arch: Hailo architecture, used to pick the model.
        fps: Frame rate cap; defaults to the source's own rate. Video files are paced to it.
        frame_dir: Where to keep the newest frame for snapshots, or None.
        should_stop: Polled once per frame; True finishes the frame and releases everything.
        emit: Receives one output line per frame.
    """
    from hailo_apps.python.core.common.core import resolve_hef_path
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
    from hailo_apps.python.core.common.toolbox import (
        InputContext,
        default_preprocess,
        init_input_source,
    )

    hef_path = resolve_hef_path(None, PIPELINE_TARGETS[app], arch)
    if hef_path is None:
        raise RuntimeError(f"no model found for {app} on {arch}")
    ctx = init_input_source(InputContext(input_src=source, batch_size=1, frame_rate=fps))
    if ctx.cap is None:
        raise RuntimeError(f"could not open input {source!r}")
    from hailo_platform import HEF

    # As in the standalone apps: raw multi-output models are read as float, HailoRT-NMS models
    # keep the HEF's output type.
    num_outputs = len(HEF(str(hef_path)).get_output_vstream_infos())
    output_type = "FLOAT32" if num_outputs > 1 else None
    hailo_infer = HailoInfer(str(hef_path), 1, output_type=output_type)
    try:
        model_h, model_w, _ = hailo_infer.get_input_shape()
        post = _make_postprocess(app, model_h, model_w)
        period = 1.0 / (fps or ctx.source_fps or 30.0)
        frame_no, last_save, next_due = 0, 0.0, time.monotonic()
        while not should_stop():
            ok, frame_bgr = ctx.cap.read()
            if not ok:
                if ctx.is_video:
                    ctx.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                raise RuntimeError(f"input {source!r} stopped delivering frames")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = _infer_once(hailo_infer, default_preprocess(frame_rgb, model_w, model_h))
            frame_no += 1
            objects = post.objects(frame_bgr, result)
            emit(f"{DETECTIONS_MARKER}{json.dumps({'frame': frame_no, 'objects': objects})}")
            now = time.monotonic()
            if frame_dir is not None and now - last_save >= SNAPSHOT_INTERVAL_S:
                ok, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    _write_atomic(frame_dir / FRAME_IMAGE_NAME, jpeg.tobytes())
                    meta = {"frame": frame_no, "objects": objects, "captured_at": time.time()}
                    _write_atomic(frame_dir / FRAME_META_NAME, json.dumps(meta).encode())
                    last_save = now
            # Cameras and streams pace themselves; video files would otherwise race ahead.
            if ctx.is_video:
                next_due += period
                time.sleep(max(0.0, next_due - time.monotonic()))
    finally:
        ctx.cap.release()
        hailo_infer.close()


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested app until the stop file appears or the parent exits.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(prog="hailo-agent-pipeline-runner")
    parser.add_argument("--app", required=True, choices=sorted(PIPELINE_TARGETS))
    parser.add_argument("--input", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, default=None)
    args = parser.parse_args(argv)
    should_stop = stop_condition(args.stop_file, args.parent_pid)

    def emit(line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    try:
        run(args.app, args.input, args.arch, args.fps, args.frame_dir, should_stop, emit)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
