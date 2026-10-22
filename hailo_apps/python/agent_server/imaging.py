"""Frame handling for agent tools: crop, annotate, downscale, encode, and mock rendering."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import cv2
import numpy as np

MAX_SIDE_PX = 1280
MOCK_FRAME_SIZE = (1280, 720)
CROP_MARGIN = 0.05
JPEG_QUALITY = 85

_COLORS_BGR = {"person": (255, 160, 0), "car": (40, 40, 220)}
_DEFAULT_COLOR_BGR = (60, 200, 60)


def validate_bbox(bbox: Sequence[float] | None) -> tuple[float, float, float, float] | None:
    """Check a normalised ``[xmin, ymin, xmax, ymax]`` box.

    Args:
        bbox: The box, or None for the whole frame.

    Returns:
        The box as a tuple, or None.

    Raises:
        ValueError: If the box does not have four values in [0, 1] with min < max.
    """
    if bbox is None:
        return None
    if len(bbox) != 4:
        raise ValueError("crop must be [xmin, ymin, xmax, ymax]")
    xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    if not (0.0 <= xmin < xmax <= 1.0 and 0.0 <= ymin < ymax <= 1.0):
        raise ValueError("crop values must be normalised to [0, 1] with min < max")
    return xmin, ymin, xmax, ymax


def crop(image: np.ndarray, bbox: Sequence[float], margin: float = CROP_MARGIN) -> np.ndarray:
    """Cut a normalised box out of an image, widened by a margin on each side.

    Args:
        image: Frame in BGR.
        bbox: Normalised ``[xmin, ymin, xmax, ymax]``.
        margin: Fraction of the frame added around the box, clipped to the frame.

    Returns:
        The cropped region.
    """
    h, w = image.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    x0 = int(max(0.0, xmin - margin) * w)
    y0 = int(max(0.0, ymin - margin) * h)
    x1 = max(int(min(1.0, xmax + margin) * w), x0 + 1)
    y1 = max(int(min(1.0, ymax + margin) * h), y0 + 1)
    return image[y0:y1, x0:x1]


def annotate(image: np.ndarray, objects: Iterable[dict[str, Any]]) -> np.ndarray:
    """Draw detection boxes and labels on a copy of an image.

    Args:
        image: Frame in BGR.
        objects: Detections with ``label``, ``confidence``, normalised ``bbox`` and optional
            ``track_id``.

    Returns:
        The annotated copy.
    """
    out = image.copy()
    h, w = out.shape[:2]
    thickness = max(1, round(min(h, w) / 300))
    for obj in objects:
        xmin, ymin, xmax, ymax = obj["bbox"]
        p0, p1 = (int(xmin * w), int(ymin * h)), (int(xmax * w), int(ymax * h))
        color = _COLORS_BGR.get(obj["label"], _DEFAULT_COLOR_BGR)
        cv2.rectangle(out, p0, p1, color, thickness)
        text = f"{obj['label']} {obj['confidence']:.2f}"
        if "track_id" in obj:
            text = f"#{obj['track_id']} {text}"
        cv2.putText(
            out,
            text,
            (p0[0], max(p0[1] - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * thickness,
            color,
            thickness,
        )
    return out


def fit(image: np.ndarray, max_side: int = MAX_SIDE_PX) -> np.ndarray:
    """Downscale an image so its longer side is at most ``max_side``; never upscale.

    Args:
        image: The image.
        max_side: Longest allowed side in pixels.

    Returns:
        The image, resized if it was larger.
    """
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def encode_jpeg(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Encode a BGR image as JPEG.

    Args:
        image: The image.
        quality: JPEG quality, 1-100.

    Returns:
        The JPEG bytes.

    Raises:
        ValueError: If OpenCV cannot encode the image.
    """
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode frame as JPEG")
    return buf.tobytes()


def render_mock_frame(
    objects: Iterable[dict[str, Any]], size: tuple[int, int] = MOCK_FRAME_SIZE
) -> np.ndarray:
    """Draw a synthetic frame: flat background with a filled block per object.

    Args:
        objects: Detections to draw, as produced by the mock pipeline.
        size: Frame size as (width, height).

    Returns:
        The frame in BGR.
    """
    w, h = size
    frame = np.full((h, w, 3), 90, dtype=np.uint8)
    frame[int(h * 0.8) :, :] = (70, 110, 70)
    for obj in objects:
        xmin, ymin, xmax, ymax = obj["bbox"]
        color = _COLORS_BGR.get(obj["label"], _DEFAULT_COLOR_BGR)
        cv2.rectangle(
            frame,
            (int(xmin * w), int(ymin * h)),
            (int(xmax * w), int(ymax * h)),
            color,
            thickness=-1,
        )
    return frame
