"""Catalogue of the Hailo-Apps applications the agent server can offer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from hailo_apps.config.config_manager import get_default_model_name, get_supported_architectures
from hailo_apps.python.agent_server.pipelines import PIPELINE_TARGETS
from hailo_apps.python.core.common.defines import (
    CLIP_PIPELINE,
    DEPTH_PIPELINE,
    DETECTION_PIPELINE,
    FACE_RECOGNITION_PIPELINE,
    INSTANCE_SEGMENTATION_PIPELINE,
    LLM_CHAT_APP,
    PADDLE_OCR_PIPELINE,
    POSE_ESTIMATION_PIPELINE,
    REID_MULTISOURCE_PIPELINE,
    SIMPLE_DETECTION_PIPELINE,
    TILING_PIPELINE,
    VLM_CHAT_APP,
    WHISPER_CHAT_APP,
)

AppKind = Literal["pipeline", "genai"]

# defines.py has no pipeline-name constant for multisource; this is its resources_config key.
MULTISOURCE_APP = "multisource"


@dataclass(frozen=True)
class AppSpec:
    """One application the agent can be told about.

    Attributes:
        name: Key of the app in ``resources_config.yaml``.
        kind: ``pipeline`` for GStreamer vision apps, ``genai`` for on-device LLM/VLM/ASR.
        command: Console script that launches it, or None for apps without one.
        description: One line for the agent, phrased as what the app gives it.
    """

    name: str
    kind: AppKind
    command: str | None
    description: str


APP_CATALOG: tuple[AppSpec, ...] = (
    AppSpec(
        DETECTION_PIPELINE,
        "pipeline",
        "hailo-detect",
        "Object detection (COCO classes) with tracking.",
    ),
    AppSpec(
        SIMPLE_DETECTION_PIPELINE,
        "pipeline",
        "hailo-detect-simple",
        "Lightweight object detection without tracking.",
    ),
    AppSpec(
        POSE_ESTIMATION_PIPELINE,
        "pipeline",
        "hailo-pose",
        "Human pose estimation (17 body keypoints per person).",
    ),
    AppSpec(
        INSTANCE_SEGMENTATION_PIPELINE,
        "pipeline",
        "hailo-seg",
        "Per-object instance segmentation masks.",
    ),
    AppSpec(DEPTH_PIPELINE, "pipeline", "hailo-depth", "Monocular depth estimation."),
    AppSpec(
        FACE_RECOGNITION_PIPELINE,
        "pipeline",
        "hailo-face-recon",
        "Face detection and recognition against a local face database.",
    ),
    AppSpec(
        TILING_PIPELINE,
        "pipeline",
        "hailo-tiling",
        "Detection of small objects in high-resolution frames by tiling.",
    ),
    AppSpec(
        MULTISOURCE_APP,
        "pipeline",
        "hailo-multisource",
        "Object detection over several video sources at once.",
    ),
    AppSpec(
        REID_MULTISOURCE_PIPELINE,
        "pipeline",
        "hailo-reid",
        "Person re-identification across several video sources.",
    ),
    AppSpec(
        CLIP_PIPELINE,
        "pipeline",
        "hailo-clip",
        "Zero-shot matching of the scene against free-text prompts (CLIP).",
    ),
    AppSpec(PADDLE_OCR_PIPELINE, "pipeline", "hailo-ocr", "Text detection and recognition (OCR)."),
    AppSpec(
        VLM_CHAT_APP,
        "genai",
        None,
        "On-device vision-language model that describes an image or answers about it.",
    ),
    AppSpec(LLM_CHAT_APP, "genai", None, "On-device small LLM for text chat."),
    AppSpec(WHISPER_CHAT_APP, "genai", None, "On-device speech-to-text (Whisper)."),
)


def list_apps(arch: str | None) -> list[dict[str, Any]]:
    """List the catalogue apps that have models for an architecture.

    Args:
        arch: Hailo-Apps architecture name, or None when no device is attached, in which case
            every app is listed with the architectures it supports.

    Returns:
        One dict per app with its name, kind, description and whether ``start_pipeline`` can run
        it (``startable``), plus the default model for ``arch``, or the supported architectures
        when ``arch`` is None. Launch commands are left out: the agent cannot run them.
    """
    apps: list[dict[str, Any]] = []
    for spec in APP_CATALOG:
        supported = get_supported_architectures(spec.name)
        if arch is not None and arch not in supported:
            continue
        entry: dict[str, Any] = {
            "name": spec.name,
            "kind": spec.kind,
            "description": spec.description,
            "startable": spec.name in PIPELINE_TARGETS,
        }
        if arch is None:
            entry["supported_archs"] = supported
        else:
            entry["default_model"] = get_default_model_name(spec.name, arch)
        apps.append(entry)
    return apps
