"""Catalogue of the Hailo-Apps applications the agent server can offer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from hailo_apps.config.config_manager import get_default_model_name, get_supported_architectures
from hailo_apps.python.agent_server.pipelines import PIPELINE_TARGETS
from hailo_apps.python.core.common.defines import (
    DETECTION_PIPELINE,
    POSE_ESTIMATION_PIPELINE,
    VLM_CHAT_APP,
)

AppKind = Literal["vision", "genai"]


@dataclass(frozen=True)
class AppSpec:
    """One application the agent can be told about.

    Attributes:
        name: Name the agent uses, e.g. in ``start_pipeline``.
        kind: ``vision`` for apps ``start_pipeline`` runs, ``genai`` for on-device models.
        resource: Key of the app's models in ``resources_config.yaml``.
        description: One line for the agent, phrased as what the app gives it.
    """

    name: str
    kind: AppKind
    resource: str
    description: str


APP_CATALOG: tuple[AppSpec, ...] = (
    AppSpec(
        DETECTION_PIPELINE,
        "vision",
        PIPELINE_TARGETS[DETECTION_PIPELINE],
        "Object detection (COCO classes) with track ids.",
    ),
    AppSpec(
        POSE_ESTIMATION_PIPELINE,
        "vision",
        PIPELINE_TARGETS[POSE_ESTIMATION_PIPELINE],
        "Person detection from a pose-estimation model.",
    ),
    AppSpec(
        VLM_CHAT_APP,
        "genai",
        VLM_CHAT_APP,
        "On-device vision-language model behind describe_scene.",
    ),
)


def supported_archs(app: str) -> list[str]:
    """Return the architectures that have a model for a catalogue app.

    Args:
        app: Catalogue app name.

    Returns:
        Sorted architecture names; empty for an unknown app.
    """
    spec = next((s for s in APP_CATALOG if s.name == app), None)
    return get_supported_architectures(spec.resource) if spec else []


def list_apps(arch: str | None) -> list[dict[str, Any]]:
    """List the catalogue apps that have models for an architecture.

    Args:
        arch: Hailo-Apps architecture name, or None when no device is attached, in which case
            every app is listed with the architectures it supports.

    Returns:
        One dict per app with its name, kind, description and whether ``start_pipeline`` can run
        it (``startable``), plus the default model for ``arch``, or the supported architectures
        when ``arch`` is None.
    """
    apps: list[dict[str, Any]] = []
    for spec in APP_CATALOG:
        supported = get_supported_architectures(spec.resource)
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
            entry["default_model"] = get_default_model_name(spec.resource, arch)
        apps.append(entry)
    return apps
