import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class RunManifest:
    manifest_version: str
    stage: str
    model_name: str
    dataset: Dict[str, Any]
    tracker: Dict[str, Any]
    splitter: Dict[str, Any]
    preprocessing: Dict[str, Any]
    trainer: Dict[str, Any]
    evaluation: Dict[str, Any]
    promotion: Dict[str, Any]
    raw: Dict[str, Any]


def _require_section(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in payload or not isinstance(payload[key], dict):
        raise ValueError(f"Manifest section '{key}' must exist and be an object")
    return payload[key]


def _validate_component(section_name: str, section: Dict[str, Any]) -> None:
    if "component" not in section or "version" not in section:
        raise ValueError(
            f"Manifest section '{section_name}' must define both 'component' and 'version'"
        )
    section.setdefault("config", {})
    if not isinstance(section["config"], dict):
        raise ValueError(f"Manifest section '{section_name}.config' must be an object")


def load_run_manifest(manifest_path: Path) -> RunManifest:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    required = [
        "dataset",
        "tracker",
        "splitter",
        "preprocessing",
        "trainer",
        "evaluation",
        "promotion",
    ]

    sections = {name: _require_section(payload, name) for name in required}
    for section_name, section in sections.items():
        _validate_component(section_name, section)

    manifest = RunManifest(
        manifest_version=str(payload.get("manifest_version", "1.0")),
        stage=str(payload.get("stage", "prod")),
        model_name=str(payload.get("model_name", "HousingModel")),
        dataset=sections["dataset"],
        tracker=sections["tracker"],
        splitter=sections["splitter"],
        preprocessing=sections["preprocessing"],
        trainer=sections["trainer"],
        evaluation=sections["evaluation"],
        promotion=sections["promotion"],
        raw=payload,
    )

    if manifest.stage not in {"dev", "prod"}:
        raise ValueError("Manifest 'stage' must be either 'dev' or 'prod'")

    return manifest
