import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=False)

from config import load_run_manifest
from services import run_training_pipeline


def resolve_project_root() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return project_root


def resolve_manifest_path(project_root: Path) -> Path:
    manifest_raw = os.getenv("CD_RUN_MANIFEST_PATH", "training_and_promotion/config/manifests/default_v1.json")
    manifest_path = Path(manifest_raw)
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    return manifest_path


@dataclass(frozen=True)
class RuntimeConfig:
    tracking_uri: str
    stage: Literal["dev", "prod"]
    model_name: str


def resolve_runtime_config(*, manifest_stage: str, manifest_model_name: str) -> RuntimeConfig:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI must be set")

    stage_raw = os.getenv("MLFLOW_STAGE", manifest_stage)
    if stage_raw not in {"dev", "prod"}:
        raise ValueError("MLFLOW_STAGE must be either 'dev' or 'prod'")
    stage = cast(Literal["dev", "prod"], stage_raw)

    model_name = os.getenv("MODEL_NAME", manifest_model_name)
    return RuntimeConfig(tracking_uri=tracking_uri, stage=stage, model_name=model_name)


def main():
    project_root = resolve_project_root()
    manifest_path = resolve_manifest_path(project_root)
    manifest = load_run_manifest(manifest_path)
    runtime = resolve_runtime_config(
        manifest_stage=manifest.stage,
        manifest_model_name=manifest.model_name,
    )

    rmse, model_version, promoted = run_training_pipeline(
        manifest=manifest,
        project_root=project_root,
        manifest_path=manifest_path,
        tracking_uri=runtime.tracking_uri,
        stage=runtime.stage,
        model_name=runtime.model_name,
    )

    print(
        f"Pipeline finished. test_rmse={rmse:.4f}, "
        f"model_version={model_version}, promoted={promoted}"
    )



if __name__ == "__main__":
    main()