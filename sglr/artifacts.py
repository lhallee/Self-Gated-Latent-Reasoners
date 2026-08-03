"""Stable run directories, manifests, JSON, JSONL, and PyTorch checkpoints."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from sglr.config import ExperimentConfig


COMPLETION_FILENAME = "run_complete.json"


def run_directory(output_root: str | Path, experiment_name: str, variant: str, seed: int) -> Path:
    """Return the deterministic directory for one experiment variant and seed."""

    return Path(output_root) / experiment_name / variant / f"seed_{seed}"


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_is_complete(path: str | Path) -> bool:
    directory = Path(path)
    required_files = (
        COMPLETION_FILENAME,
        "evaluation_summary.json",
        "evaluation.jsonl",
        "evaluation_images.npz",
        "manifest.json",
        "best_model.pt",
        "last_state.pt",
        "training_history.json",
    )
    if not all((directory / filename).is_file() for filename in required_files):
        return False
    try:
        completion = load_json(directory / COMPLETION_FILENAME)
        manifest = load_json(directory / "manifest.json")
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return False
    return (
        completion.get("summary") == "evaluation_summary.json"
        and completion.get("schema_version") == manifest.get("schema_version")
        and completion.get("variant") == manifest.get("variant")
        and completion.get("seed") == manifest.get("seed")
    )


def validate_run_config(path: str | Path, experiment: ExperimentConfig) -> None:
    """Reject reuse of a deterministic run path for a different resolved configuration."""

    manifest = load_json(Path(path) / "manifest.json")
    expected = json.loads(json.dumps(asdict(experiment)))
    if manifest.get("config") != expected:
        raise ValueError(
            f"Existing run has a different configuration: {path}. "
            "Choose a new experiment_name or output_root."
        )


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Expected JSON file at {input_path}")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {input_path}")
    return payload


def save_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")))
            handle.write("\n")


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), output_path)


def load_checkpoint(path: str | Path, device: torch.device | str) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Expected checkpoint at {input_path}")
    payload = torch.load(input_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a checkpoint mapping at {input_path}")
    return payload


def build_run_manifest(
    experiment: ExperimentConfig,
    variant: str,
    seed: int,
    device: torch.device,
    run_path: str | Path,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Capture enough provenance to reproduce or audit one run."""

    repository = _find_repository(Path(run_path).resolve())
    return {
        "schema_version": experiment.schema_version,
        "variant": variant,
        "seed": seed,
        "device": str(device),
        "run_directory": str(Path(run_path).resolve()),
        "command": list(sys.argv if command is None else command),
        "config": asdict(experiment),
        "git": _git_state(repository),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(("torch", "torchvision", "numpy", "matplotlib")),
        },
    }


def _find_repository(start: Path) -> Path:
    candidates = (start, *start.parents, Path(__file__).resolve().parent.parent)
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return Path(__file__).resolve().parent.parent


def _package_versions(package_names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _git_state(repository: Path) -> dict[str, str | bool | None]:
    command_prefix = [
        "git",
        "-c",
        f"safe.directory={repository.as_posix()}",
        "-C",
        str(repository),
    ]
    try:
        commit = subprocess.run(
            [*command_prefix, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [*command_prefix, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": True}
    return {"commit": commit, "dirty": bool(status.strip())}
