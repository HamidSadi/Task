"""Lightweight experiment logging inspired by MLflow/Weights & Biases."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator
import json
import shutil


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


@dataclass
class RunState:
    params: Dict[str, object] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    started_at: str = field(default_factory=_timestamp)
    ended_at: str | None = None

    def to_json(self) -> str:
        payload = {
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"


class MLFlowLikeLogger:
    """Persist experiment metadata to disk without external services."""

    def __init__(self, experiment_name: str = "default", base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path("reports/experiments")
        self.experiment_dir = self.base_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir: Path | None = None
        self._state: RunState | None = None

    @contextmanager
    def run(self, run_name: str | None = None) -> Iterator["MLFlowLikeLogger"]:
        self.start_run(run_name)
        try:
            yield self
        finally:
            self.end_run()

    def start_run(self, run_name: str | None = None) -> None:
        if self._run_dir is not None:
            raise RuntimeError("A run is already active; end it before starting another.")
        run_id = run_name or _timestamp()
        self._run_dir = self.experiment_dir / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._state = RunState()
        self._write_state()

    def end_run(self) -> None:
        if self._state is None or self._run_dir is None:
            return
        self._state.ended_at = _timestamp()
        self._write_state()
        self._run_dir = None
        self._state = None

    def log_params(self, params: Dict[str, object]) -> None:
        if self._state is None:
            raise RuntimeError("No active run to log parameters.")
        self._state.params.update(params)
        self._write_state()

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        if self._state is None:
            raise RuntimeError("No active run to log metrics.")
        self._state.metrics.update(metrics)
        self._write_state()

    def set_tags(self, tags: Dict[str, str]) -> None:
        if self._state is None:
            raise RuntimeError("No active run to set tags.")
        self._state.tags.update(tags)
        self._write_state()

    def log_text(self, text: str, filename: str) -> Path:
        if self._run_dir is None:
            raise RuntimeError("No active run to log text.")
        path = self._run_dir / filename
        path.write_text(text, encoding="utf-8")
        return path

    def log_artifact(self, artifact_path: Path, destination: str | None = None) -> Path:
        if self._run_dir is None:
            raise RuntimeError("No active run to log artifacts.")
        target_dir = self._run_dir
        if destination:
            target_dir = self._run_dir / destination
            target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / artifact_path.name
        shutil.copyfile(artifact_path, target_path)
        return target_path

    def _write_state(self) -> None:
        if self._run_dir is None or self._state is None:
            return
        state_path = self._run_dir / "run.json"
        state_path.write_text(self._state.to_json(), encoding="utf-8")


__all__ = ["MLFlowLikeLogger"]

