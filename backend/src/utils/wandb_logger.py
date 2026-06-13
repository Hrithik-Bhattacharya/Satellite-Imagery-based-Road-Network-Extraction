"""
wandb_logger.py – Member 4 (Validation & MLOps)
Weights & Biases integration with automatic local fallback.

Behaviour:
  - If WANDB_API_KEY is set in the environment AND wandb is installed:
    logs metrics and artifacts to W&B.
  - Otherwise silently falls back to local JSON logging (outputs/wandb_local.json).
  - Never crashes regardless of W&B availability.

Usage:
    logger = WandbLogger(project="rural-roads", run_name="exp-001")
    logger.log_metric("iou", 0.74, step=10)
    logger.log_metrics({"dice": 0.81, "apls": 0.66}, step=10)
    logger.log_artifact("outputs/model.onnx", artifact_type="model")
    logger.finish()
"""

from __future__ import annotations

import os
import json
import time
import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# W&B availability check
# ---------------------------------------------------------------------------

def _wandb_available() -> bool:
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not api_key:
        return False
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# WandbLogger
# ---------------------------------------------------------------------------

class WandbLogger:
    """
    Unified logger for Weights & Biases with graceful local fallback.

    Args:
        project:    W&B project name.
        run_name:   Human-readable run identifier.
        config:     dict of hyperparameters / experiment config to log.
        output_dir: Local directory for fallback JSON log.
        tags:       List of tags for the W&B run.
    """

    def __init__(
        self,
        project: str = "rural-road-extraction",
        run_name: str | None = None,
        config: dict | None = None,
        output_dir: str = "outputs",
        tags: list | None = None,
    ):
        self._use_wandb = _wandb_available()
        self._run_name = run_name or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._local_log_path = self._output_dir / "wandb_local.json"
        self._local_records: list[dict] = []
        self._step = 0

        if self._use_wandb:
            try:
                import wandb
                self._run = wandb.init(
                    project=project,
                    name=self._run_name,
                    config=config or {},
                    tags=tags or [],
                    reinit=True,
                )
                print(f"[WandbLogger] Connected to W&B project='{project}' run='{self._run_name}'")
            except Exception as e:
                print(f"[WandbLogger] W&B init failed ({e}), falling back to local logging.")
                self._use_wandb = False
                self._run = None
        else:
            self._run = None
            print(f"[WandbLogger] W&B not available. Logging locally to {self._local_log_path}")

    # -----------------------------------------------------------------------
    # Metric logging
    # -----------------------------------------------------------------------

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a single scalar metric."""
        self.log_metrics({name: value}, step=step)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """
        Log a dict of scalar metrics.

        Args:
            metrics: {metric_name: value}
            step:    global training step; auto-incremented if None.
        """
        if step is None:
            self._step += 1
            step = self._step

        record = {"step": step, "timestamp": time.time(), **metrics}

        if self._use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"[WandbLogger] Failed to log metrics to W&B: {e}")

        # Always write locally as backup
        self._local_records.append(record)
        self._flush_local()

    def log_artifact(
        self,
        file_path: str,
        artifact_type: str = "model",
        name: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Log a file artifact (model checkpoint, CSV, etc.) to W&B.
        Falls back to printing the path if W&B is unavailable.
        """
        file_path = str(file_path)
        name = name or Path(file_path).stem

        if self._use_wandb:
            try:
                import wandb
                artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
                artifact.add_file(file_path)
                self._run.log_artifact(artifact)
                print(f"[WandbLogger] Artifact logged: {file_path}")
            except Exception as e:
                print(f"[WandbLogger] Failed to log artifact ({e}). Path: {file_path}")
        else:
            print(f"[WandbLogger] Artifact (local only): {file_path}")
            self._local_records.append({
                "artifact": file_path,
                "type": artifact_type,
                "timestamp": time.time(),
            })
            self._flush_local()

    def log_config(self, config: dict) -> None:
        """Update the run configuration."""
        if self._use_wandb and self._run is not None:
            try:
                import wandb
                wandb.config.update(config)
            except Exception:
                pass
        self._local_records.append({"config": config, "timestamp": time.time()})
        self._flush_local()

    def log_summary(self, summary: dict) -> None:
        """Log final summary values (e.g. best epoch metrics)."""
        if self._use_wandb and self._run is not None:
            try:
                import wandb
                for k, v in summary.items():
                    wandb.run.summary[k] = v
            except Exception:
                pass
        self._local_records.append({"summary": summary, "timestamp": time.time()})
        self._flush_local()

    def finish(self) -> None:
        """Close the W&B run and flush local logs."""
        if self._use_wandb and self._run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        self._flush_local()
        print(f"[WandbLogger] Run finished. Local log: {self._local_log_path}")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _flush_local(self) -> None:
        try:
            with open(self._local_log_path, "w") as f:
                json.dump(self._local_records, f, indent=2, default=str)
        except Exception:
            pass

    @property
    def active(self) -> bool:
        """True if connected to W&B, False if in local-only mode."""
        return self._use_wandb
