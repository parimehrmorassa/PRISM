"""
Experiment logging utilities with optional W&B support.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import LOGGING_CONFIG

logger = logging.getLogger(__name__)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_str: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
):
    """
    Configure root logger with console (and optional file) handler.

    Args:
        level: Logging level (e.g., logging.INFO)
        log_file: Optional path to write logs to a file
        format_str: Log format string
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=format_str, handlers=handlers)


class ExperimentLogger:
    """
    Lightweight experiment logger that optionally integrates with Weights & Biases.

    Usage:
        logger = ExperimentLogger("nbfnet_fb15k237")
        logger.log({"loss": 0.4, "mrr": 0.32}, step=100)
        logger.log_summary({"best_mrr": 0.35})
        logger.finish()
    """

    def __init__(
        self,
        run_name: str,
        config: Optional[Dict] = None,
        tags: Optional[list] = None,
        use_wandb: Optional[bool] = None,
    ):
        self.run_name = run_name
        self.config = config or {}
        self.history: list = []
        self.summary: Dict = {}
        self._wandb = None

        use_wandb = use_wandb if use_wandb is not None else LOGGING_CONFIG["use_wandb"]
        if use_wandb:
            self._init_wandb(tags)

        self._log = logging.getLogger(f"exp.{run_name}")
        self._log.info(f"Experiment '{run_name}' started at {datetime.now().isoformat()}")

    def _init_wandb(self, tags: Optional[list]):
        try:
            import wandb
            self._wandb = wandb.init(
                project=LOGGING_CONFIG["project_name"],
                name=self.run_name,
                config=self.config,
                tags=tags or [],
            )
        except Exception as e:
            self._log.warning(f"W&B init failed: {e}. Continuing without W&B.")
            self._wandb = None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log a dict of metrics at a given step."""
        entry = {"step": step, **metrics}
        self.history.append(entry)
        self._log.info(
            f"Step {step}: "
            + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in metrics.items())
        )
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def log_summary(self, summary: Dict[str, Any]):
        """Log final summary metrics."""
        self.summary.update(summary)
        self._log.info(
            "Summary: "
            + ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in summary.items()
            )
        )
        if self._wandb is not None:
            for k, v in summary.items():
                self._wandb.summary[k] = v

    def save_history(self, path: str):
        """Save training history to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"run_name": self.run_name, "history": self.history,
                       "summary": self.summary}, f, indent=2)
        self._log.info(f"Training history saved to {path}")

    def finish(self):
        """Finalize the run."""
        if self._wandb is not None:
            self._wandb.finish()
        self._log.info(f"Experiment '{self.run_name}' finished.")


def format_metrics_table(metrics: Dict[str, Any], title: str = "") -> str:
    """
    Format a dict of metrics as a readable table.

    Args:
        metrics: Dict of metric names → values
        title: Optional header

    Returns:
        Formatted string
    """
    lines = []
    if title:
        lines.append(f"\n{'='*50}")
        lines.append(f"  {title}")
        lines.append(f"{'='*50}")

    max_key_len = max(len(k) for k in metrics) if metrics else 10
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:<{max_key_len}}  {v:.4f}")
        else:
            lines.append(f"  {k:<{max_key_len}}  {v}")

    lines.append(f"{'='*50}")
    return "\n".join(lines)
