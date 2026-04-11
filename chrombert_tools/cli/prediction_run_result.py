"""
Structured return type for classification / regression pipelines that produce
``predictions.csv`` and optionally retain the in-memory model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass
class ChrombertPredictionRunResult:
    """
    Attributes:
        model:
            Fine-tuned (or loaded) PyTorch module on CUDA, in ``eval()``-ready state.
        model_config:
            Path to saved model config JSON, or config object (tool-specific).
        data_config:
            Path to saved dataset config JSON, or ``DatasetConfig`` (tool-specific).
        predictions_df:
            Prediction table (same as ``predictions.csv``).
        model_ckpt:
            Fine-tuned checkpoint path, or ``None``.
        train_output_dir:
            Training output directory (e.g. ``{odir}/train``), or ``None`` in predict-only mode.
    """

    # Required fields first; optional fields with defaults must follow (dataclass rule).
    model: Any
    model_config: Any
    data_config: Any
    predictions_df: pd.DataFrame
    model_ckpt: Optional[str] = None
    train_output_dir: Optional[str] = None
