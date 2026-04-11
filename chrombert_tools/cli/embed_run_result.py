"""
Structured return types for embedding APIs:

- :class:`ChrombertEmbedRunResult` — :func:`~chrombert_tools.api.embed_region.embed_region`
- :class:`ChrombertEmbedRegulatorRunResult` — :func:`~chrombert_tools.api.embed_regulator.embed_regulator`
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ChrombertEmbedRunResult:
    """
    Attributes:
        region_emb:
            Matrix of region-bin embeddings, shape about ``(n_overlap, 768)``, or
            ``None`` if ``region`` was not requested or ``return_embeddings`` was False.
        overlap_region_bed:
            Overlap table with ``build_region_index`` (and related columns), or ``None``
            if no region run / no in-memory return.
        gene_emb_dict:
            Lowercased gene key → mean-pooled 1D vector, or ``None`` if no gene run /
            no in-memory return.
        ft_ckpt:
            Fine-tuned checkpoint used, or ``None`` if no fine-tuning was performed.
        train_output_dir:
            Output directory for the fine-tuned model, or ``None`` if no fine-tuning was performed.
    """

    region_emb: Optional[np.ndarray]
    overlap_region_bed: Optional[pd.DataFrame]
    gene_emb_dict: Optional[Dict[str, np.ndarray]]
    ft_ckpt: Optional[str]
    train_output_dir: Optional[str]

    def as_tuple(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[Dict[str, np.ndarray]]]:
        """Same order as the historical API: ``(region_emb, overlap_bed, gene_emb_dict)``."""
        return self.region_emb, self.overlap_region_bed, self.gene_emb_dict


@dataclass
class ChrombertEmbedRegulatorRunResult:
    """
    Return type for :func:`~chrombert_tools.api.embed_regulator.embed_regulator`.

    Attributes:
        regulator_means:
            Per-regulator mean embedding across regions (~768-d), or ``None`` if
            ``return_embeddings`` was False.
        regulator_emb_dict:
            Per-regulator stacked embeddings per region, or ``None`` if not returned.
        overlap_region_bed:
            Overlap table used for the forward pass (dataloader row order), or ``None``.
        odir:
            Output directory for this run (absolute).
        oname:
            Basename tag for ``mean_{oname}.pkl`` and ``region_aware_{oname}.hdf5``.
        ft_ckpt:
            Fine-tuned checkpoint used, or ``None`` if no fine-tuning was performed.
        train_output_dir:
            Output directory for the fine-tuned model, or ``None`` if no fine-tuning was performed.
    """

    regulator_means: Optional[Dict[str, np.ndarray]]
    regulator_emb_dict: Optional[Dict[str, np.ndarray]]
    overlap_region_bed: Optional[pd.DataFrame]
    ft_ckpt: Optional[str]
    train_output_dir: Optional[str]

    def as_tuple(
        self,
    ) -> Tuple[
        Optional[Dict[str, np.ndarray]],
        Optional[Dict[str, np.ndarray]],
        Optional[pd.DataFrame],
    ]:
        """Historical order: ``(regulator_means, regulator_emb_dict, overlap_bed)``."""
        return self.regulator_means, self.regulator_emb_dict, self.overlap_region_bed
