"""
Python API for region (functional) classification.

Thin wrapper around ``chrombert_tools.cli.region_function_classification.run`` with
explicit parameters and CLI-aligned defaults so callers need not build
``SimpleNamespace`` by hand.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Optional, Sequence, Union

from ..cli.region_function_classification import run as _cli_run
from ..cli.prediction_run_result import ChrombertPredictionRunResult


def _norm_beds(
    function_beds: Optional[Union[str, Sequence[str]]],
) -> tuple:
    if function_beds is None:
        return ()
    if isinstance(function_beds, str):
        return (function_beds,)
    return tuple(function_beds)


def region_function_classification(
    function_beds: Optional[Union[str, Sequence[str]]] = None,
    function_modes: Optional[Sequence[str]] = None,
    function_names: Optional[Sequence[str]] = None,
    predict_file: Optional[str] = None,
    ignore_regulator: Optional[str] = None,
    odir: str = "./output",
    genome: str = "hg38",
    resolution: str = "1kb",
    ft_ckpt: Optional[str] = None,
    batch_size: int = 4,
    mode: str = "fast",
    lite: bool = False,
    chrombert_cache_dir: Optional[str] = None,
    chrombert_region_file: Optional[str] = None,
    chrombert_region_emb_file: Optional[str] = None,
    chrombert_regulator_file: Optional[str] = None,
    hdf5_file: Optional[str] = None,
    pretrain_ckpt: Optional[str] = None,
    mtx_mask: Optional[str] = None,
    meta_file: Optional[str] = None,
    train_chr: Optional[str] = None,
    valid_chr: Optional[str] = None,
    fast_max_total: int = 20000,
) -> ChrombertPredictionRunResult:
    """
    Train or load a ChromBERT functional classifier and run prediction (same pipeline
    as ``chrombert-tools region_function_classification``).

    Args:
        function_beds:
            One path or sequence of paths per class (same as repeating ``--function-bed``).
            Use ``";"`` inside a string to merge multiple BEDs for one class.
            Omit (or ``None``) only in **predict-only** mode when both ``ft_ckpt`` and
            ``predict_file`` are set.
        function_modes:
            ``"and"`` / ``"or"`` per class, or one value applied to all. ``None`` → all ``"and"``.
        function_names:
            Class names; ``None`` → ``function_0``, ``function_1``, ...
            Required (non-empty) in predict-only mode.
        predict_file:
            Optional regions file for prediction. If omitted, the held-out test split
            from stage 1 is used (non-predict-only). With ``ft_ckpt``, enables predict-only.
        ignore_regulator:
            Regulators to mask; ``";"``-separated names matching ChromBERT list.
        odir:
            Output root (dataset/, train/, predict/ as in CLI).
        genome:
            ``hg38`` or ``mm10`` (lowercased internally).
        resolution:
            ``1kb``, ``200bp``, ``2kb``, or ``4kb``.
        ft_ckpt:
            Fine-tuned checkpoint; skips training when set.
        batch_size:
            DataLoader batch size.
        mode:
            ``fast`` (downsample) or ``full``.
        fast_max_total:
            In ``fast`` mode only: approximate total region budget, divided evenly
            across classes (default ``20000``). Ignored when ``mode`` is ``full``.
        train_chr, valid_chr:
            Optional ``;``-separated chromosome lists for train/validation splits;
            remaining chromosomes are test. Both must be set together or omitted.
            In ``fast`` mode the per-class budget from ``fast_max_total`` still applies,
            then splits follow these chromosomes. Default is random 80/10/10 split.
        chrombert_cache_dir:
            ChromBERT data root; ``None`` → ``~/.cache/chrombert/data``.
        chrombert_region_file, chrombert_region_emb_file, chrombert_regulator_file,
        hdf5_file, pretrain_ckpt, mtx_mask, meta_file:
            Optional path overrides (same semantics as other ChromBERT-tools APIs).

    Returns:
        :class:`~chrombert_tools.prediction_run_result.ChrombertPredictionRunResult`:
        use ``.model``, ``.predictions_path``, ``.model_input_path``,
        ``.source_predict_path``, and ``.predictions_df``.
    """
    if chrombert_cache_dir is None:
        chrombert_cache_dir = os.path.expanduser("~/.cache/chrombert/data")

    beds = _norm_beds(function_beds)
    modes = () if function_modes is None else tuple(function_modes)
    names = () if function_names is None else tuple(function_names)

    args = SimpleNamespace(
        function_beds=beds,
        function_modes=list(modes),
        function_names=list(names),
        predict_file=predict_file,
        ignore_regulator=ignore_regulator,
        odir=odir,
        genome=genome.lower() if isinstance(genome, str) else genome,
        resolution=resolution,
        ft_ckpt=ft_ckpt,
        batch_size=batch_size,
        mode=str(mode).lower(),
        lite=lite,
        chrombert_cache_dir=chrombert_cache_dir,
        train_chr=train_chr,
        valid_chr=valid_chr,
        fast_max_total=fast_max_total,
    )

    if chrombert_region_file is not None:
        args.chrombert_region_file = chrombert_region_file
    if chrombert_region_emb_file is not None:
        args.chrombert_region_emb_file = chrombert_region_emb_file
    if chrombert_regulator_file is not None:
        args.chrombert_regulator_file = chrombert_regulator_file
    if hdf5_file is not None:
        args.hdf5_file = hdf5_file
    if pretrain_ckpt is not None:
        args.pretrain_ckpt = pretrain_ckpt
    if mtx_mask is not None:
        args.mtx_mask = mtx_mask
    if meta_file is not None:
        args.meta_file = meta_file

    return _cli_run(args)
