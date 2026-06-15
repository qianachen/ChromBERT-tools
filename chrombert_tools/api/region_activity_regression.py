"""
Python API for region accessibility regression (``region_activity_regression`` CLI).
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Optional

from ..cli.region_activity_regression import run as _cli_run
from ..cli.prediction_run_result import ChrombertPredictionRunResult


def region_activity_regression(
    acc_peak1: Optional[str] = None,
    acc_peak2: Optional[str] = None,
    acc_signal1: Optional[str] = None,
    acc_signal2: Optional[str] = None,
    predict_file: Optional[str] = None,
    direction: str = "2-1",
    odir: str = "./output",
    genome: str = "hg38",
    resolution: str = "1kb",
    mode: str = "fast",
    lite: bool = False,
    ft_ckpt: Optional[str] = None,
    tss_flank: int = 10000,
    include_tss_background: bool = False,
    subtract_reference_baseline: bool = True,
    chrombert_cache_dir: Optional[str] = None,
    batch_size: int = 4,
    train_chr: Optional[str] = None,
    valid_chr: Optional[str] = None,
    test_chr: Optional[str] = None,
    chrombert_region_file: Optional[str] = None,
    chrombert_region_emb_file: Optional[str] = None,
    hdf5_file: Optional[str] = None,
    pretrain_ckpt: Optional[str] = None,
    mtx_mask: Optional[str] = None,
    meta_file: Optional[str] = None,
) -> ChrombertPredictionRunResult:
    """
    Same pipeline as ``chrombert-tools region_activity_regression``.

    Args:
        train_chr, valid_chr, test_chr:
            Optional ``;``-separated chromosome lists for the dataset split (same semantics
            as the CLI: omit all three for a random 80/10/10 split; with train and
            valid, test is implicit unless ``test_chr`` is set). See
            :func:`chrombert_tools.cli.utils.resolve_chrom_split_sets`.
        subtract_reference_baseline:
            Single-state only; same meaning as the CLI flag ``--subtract-reference-baseline``.
            Default ``True``: label uses log2(1+state-1) minus log2(1+reference baseline) per bin.
            Set ``False`` for the raw log2(1+state-1) label only. On the command line, passing
            ``--subtract-reference-baseline`` selects the raw log2 label (Click inverts the default).

    Returns:
        :class:`~chrombert_tools.prediction_run_result.ChrombertPredictionRunResult`
        with ``model``, ``predictions_path``, ``model_input_path``, etc.
    """
    if chrombert_cache_dir is None:
        chrombert_cache_dir = os.path.expanduser("~/.cache/chrombert/data")

    args = SimpleNamespace(
        acc_peak1=acc_peak1,
        acc_peak2=acc_peak2,
        acc_signal1=acc_signal1,
        acc_signal2=acc_signal2,
        predict_file=predict_file,
        direction=direction,
        odir=odir,
        genome=genome.lower() if isinstance(genome, str) else genome,
        resolution=resolution,
        mode=str(mode).lower(),
        lite=lite,
        ft_ckpt=ft_ckpt,
        tss_flank=tss_flank,
        include_tss_background=include_tss_background,
        subtract_background_signal=subtract_reference_baseline,
        chrombert_cache_dir=chrombert_cache_dir,
        batch_size=batch_size,
        train_chr=train_chr,
        valid_chr=valid_chr,
        test_chr=test_chr,
    )
    if chrombert_region_file is not None:
        args.chrombert_region_file = chrombert_region_file
    if chrombert_region_emb_file is not None:
        args.chrombert_region_emb_file = chrombert_region_emb_file
    if hdf5_file is not None:
        args.hdf5_file = hdf5_file
    if pretrain_ckpt is not None:
        args.pretrain_ckpt = pretrain_ckpt
    if mtx_mask is not None:
        args.mtx_mask = mtx_mask
    if meta_file is not None:
        args.meta_file = meta_file

    return _cli_run(args)
