"""
Region-region interaction interpretation API.

Thin wrapper around ``chrombert_tools.cli.interpret_region_region_interactions.run``,
matching the ``chrombert-tools interpret_region_region_interactions`` CLI.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Optional

import pandas as pd

from ..cli.interpret_region_region_interactions import run as _cli_run


def interpret_region_region_interactions(
    region: str,
    odir: str = "./output",
    genome: str = "hg38",
    resolution: str = "1kb",
    lite: bool = False,
    chrombert_cache_dir: Optional[str] = None,
    chrombert_region_file: Optional[str] = None,
    chrombert_region_emb_file: Optional[str] = None,
    return_data: bool = True,
    region2: Optional[str] = None,
    filter_gene_name: Optional[str] = None,
    filter_gene_id: Optional[str] = None,
    distance_min: int = 0,
    distance_max: int = 250_000,
    distance_window: Optional[int] = None,
    batch_size: int = 4,
    ft_ckpt: Optional[str] = None,
    ignore_regulator: Optional[str] = None,
    gep: bool = False,
    flank_window: int = 4,
    chrombert_gene_meta: Optional[str] = None,
    model_config: Optional[str] = None,
    data_config: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Compute cosine similarities between region embeddings: either enhancer-promoter
    style pairs (single ``region`` BED vs gene TSS from metadata) or all pairs
    between two BEDs on the same chromosome within
    ``[distance_min, distance_max]`` (absolute genomic distance in bp).

    Args:
        region:
            Path to BED file (set 1, e.g. candidate enhancers). Required.
        odir:
            Output directory; writes TSV results and embedding intermediates.
        genome:
            ``hg38`` or ``mm10`` (case-insensitive).
        resolution:
            ChromBERT bin size: ``1kb``, ``200bp``, ``2kb``, or ``4kb``.
        chrombert_cache_dir:
            ChromBERT data root; ``None`` → ``~/.cache/chrombert/data``.
        chrombert_region_file:
            Override for ChromBERT reference region BED.
        chrombert_region_emb_file:
            Override for precomputed region embedding ``.npy``.
        return_data:
            If ``True``, return the result DataFrame; if ``False``, only write files
            and return ``None``.
        region2:
            Optional second BED. If ``None``: TSS–region mode (requires gene meta in
            cache). If set: pairwise cosines between the two region sets (same chrom,
            within ``[distance_min, distance_max]``); gene filters are ignored.
        filter_gene_name:
            TSS/EP mode only (``region2 is None``). Semicolon-separated ``gene_name``
            symbols; only those genes' TSS are kept, and the input ``region`` BED is
            reduced to the same chromosome set as the filtered TSS. Ignored in two-BED
            mode.
        filter_gene_id:
            TSS/EP mode only. Semicolon-separated ``gene_id`` values. Combined with
            ``filter_gene_name`` with OR: a TSS row is kept if its name or id matches
            the respective list. Region1 (``region``) is also limited to those
            chromosomes, like ``filter_gene_name``.
        distance_min:
            Minimum absolute genomic separation (bp, ``>=0``) for kept pairs. Pairs
            whose unsigned interval gap (or ``|TSS-distal|`` in EP mode) is below this
            value are dropped. Direction (upstream vs downstream) is ignored.
        distance_max:
            Maximum absolute genomic separation (bp, ``>=0``) for kept pairs. Pairs
            beyond this value, or on different chromosomes, are dropped.
        distance_window:
            Deprecated alias of ``distance_max`` (with ``distance_min=0``). When set,
            it overrides ``distance_max``/``distance_min``.
        batch_size:
            Batch size when computing embeddings from the model.
        ft_ckpt:
            Optional fine-tuned checkpoint passed through to the interpret config.
        ignore_regulator:
            Optional; passed when building the interpret model (regulator mask).
        gep:
            If ``True``, use the GEP (multi-flank-window) path.
        flank_window:
            Flank window size when ``gep`` is ``True``.
        chrombert_gene_meta:
            Optional override TSV for gene/TSS metadata (TSS mode only); ``None``
            uses the cache file from ``resolve_paths``.

    Returns:
        DataFrame of similarity pairs (columns depend on mode). ``None`` if
        ``return_data`` is ``False``.
    """
    if chrombert_cache_dir is None:
        chrombert_cache_dir = os.path.expanduser("~/.cache/chrombert/data")

    args = SimpleNamespace(
        region=region,
        region2=region2,
        filter_gene_name=filter_gene_name,
        filter_gene_id=filter_gene_id,
        odir=odir,
        genome=genome.lower() if isinstance(genome, str) else genome,
        resolution=resolution,
        chrombert_cache_dir=chrombert_cache_dir,
        chrombert_region_file=chrombert_region_file,
        chrombert_region_emb_file=chrombert_region_emb_file,
        batch_size=batch_size,
        ft_ckpt=ft_ckpt,
        ignore_regulator=ignore_regulator,
        gep=gep,
        flank_window=flank_window,
        distance_min=distance_min,
        distance_max=distance_max,
        distance_window=distance_window,
        lite=lite,
        model_config=model_config,
        data_config=data_config,
    )
    if chrombert_gene_meta is not None:
        args.chrombert_gene_meta = chrombert_gene_meta

    return _cli_run(args, return_data=return_data)
