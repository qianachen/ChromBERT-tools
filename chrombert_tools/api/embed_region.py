"""
API interface for region embedding
Provides a Python-friendly interface reusing the CLI implementation.
"""
from types import SimpleNamespace
from typing import Optional
import numpy as np
import pandas as pd
from ..cli.embed_region import run_gene_general, run_region_general, run_gene_cell, run_region_cell


def embed_region(
    region: str,
    gene: Optional[str] = None,
    odir: str = "./output",
    oname: str = "emb",
    genome: str = "hg38",
    resolution: str = "1kb",
    chrombert_cache_dir: Optional[str] = None,
    chrombert_region_file: Optional[str] = None,
    chrombert_region_emb_file: Optional[str] = None,
    chrombert_gene_meta: Optional[str] = None,
    ft_ckpt: Optional[str] = None,
    return_embeddings: bool = True,
) -> Optional[np.ndarray]:

    import os
    
    # Set default cache dir if not provided
    if chrombert_cache_dir is None:
        chrombert_cache_dir = os.path.expanduser("~/.cache/chrombert/data")
    
    # Create args namespace (same as CLI)
    args_region = SimpleNamespace(
        region=region,
        odir=odir,
        oname=oname,
        genome=genome,
        resolution=resolution,
        chrombert_cache_dir=chrombert_cache_dir,
        chrombert_region_file=chrombert_region_file,
        chrombert_region_emb_file=chrombert_region_emb_file,
    )
    
    
    args_gene = SimpleNamespace(
        gene=gene,
        odir=odir,
        oname=oname,
        genome=genome,
        resolution=resolution,
        chrombert_cache_dir=chrombert_cache_dir,
        chrombert_region_file=chrombert_region_file,
        chrombert_region_emb_file=chrombert_region_emb_file,
        chrombert_gene_meta=chrombert_gene_meta,
    )
    
    args_region_cell = SimpleNamespace(
        region=region,
        odir=odir,
        oname=oname,
        genome=genome,
        resolution=resolution,
        chrombert_cache_dir=chrombert_cache_dir,
        chrombert_region_file=chrombert_region_file,
        chrombert_region_emb_file=chrombert_region_emb_file,
        ft_ckpt=ft_ckpt,
    )
    args_gene_cell = SimpleNamespace(
        gene=gene,
        odir=odir,
        oname=oname,
        genome=genome,
        resolution=resolution,
        chrombert_cache_dir=chrombert_cache_dir,
        chrombert_region_file=chrombert_region_file,
        chrombert_region_emb_file=chrombert_region_emb_file,    
        chrombert_gene_meta=chrombert_gene_meta,
        ft_ckpt=ft_ckpt,
    )
    # Run the core logic (reuse CLI implementation)
    gene_emb_dict = None
    region_emb = None
    region_bed = None
    
    if gene is not None:
        gene_emb_dict = run_gene_general(args_gene, return_data=True)
    if region is not None:
        region_emb, region_bed = run_region_general(args_region, return_data=True)
    if gene is not None and ft_ckpt is not None:
        gene_emb_dict = run_gene_cell(args_gene_cell, return_data=True)
    if region is not None and ft_ckpt is not None:
        region_emb, region_bed = run_region_cell(args_region_cell, return_data=True)
        
    return region_emb, region_bed, gene_emb_dict