"""
API interface for region embedding
Provides a Python-friendly interface reusing the CLI implementation.
"""
from types import SimpleNamespace
from typing import Optional
import numpy as np
import pandas as pd
from ..cli.infer_ep import run as _cli_run


def infer_ep(
    region: str,
    odir: str = "./output",
    genome: str = "hg38",
    resolution: str = "1kb",
    chrombert_cache_dir: Optional[str] = None,
    chrombert_region_file: Optional[str] = None,
    chrombert_region_emb_file: Optional[str] = None,
    return_data: bool = True,
) -> Optional[pd.DataFrame]:

    import os
    
    # Set default cache dir if not provided
    if chrombert_cache_dir is None:
        chrombert_cache_dir = os.path.expanduser("~/.cache/chrombert/data")
    
    # Create args namespace (same as CLI)
    args = SimpleNamespace(
        region=region,
        odir=odir,
        genome=genome,
        resolution=resolution,
        chrombert_cache_dir=chrombert_cache_dir,
        chrombert_region_file=chrombert_region_file,
        chrombert_region_emb_file=chrombert_region_emb_file,
    )
    
    # Run the core logic (reuse CLI implementation)
    pairs_cos = _cli_run(args,return_data=True)
    
    return pairs_cos
    


