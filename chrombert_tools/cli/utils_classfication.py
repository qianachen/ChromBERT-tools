import os
import pickle
import shutil
import itertools
from collections import defaultdict

import click
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import networkx as nx
import nxviz as nv
from nxviz import annotate
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from chrombert import ChromBERTFTConfig, DatasetConfig

from .utils import resolve_paths, check_files, overlap_regulator_func, overlap_region
from .utils import split_data, split_data_by_chrom, parse_chrom_list_string, cal_metrics_binary, factor_rank
from .utils_train_cell import retry_train



# =========================
# validation
# =========================

def validate_args(args):
    """Validate and normalize arguments."""
    n = len(args.function_beds)
    # if n < 2:
    #     raise ValueError("At least 2 --function-bed groups are required.")

    if len(args.function_modes) == 0:
        args.function_modes = ["and"] * n
    elif len(args.function_modes) == 1:
        args.function_modes = list(args.function_modes) * n
    elif len(args.function_modes) != n:
        raise ValueError(
            f"--function-mode count ({len(args.function_modes)}) must be 1 or "
            f"match --function-bed count ({n})."
        )
    else:
        args.function_modes = list(args.function_modes)

    if len(args.function_names) == 0:
        args.function_names = [f"function_{i}" for i in range(n)]
    elif len(args.function_names) != n:
        raise ValueError(
            f"--function-name count ({len(args.function_names)}) must match "
            f"--function-bed count ({n})."
        )
    else:
        args.function_names = list(args.function_names)

    args.function_modes = [m.lower() for m in args.function_modes]

    train_chr = getattr(args, "train_chr", None)
    valid_chr = getattr(args, "valid_chr", None)
    if train_chr is None and valid_chr is None:
        args.train_chrom_set = None
        args.valid_chrom_set = None
    else:
        if train_chr is None or valid_chr is None:
            raise ValueError(
                "--train-chr and --valid-chr must be provided together, or omit both."
            )
        tr_set = parse_chrom_list_string(train_chr, args.genome)
        va_set = parse_chrom_list_string(valid_chr, args.genome)
        inter = tr_set & va_set
        if inter:
            raise ValueError(
                f"--train-chr and --valid-chr must not overlap: {sorted(inter)}"
            )
        args.train_chrom_set = tr_set
        args.valid_chrom_set = va_set


# =========================
# dataset preparation
# =========================

def merge_regions_by_mode(dfs, mode, name):
    """Merge multiple region DataFrames using 'and' (intersection) or 'or' (union)."""
    if mode not in {"and", "or"}:
        raise ValueError(f"{name}: mode must be 'and' or 'or', got '{mode}'")
    if not dfs:
        raise ValueError(f"{name}: no overlapping regions found.")

    if mode == "or":
        out = pd.concat(dfs, ignore_index=True)
        return out.drop_duplicates(subset=["build_region_index"]).reset_index(drop=True)

    keep = set(dfs[0]["build_region_index"])
    for df in dfs[1:]:
        keep &= set(df["build_region_index"])
    out = dfs[0][dfs[0]["build_region_index"].isin(keep)].copy()
    return out.drop_duplicates(subset=["build_region_index"]).reset_index(drop=True)


def prepare_dataset(args, files_dict, d_odir):
    """Build multi-class dataset from N function BED groups (labels 0..N-1).
    
    When only 1 function BED is provided, a background (negative) class is
    automatically created from all ChromBERT reference regions that are NOT
    in the provided positive set.
    """
    ref_regions = files_dict["chrombert_region_file"]

    all_regions = []
    for idx, (bed_group, mode, name) in enumerate(
        zip(args.function_beds, args.function_modes, args.function_names)
    ):
        bed_files = [x.strip() for x in bed_group.split(";") if x.strip()]
        dfs = []
        for bf in bed_files:
            bf_basename = os.path.basename(bf)
            tag = f"{name} | {bf_basename}" if len(bed_files) > 1 else name
            df = overlap_region(bf, ref_regions, d_odir, tag=tag)
            dfs.append(df)
        merged = merge_regions_by_mode(dfs, mode, name)
        merged["label"] = idx
        all_regions.append(merged)
        print(f"  {name} (class {idx}): {len(merged)} regions")

    # Single function BED → build negative class from remaining ref regions
    if len(args.function_beds) == 1:
        positive_indices = set(all_regions[0]["build_region_index"])
        ref_df = pd.read_csv(
            ref_regions, sep="\t", header=None,
            names=["chrom", "start", "end", "build_region_index"],
        )
        neg_df = ref_df[~ref_df["build_region_index"].isin(positive_indices)].copy()
        neg_df["label"] = 1
        all_regions.append(neg_df)
        args.function_names.append("background")
        print(f"  background (class 1): {len(neg_df)} regions "
              f"(ref_regions - {args.function_names[0]})")

    n_classes = len(args.function_names)

    # Deduplicate overlapping regions; earlier classes take priority
    seen = set()
    deduped = []
    for idx, df in enumerate(all_regions):
        clean = df[~df["build_region_index"].isin(seen)].reset_index(drop=True)
        removed = len(df) - len(clean)
        if removed > 0:
            print(f"  {args.function_names[idx]}: removed {removed} overlapping regions")
        seen.update(clean["build_region_index"])
        deduped.append(clean)

    combined = pd.concat(deduped, ignore_index=True)
    combined.to_csv(os.path.join(d_odir, "total.csv"), index=False)

    for idx, name in enumerate(args.function_names):
        print(f"  {name}: {(combined['label'] == idx).sum()} (final)")
    print(f"  Total: {len(combined)}")

    use_chrom_split = args.train_chrom_set is not None

    if args.mode == "fast":
        cap = int(getattr(args, "fast_max_total", 20000))
        if cap < 1:
            raise ValueError("--fast-max-total must be >= 1")
        max_per_class = max(1, cap // n_classes)
        print(
            f"  Fast mode: up to {max_per_class} regions per class "
            f"(total budget ≈ {cap})"
        )
        sampled = (
            combined.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(n=min(max_per_class, len(g)), random_state=55))
            .reset_index(drop=True)
        )
        sampled.to_csv(os.path.join(d_odir, "total_sampled.csv"), index=False)
        if use_chrom_split:
            print(
                "  Fast mode + chromosome split: train/valid/test by --train-chr / "
                "--valid-chr (test = remaining chromosomes)"
            )
            split_data_by_chrom(
                sampled,
                "_sampled",
                d_odir,
                args.genome,
                args.train_chrom_set,
                args.valid_chrom_set,
            )
        else:
            split_data(sampled, "_sampled", d_odir)
    else:
        if use_chrom_split:
            print(
                "  Full mode: train/valid/test by chromosome "
                "(--train-chr / --valid-chr; test = remaining chromosomes)"
            )
            split_data_by_chrom(
                combined,
                "",
                d_odir,
                args.genome,
                args.train_chrom_set,
                args.valid_chrom_set,
            )
        else:
            split_data(combined, "", d_odir)
        args.mode = "full"



    return combined