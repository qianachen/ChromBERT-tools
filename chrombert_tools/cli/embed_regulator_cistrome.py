import os
import re
import click
from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import chrombert
from chrombert import ChromBERTFTConfig, DatasetConfig
from chrombert.scripts.utils import HDF5Manager
import pickle
from .utils import (
    resolve_paths,
    check_files,
    check_region_file,
    chrom_to_int_series,
    overlap_regulator_func,
    overlap_cistrome_func,
)


def _normalize_query_arg(query, arg_name):
    """
    Support both direct query string and a file path input.
    File format can be one-id-per-line, or separated by ';' / ',' / tab.
    """
    if not query:
        return None

    if os.path.isfile(query):
        items = []
        with open(query, "r") as f:
            for i, line in enumerate(f):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = [p.strip() for p in re.split(r"[;,\t]", s) if p.strip()]
                if not parts:
                    continue
                # Skip a simple header line like "regulator" / "cistrome".
                if i == 0 and len(parts) == 1 and parts[0].lower() in {
                    "regulator",
                    "cistrome",
                    "factor",
                    "name",
                    "id",
                }:
                    continue
                items.extend(parts)
        if len(items) == 0:
            raise ValueError(f"--{arg_name} file is empty or has no valid entries: {query}")
        return ";".join(items)

    return query


def _safe_leaf(raw) -> str:
    """Flat HDF5 path segment for emb/{leaf} (no '/' hierarchy in names)."""
    leaf = str(raw).replace("\\", "_").replace("/", "_")
    leaf = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf)
    return leaf.strip("._-") or "empty"


def _flat_emb_h5_paths_for_dict(keys):
    """
    Same layout as embed_regulator.py / embed_cistrome.py: one dataset per emb/{name}.
    Sanitize '/' in ids; uniquify within this output file only (regulator and cistrome are separate HDF5s).
    """
    used = set()
    out = {}

    def take_leaf(raw):
        leaf = _safe_leaf(raw)
        base = leaf
        dup_i = 2
        while base in used:
            base = f"{leaf}__dup{dup_i}"
            dup_i += 1
        used.add(base)
        return base

    for k in keys:
        out[k] = f"emb/{take_leaf(k)}"
    return out


def run(args, return_data=False):
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    # mm10 resolution limit (only have 1kb data)
    if args.genome.lower() == "mm10" and args.resolution != "1kb":
        raise ValueError("mm10 currently only supports 1kb in this cache layout (adjust if you have more).")

    files_dict = resolve_paths(args)
    args.regulator = _normalize_query_arg(args.regulator, "regulator")
    args.cistrome = _normalize_query_arg(args.cistrome, "cistrome")

    if not args.regulator and not args.cistrome:
        raise ValueError("At least one of --regulator or --cistrome must be provided.")

    # Match embed_regulator.py / embed_cistrome.py: only check what each mode needs, plus shared inputs.
    if args.regulator and args.cistrome:
        check_files(files_dict, required_keys=[
            "chrombert_region_file",
            "chrombert_regulator_file",
            "meta_file",
            "hdf5_file",
            "pretrain_ckpt",
            "mtx_mask",
        ])
    elif args.regulator:
        # Same as embed_regulator.py (ChromBERTFTConfig still expects mtx_mask on disk via resolve_paths).
        check_files(files_dict, required_keys=[
            "chrombert_region_file",
            "chrombert_regulator_file",
            "hdf5_file",
            "pretrain_ckpt",
        ])
    else:
        check_files(files_dict, required_keys=[
            "chrombert_region_file",
            "meta_file",
            "hdf5_file",
            "pretrain_ckpt",
            "mtx_mask",
        ])

    focus_region = args.region
    overlap_bed = check_region_file(focus_region, files_dict, odir)

    # chromosome mapping to integer
    first_chrom = str(overlap_bed["chrom"].iloc[0])
    if "chr" in first_chrom.lower():
        overlap_bed["chrom"] = chrom_to_int_series(overlap_bed["chrom"].astype(str), args.genome)
    overlap_bed = overlap_bed.dropna(subset=["chrom"]).copy()
    overlap_bed["chrom"] = overlap_bed["chrom"].astype(int)
    overlap_bed.to_csv(f"{odir}/model_input.tsv", sep="\t", index=False)

    regulator_idx_dict = {}
    cistrome_gsmid_dict = {}
    if args.regulator:
        _, _, regulator_idx_dict = overlap_regulator_func(
            args.regulator, files_dict["chrombert_regulator_file"]
        )
        if len(regulator_idx_dict) == 0:
            raise ValueError("No requested regulators matched ChromBERT regulator list. Nothing to embed.")
    if args.cistrome:
        _, _, cistrome_gsmid_dict = overlap_cistrome_func(args.cistrome, files_dict["meta_file"])
        if len(cistrome_gsmid_dict) == 0:
            raise ValueError("No requested cistromes matched ChromBERT meta. Nothing to embed.")
    if len(regulator_idx_dict) == 0 and len(cistrome_gsmid_dict) == 0:
        raise ValueError("Nothing to embed after matching --regulator/--cistrome.")

    reg_h5_path = _flat_emb_h5_paths_for_dict(regulator_idx_dict) if regulator_idx_dict else {}
    cis_h5_path = _flat_emb_h5_paths_for_dict(cistrome_gsmid_dict) if cistrome_gsmid_dict else {}

    # dataloader
    data_config = DatasetConfig(
        kind="GeneralDataset",
        supervised_file=f"{odir}/model_input.tsv",
        hdf5_file=files_dict["hdf5_file"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dl = data_config.init_dataloader()
    ds = data_config.init_dataset()

    # model
    model_config = ChromBERTFTConfig(
        genome=args.genome,
        dropout=0,
        task="general",
        pretrain_ckpt=files_dict["pretrain_ckpt"],
        mtx_mask=files_dict["mtx_mask"],
    )
    model = model_config.init_model().get_embedding_manager().cuda().bfloat16()

    # Two HDF5 files: regulator-only and cistrome-only (same emb/<id> layout as embed_regulator / embed_cistrome).
    reg_shapes = {reg_h5_path[k]: [(len(ds), 768), np.float16] for k in regulator_idx_dict}
    cis_shapes = {cis_h5_path[k]: [(len(ds), 768), np.float16] for k in cistrome_gsmid_dict}
    reg_h5_file = f"{odir}/{args.oname}_regulator_region_aware.hdf5"
    cis_h5_file = f"{odir}/{args.oname}_cistrome_region_aware.hdf5"
    total_counts = 0
    # save mean embeddings
    reg_sums = {name: np.zeros(768, dtype=np.float64) for name in regulator_idx_dict}
    cis_sums = {name: np.zeros(768, dtype=np.float64) for name in cistrome_gsmid_dict}
    
    reg_emb_dict = {}
    cis_emb_dict = {}
    with ExitStack() as stack:
        h5_reg = (
            stack.enter_context(
                HDF5Manager(reg_h5_file, region=[(len(ds), 4), np.int64], **reg_shapes)
            )
            if regulator_idx_dict
            else None
        )
        h5_cis = (
            stack.enter_context(
                HDF5Manager(cis_h5_file, region=[(len(ds), 4), np.int64], **cis_shapes)
            )
            if cistrome_gsmid_dict
            else None
        )
        with torch.no_grad():
            for batch in tqdm(dl, total=len(dl)):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()

                model(batch)  # init cache inside embedding_manager

                bs = batch["region"].shape[0]
                start_idx = total_counts
                total_counts += bs
                end_idx = total_counts

                batch_index = batch["build_region_index"].long().cpu().numpy().reshape(-1)
                region = overlap_bed.iloc[start_idx:end_idx].values
                assert (batch_index == region[:, -1].reshape(-1)).all(), "Batch index and region index do not match"

                reg_embs = {
                    reg_h5_path[k]: model.get_regulator_embedding(k).float().cpu().numpy()
                    for k in regulator_idx_dict
                }
                cis_embs = {
                    cis_h5_path[k]: model.get_cistrome_embedding(v).float().cpu().numpy()
                    for k, v in cistrome_gsmid_dict.items()
                }
                if h5_reg is not None:
                    h5_reg.insert(region=region, **reg_embs)
                if h5_cis is not None:
                    h5_cis.insert(region=region, **cis_embs)
                
                # Store for return if needed
                if return_data:
                    for rk in regulator_idx_dict:
                        v = reg_embs[reg_h5_path[rk]]
                        if rk not in reg_emb_dict:
                            reg_emb_dict[rk] = []
                        reg_emb_dict[rk].append(v)
                    for ck in cistrome_gsmid_dict:
                        v = cis_embs[cis_h5_path[ck]]
                        if ck not in cis_emb_dict:
                            cis_emb_dict[ck] = []
                        cis_emb_dict[ck].append(v)
                
                for reg_name in regulator_idx_dict:
                    emb = model.get_regulator_embedding(reg_name)
                    emb_np = emb.float().cpu().numpy()
                    reg_sums[reg_name] += emb_np.sum(axis=0)
                for cis_name, cis_idx in cistrome_gsmid_dict.items():
                    emb = model.get_cistrome_embedding(cis_idx)
                    emb_np = emb.float().cpu().numpy()
                    cis_sums[cis_name] += emb_np.sum(axis=0)
                    
    # Concatenate collected data if return_data
    if return_data:
        for k in reg_emb_dict:
            reg_emb_dict[k] = np.concatenate(reg_emb_dict[k], axis=0)
        for k in cis_emb_dict:
            cis_emb_dict[k] = np.concatenate(cis_emb_dict[k], axis=0)
            
    reg_means = {
        reg_name: (sum_vec / total_counts)
        for reg_name, sum_vec in reg_sums.items()
    }
    cis_means = {
        cis_name: (sum_vec / total_counts)
        for cis_name, sum_vec in cis_sums.items()
    }
    # Separate mean pickles (match flat mean dict style; add h5 path map when names are sanitized).
    reg_mean_pkl = os.path.join(odir, f"{args.oname}_regulator_mean.pkl")
    cis_mean_pkl = os.path.join(odir, f"{args.oname}_cistrome_mean.pkl")
    if regulator_idx_dict:
        with open(reg_mean_pkl, "wb") as f:
            pickle.dump({"means": reg_means, "h5_dataset_paths": dict(reg_h5_path)}, f)
    if cistrome_gsmid_dict:
        with open(cis_mean_pkl, "wb") as f:
            pickle.dump({"means": cis_means, "h5_dataset_paths": dict(cis_h5_path)}, f)

    print("Finished!")
    if regulator_idx_dict:
        print("Saved mean regulator embeddings to pickle file:", reg_mean_pkl)
        print("Saved region-aware regulator embeddings to hdf5 file:", reg_h5_file)
    if cistrome_gsmid_dict:
        print("Saved mean cistrome embeddings to pickle file:", cis_mean_pkl)
        print("Saved region-aware cistrome embeddings to hdf5 file:", cis_h5_file)

    means_out = {
        "regulator": reg_means if regulator_idx_dict else {},
        "cistrome": cis_means if cistrome_gsmid_dict else {},
        "regulator_mean_pkl": reg_mean_pkl if regulator_idx_dict else None,
        "cistrome_mean_pkl": cis_mean_pkl if cistrome_gsmid_dict else None,
        "regulator_h5": reg_h5_file if regulator_idx_dict else None,
        "cistrome_h5": cis_h5_file if cistrome_gsmid_dict else None,
        "h5_dataset_paths": {"regulator": dict(reg_h5_path), "cistrome": dict(cis_h5_path)},
    }
    if return_data:
        return means_out, {"regulator": reg_emb_dict, "cistrome": cis_emb_dict}, overlap_bed

@click.command(name="embed_regulator_cistrome", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--region", "region",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, help="Region file.")
@click.option("--regulator", required=False,
              help="Regulators of interest OR a file path. Example: EZH2 or EZH2;BRD4, or a txt/csv/tsv file.")
@click.option("--cistrome", required=False,
              help="Cistrome ids OR a file path. Example: ENCSR...;GSM...;ATAC-seq:HEK293T, or a txt/csv/tsv file.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False), help="Output directory.")
@click.option("--oname", default="emb", show_default=True,
              type=str, 
              help="Output basename: writes <oname>_regulator_region_aware.hdf5 and <oname>_cistrome_region_aware.hdf5 (and mean pkls) when each mode is used.")
@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False), help="Genome.")
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["1kb", "200bp", "2kb", "4kb"], case_sensitive=False), help="Resolution.")
@click.option("--batch-size", default=64, show_default=True, type=int, help="Batch size.")
@click.option("--num-workers", default=8, show_default=True, type=int, help="Dataloader workers.")

@click.option("--chrombert-cache-dir", "chrombert_cache_dir",
              default="~/.cache/chrombert/data",
              show_default=True, type=click.Path(file_okay=False),
              help="ChromBERT cache dir (contains config/ checkpoint/ etc).")

def embed_regulator_cistrome(region, regulator, cistrome, odir, oname, genome, resolution, batch_size, num_workers,
        chrombert_cache_dir):
    '''
    Extract general regulator and cistrome embeddings on specified regions
    '''
    args = SimpleNamespace(
        region=region,
        regulator=regulator,
        cistrome=cistrome,
        odir=odir,
        oname=oname,
        genome=genome.lower(),
        resolution=resolution,
        batch_size=batch_size,
        num_workers=num_workers,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    run(args)


if __name__ == "__main__":
    embed_regulator_cistrome()
