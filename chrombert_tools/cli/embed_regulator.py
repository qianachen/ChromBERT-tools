import os
import re
import click
from types import SimpleNamespace
import subprocess as sp

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import chrombert
from chrombert import ChromBERTFTConfig, DatasetConfig
from chrombert.scripts.utils import HDF5Manager
import pickle

def _nd_from_genome(genome: str) -> str:
    genome = genome.lower()
    if genome == "hg38":
        return "6k"
    elif genome == "mm10":
        return "5k"
    raise ValueError(f"Genome {genome} not supported!")


def resolve_paths(args):
    """
    Resolve all required ChromBERT files based on:
      - genome: hg38/mm10
      - resolution: 1kb/200bp/2kb/4kb
      - optional overrides: --chrombert-xxx-file
    """
    n_d = _nd_from_genome(args.genome)

    # region bed
    chrombert_region_file = os.path.join(args.chrombert_cache_dir, f"config/{args.genome}_{n_d}_{args.resolution}_region.bed")

    # regulator list
    chrombert_regulator_file = os.path.join(args.chrombert_cache_dir, f"config/{args.genome}_{n_d}_regulators_list.txt")

    # hdf5
    hdf5_file = os.path.join(args.chrombert_cache_dir, f"{args.genome}_{n_d}_{args.resolution}.hdf5")
    

    # ckpt
    pretrain_ckpt = os.path.join(args.chrombert_cache_dir, "checkpoint", f"{args.genome}_{n_d}_{args.resolution}_pretrain.ckpt")
    
    # mask matrix
    mtx_mask = os.path.join(args.chrombert_cache_dir, "config", f"{args.genome}_{n_d}_mask_matrix.tsv")

    return {
        "chrombert_region_file": chrombert_region_file,
        "chrombert_regulator_file": chrombert_regulator_file,
        "hdf5_file": hdf5_file,
        "pretrain_ckpt": pretrain_ckpt,
        "mtx_mask": mtx_mask,
    }


def check_files(files_dict):
    missing = [f"{k}: {v}" for k, v in files_dict.items() if not os.path.exists(v)]
    if missing:
        msg = (
            "ChromBERT required file(s) not found:\n  - "
            + "\n  - ".join(missing)
            + "\nHint: run `chrombert_prepare_env` or pass the missing path(s) explicitly."
        )
        raise FileNotFoundError(msg)


def chrom_to_int_series(chrom_series: pd.Series, genome: str) -> pd.Series:
    """
    hg38: 1-22, X=23, Y=24
    mm10: 1-19, X=20, Y=21
    """
    genome = genome.lower()
    if genome == "hg38":
        x_id, y_id, max_auto = 23, 24, 22
    elif genome == "mm10":
        x_id, y_id, max_auto = 20, 21, 19
    else:
        raise ValueError(f"Genome {genome} not supported for chrom mapping")

    def _map_one(c):
        if pd.isna(c):
            return np.nan
        c = str(c).strip()
        c = c[3:] if c.lower().startswith("chr") else c
        if c.upper() == "X":
            return x_id
        if c.upper() == "Y":
            return y_id
        # numbers
        m = re.fullmatch(r"\d+", c)
        if m:
            v = int(c)
            return v if 1 <= v <= max_auto else np.nan
        return np.nan

    return chrom_series.map(_map_one)


def overlap_region(region_bed: str, chrombert_region_file: str, odir: str):
    """
    out:
      - overlap_focus.bed (chrom, start, end, build_region_index)
      - model_input.tsv   (for DatasetConfig)
      - no_overlap_focus.bed
    """
    os.makedirs(odir, exist_ok=True)

    cmd_overlap = f"""
    cut -f 1-3 {region_bed} \
    | sort -k1,1 -k2,2n \
    | bedtools intersect -F 0.5 -wa -wb -a {chrombert_region_file} -b - \
    | awk 'BEGIN{{OFS="\\t"}}{{print $5,$6,$7,$4}}' \
    > {odir}/overlap_focus.bed
    """
    sp.run(cmd_overlap, shell=True, check=True, executable="/bin/bash")

    overlap_bed = pd.read_csv(
        f"{odir}/overlap_focus.bed",
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "build_region_index"],
    )
    overlap_bed.to_csv(f"{odir}/model_input.tsv", sep="\t", index=False)
    overlap_idx = overlap_bed["build_region_index"].to_numpy()

    cmd_no = f"""
    cut -f 1-3 {region_bed} \
    | sort -k1,1 -k2,2n \
    | bedtools intersect -f 0.5 -v -a - -b {chrombert_region_file} \
    > {odir}/no_overlap_focus.bed
    """
    sp.run(cmd_no, shell=True, check=True, executable="/bin/bash")

    total_focus = sum(1 for _ in open(region_bed))
    no_overlap_region_len = sum(1 for _ in open(f"{odir}/no_overlap_focus.bed"))
    print(
        f"Focus region summary - total: {total_focus}, "
        f"overlapping with ChromBERT: {overlap_bed.shape[0]}, "
        f"non-overlapping: {no_overlap_region_len}"
    )
    return overlap_bed, overlap_idx


def overlap_regulator_func(regulator: str, chrombert_regulator_file: str):
    chrombert_regulator = pd.read_csv(
        chrombert_regulator_file,
        sep="\t",
        header=None,
        names=["regulator"],
    )["regulator"].tolist()
    chrombert_regulator = [i.lower() for i in chrombert_regulator]

    focus_regulator_list = [r.strip().lower() for r in regulator.split(";") if r.strip()]
    overlap_regulator = list(set(chrombert_regulator) & set(focus_regulator_list))
    not_overlap_regulator = list(set(focus_regulator_list) - set(chrombert_regulator))
    regulator_dict = {r: chrombert_regulator.index(r) for r in overlap_regulator}

    print("Note: All regulator names were converted to lowercase for matching.")
    print(
        f"Regulator count summary - requested: {len(focus_regulator_list)}, "
        f"matched in ChromBERT: {len(overlap_regulator)}, "
        f"not found: {len(not_overlap_regulator)}"
    )
    return overlap_regulator, not_overlap_regulator, regulator_dict


def run(args):
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    # mm10 resolution limit (only have 1kb data)
    if args.genome.lower() == "mm10" and args.resolution != "1kb":
        raise ValueError("mm10 currently only supports 1kb in this cache layout (adjust if you have more).")

    files_dict = resolve_paths(args)
    check_files(files_dict)

    overlap_bed, _ = overlap_region(args.region_bed, files_dict["chrombert_region_file"], odir)

    # chromosome mapping to integer
    overlap_bed["chrom"] = chrom_to_int_series(overlap_bed["chrom"], args.genome)
    overlap_bed = overlap_bed.dropna(subset=["chrom"]).copy()
    overlap_bed["chrom"] = overlap_bed["chrom"].astype(int)
    overlap_bed.to_csv(f"{odir}/model_input.tsv", sep="\t", index=False)

    _, _, regulator_idx_dict = overlap_regulator_func(args.regulator, files_dict["chrombert_regulator_file"])
    if len(regulator_idx_dict) == 0:
        raise ValueError("No requested regulators matched ChromBERT regulator list. Nothing to embed.")

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

    # save HDF5
    shapes = {f"emb/{k}": [(len(ds), 768), np.float16] for k in regulator_idx_dict}
    total_counts = 0
    # save mean regulator emb
    reg_sums = {name: np.zeros(768, dtype=np.float64) for name in regulator_idx_dict}
    
    with HDF5Manager(f"{odir}/save_regulator_emb_on_specific_region.hdf5",
                     region=[(len(ds), 4), np.int64],
                     **shapes) as h5:
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

                embs = {
                    f"emb/{k}": model.get_regulator_embedding(k).float().cpu().numpy()
                    for k in regulator_idx_dict
                }
                h5.insert(region=region, **embs)
                
                for reg_name, reg_idx in regulator_idx_dict.items():
                    emb = model.get_regulator_embedding(reg_name)
                    emb_np = emb.float().cpu().numpy()            
                    reg_sums[reg_name] += emb_np.sum(axis=0)
    reg_means = {
        reg_name: (sum_vec / total_counts)
        for reg_name, sum_vec in reg_sums.items()
    }
    out_pkl = os.path.join(odir, "mean_regulator_emb.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(reg_means, f)
    print("Finished!")  
    print("Saved mean regulator embeddings to pickle file:", out_pkl)
    print("Saved regulator embeddings to hdf5 file:", f"{odir}/regulator_emb_on_region.hdf5")

@click.command(name="embed_regulator", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--region-bed", "region_bed",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, help="Region BED file.")
@click.option("--regulator", required=True,
              help="Regulators of interest, e.g. EZH2 or EZH2;BRD4. Use ';' to separate multiple regulators.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False), help="Output directory.")
@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False), help="Genome.")
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["1kb", "200bp", "2kb", "4kb"], case_sensitive=False), help="Resolution.")
@click.option("--batch-size", default=64, show_default=True, type=int, help="Batch size.")
@click.option("--num-workers", default=8, show_default=True, type=int, help="Dataloader workers.")

@click.option("--chrombert-cache-dir", "chrombert_cache_dir",
              default=os.path.expanduser("~/.cache/chrombert/data"),
              show_default=True, type=click.Path(file_okay=False),
              help="ChromBERT cache dir (contains config/ checkpoint/ etc).")

def embed_regulator(region_bed, regulator, odir, genome, resolution, batch_size, num_workers,
        chrombert_cache_dir):

    args = SimpleNamespace(
        region_bed=region_bed,
        regulator=regulator,
        odir=odir,
        genome=genome.lower(),
        resolution=resolution,
        batch_size=batch_size,
        num_workers=num_workers,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    run(args)


if __name__ == "__main__":
    embed_regulator()
