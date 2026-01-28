import os
import glob
import json
import pickle
import click
from tqdm import tqdm
import chrombert
import networkx as nx
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from types import SimpleNamespace
from chrombert import ChromBERTFTConfig, DatasetConfig
from sklearn.metrics.pairwise import cosine_similarity
from .utils import set_seed
from .utils import resolve_paths, check_files
from .utils import cal_metrics_regression, model_eval
from .utils import model_embedding, factor_rank
from .utils_train_cell import make_dataset, retry_train


def generate_emb(data_config, model_emb, sup_file, odir, name):
    data_config.supervised_file = sup_file
    dl = data_config.init_dataloader()
    ds = data_config.init_dataset()  # kept as in file1
    regulators = model_emb.list_regulator
    regulator_idx_dict = {regulator: idx for idx, regulator in enumerate(regulators)}

    total_counts = 0
    embs_pool = np.zeros((len(regulators), 768), dtype=np.float64)

    with torch.no_grad():
        for batch in tqdm(dl, total=len(dl)):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            total_counts += batch["region"].shape[0]
            emb = model_emb(batch)  # initialize the cache
            emb_np = emb.float().cpu().numpy()
            embs_pool += emb_np.sum(axis=0)

        embs_pool /= total_counts

    embs_pool_dict = {}
    for reg_name, reg_idx in regulator_idx_dict.items():
        embs_pool_dict[reg_name] = embs_pool[reg_idx]
    out_pkl = os.path.join(odir, f"{name}_region_mean_regulator_embs_dict.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(embs_pool_dict, f)
    return embs_pool

def run(args):
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    files_dict = resolve_paths(args)
    required_keys=[
        "chrombert_region_file",
        "hdf5_file",
        "pretrain_ckpt",
        "mtx_mask"
    ]
    check_files(files_dict, required_keys=required_keys)

    d_odir = f"{odir}/dataset";  os.makedirs(d_odir, exist_ok=True)
    train_odir = f"{odir}/train"; os.makedirs(train_odir, exist_ok=True)
    results_odir = f"{odir}/results"; os.makedirs(results_odir, exist_ok=True)
    emb_odir = f"{odir}/emb"; os.makedirs(emb_odir, exist_ok=True)

    # 1) prepare dataset
    print("Stage 1: Praparing the dataset")
    make_dataset(args.cell_type_peak, args.cell_type_bw, d_odir, files_dict, args.mode)
    print("Finished stage 1")

    # 2) train or load fine-tuned model
    if args.ft_ckpt is not None:
        print(f"Use fine-tuned ChromBERT checkpoint file: {args.ft_ckpt} to infer cell-specific trn")
        model_config = ChromBERTFTConfig(
            genome=args.genome,
            task="general",
            dropout=0,
            pretrain_ckpt=files_dict["pretrain_ckpt"],
            mtx_mask=files_dict["mtx_mask"],
            finetune_ckpt=args.ft_ckpt,
        )
        data_config = DatasetConfig(
            kind="GeneralDataset",
            supervised_file=None,
            hdf5_file=files_dict["hdf5_file"],
            batch_size=args.batch_size,
            num_workers=8,
        )
        model_tuned = model_config.init_model()
        print("Finished stage 2")
    else:
        print("Stage 2: Fine-tuning the model")
        model_tuned, train_odir, model_config, data_config = retry_train(args, files_dict, cal_metrics_regression, metcic='pearsonr', min_threshold=0.4)

    # 3) embedding
    print("Stage 3: generate regulator embedding on different activity regions")
    model_emb = model_embedding(train_odir, model_config, ft_ckpt=args.ft_ckpt, model_tuned=model_tuned)
    up_emb = generate_emb(data_config, model_emb, f"{d_odir}/highly_accessible_region.csv", emb_odir, "up")
    nochange_emb = generate_emb(data_config, model_emb, f"{d_odir}/background_region.csv", emb_odir, "background")
    print("Finished stage 3")

    # 4) key regulator
    print("Stage 4: find key regulator")
    cos_sim_df = factor_rank(up_emb, nochange_emb, model_emb.list_regulator, results_odir)
    print("Finished stage 4: identify cell-specific key regulators (top 25)")
    print(cos_sim_df.head(n=25))

    print("Finished all stages!")
    if args.ft_ckpt is not None:
        print(f"Used fine-tuned ChromBERT checkpoint: {args.ft_ckpt}")
    else:
        print(f"Cell-specific ChromBERT model saved to: {train_odir}")
    print(f"Key regulators for this cell type saved to: {results_odir}/factor_importance_rank.csv")


@click.command(name="infer_cell_key_regulator", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--cell-type-bw", "cell_type_bw",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, help="Cell type accessibility BigWig file.")
@click.option("--cell-type-peak", "cell_type_peak",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, help="Cell type accessibility Peak BED file.")
@click.option("--ft-ckpt", "ft_ckpt",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=False, default=None, show_default=True,
              help="Fine-tuned ChromBERT checkpoint. If provided, skip fine-tuning and use this ckpt.")
@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False),
              help="Reference genome (hg38 or mm10).")
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["200bp", "1kb", "2kb", "4kb"], case_sensitive=False),
              help="ChromBERT resolution.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False), help="Output directory.")
@click.option("--mode", default="fast", show_default=True,
              type=click.Choice(["fast", "full"], case_sensitive=False),
              help="Fast: downsample regions to 20k for training; Full: use all regions.")
@click.option("--batch-size", "batch_size", default=4, show_default=True, type=int,
              help="Batch size.")
@click.option("--chrombert-cache-dir", "chrombert_cache_dir",
              default="~/.cache/chrombert/data",
              show_default=True, type=click.Path(file_okay=False),
              help="ChromBERT cache directory (contains config/ anno/ checkpoint/ etc).")


def infer_cell_key_regulator(cell_type_bw, cell_type_peak, ft_ckpt, odir, mode, batch_size,
                   chrombert_cache_dir, genome, resolution):
    '''
    Infer cell-specific key regulators
    '''
    args = SimpleNamespace(
        cell_type_bw=cell_type_bw,
        cell_type_peak=cell_type_peak,
        ft_ckpt=ft_ckpt,
        genome=genome,
        resolution=resolution,
        odir=odir,
        mode=mode,
        batch_size=batch_size,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    run(args)


if __name__ == "__main__":
    infer_cell_key_regulator()
