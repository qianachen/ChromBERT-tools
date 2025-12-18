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
import torchmetrics as tm
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from types import SimpleNamespace
from chrombert import ChromBERTFTConfig, DatasetConfig
from chrombert.scripts.chrombert_make_dataset import process
from lightning.pytorch.callbacks import TQDMProgressBar
from sklearn.metrics.pairwise import cosine_similarity

from .utils import resolve_paths, check_files
from .utils import bw_getSignal_bins, split_data
from .utils import cal_metrics_regression, model_eval
from .utils import model_embedding, factor_rank

def make_dataset(peak, bw, d_odir, files_dict):
    # 1.prepare_dataset
    total_peak_process = (
        process(peak, files_dict["chrombert_region_file"], mode="region")[["chrom", "start", "end", "build_region_index"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    total_peak_process.to_csv(f"{d_odir}/chrombert_region_overlap_peak.csv", index=False)

    total_region_processed_sampled = (
        total_peak_process.sample(n=min(20000, len(total_peak_process)), random_state=55).reset_index(drop=True)
    )
    total_region_processed_sampled.to_csv(f"{d_odir}/total_region_processed_sampled.csv", index=False)

    # 2. scan signal
    signal = bw_getSignal_bins(bw=bw, regions=total_peak_process)
    total_region_signal_processed = pd.concat([total_peak_process, signal], axis=1)
    base_ca_signal_array = np.load(files_dict["base_ca_signal"])  # baseline chrom accessibility
    total_region_signal_processed["baseline"] = base_ca_signal_array[
        total_region_signal_processed["build_region_index"].values
    ]

    # 3. log2 signal and fold change
    total_region_signal_processed["log2_signal"] = np.log2(total_region_signal_processed["signal"] + 1)
    total_region_signal_processed["log2_baseline"] = np.log2(total_region_signal_processed["baseline"] + 1)
    total_region_signal_processed["label"] = (
        total_region_signal_processed["log2_signal"] - total_region_signal_processed["log2_baseline"]
    )
    total_region_signal_processed = total_region_signal_processed[
        ["chrom", "start", "end", "build_region_index", "label", "log2_signal", "log2_baseline", "signal", "baseline"]
    ]

    total_region_signal_processed.to_csv(f"{d_odir}/total_region_signal_processed.csv", index=False)
    total_region_signal_processed_sampled = total_region_signal_processed[
        total_region_signal_processed.build_region_index.isin(total_region_processed_sampled.build_region_index)
    ]
    total_region_signal_processed_sampled.to_csv(f"{d_odir}/total_region_signal_processed_sampled.csv", index=False)

    # 4. train/test/valid split
    split_data(total_region_signal_processed, "", d_odir)
    split_data(total_region_signal_processed_sampled, "_sampled", d_odir)

    # 5.up region and nochange region
    up_region = (
        total_region_signal_processed[total_region_signal_processed["label"] > 1]
        .sort_values("label", ascending=False)
        .head(1000)
        .reset_index(drop=True)
    )

    total_region_signal_processed["abs_label"] = np.abs(total_region_signal_processed["label"])
    nochange_region = (
        total_region_signal_processed.query("signal > 0 or baseline > 0")
        .query("label <1 and label > -1")
        .sort_values("abs_label")
        .reset_index(drop=True)
        .iloc[0:1000]
    )

    up_region.to_csv(f"{d_odir}/up_region.csv", index=False)
    nochange_region.to_csv(f"{d_odir}/nochange_region.csv", index=False)


def init_datamodule(d_odir, args, files_dict):
    # 1. init dataconfig
    data_config = DatasetConfig(
        kind="GeneralDataset",
        supervised_file=None,
        hdf5_file=files_dict["hdf5_file"],
        batch_size=args.batch_size,
        num_workers=8,
    )
    # 2. init datamodule
    if args.mode == "fast":
        data_module = chrombert.LitChromBERTFTDataModule(
            config=data_config,
            train_params={"supervised_file": f"{d_odir}/train_sampled.csv"},
            val_params={"supervised_file": f"{d_odir}/valid_sampled.csv"},
            test_params={"supervised_file": f"{d_odir}/test_sampled.csv"},
        )
    else:
        data_module = chrombert.LitChromBERTFTDataModule(
            config=data_config,
            train_params={"supervised_file": f"{d_odir}/train.csv"},
            val_params={"supervised_file": f"{d_odir}/valid.csv"},
            test_params={"supervised_file": f"{d_odir}/test.csv"},
        )
    data_module.setup()
    return data_config, data_module


def model_train(d_odir, train_odir, args, files_dict):
    data_config, data_module = init_datamodule(d_odir, args, files_dict)

    # 3. init chrombert
    model_config = ChromBERTFTConfig(
        genome=args.genome,
        task="general",
        pretrain_ckpt=files_dict["pretrain_ckpt"],
        mtx_mask=files_dict["mtx_mask"],
    )
    model = model_config.init_model()
    model.freeze_pretrain(2)  # freeze chrombert 6 transformer blocks during fine-tuning

    # 4. init trainer
    train_config = chrombert.finetune.TrainConfig(
        kind="regression",
        loss="rmse",
        max_epochs=5,
        accumulate_grad_batches=8,
        val_check_interval=0.2,
        limit_val_batches=0.5,
        tag="cell_specific",
    )
    train_module = train_config.init_pl_module(model)
    callback_ckpt = pl.callbacks.ModelCheckpoint(monitor=f"{train_config.tag}_validation/pcc", mode="max")
    early_stop = pl.callbacks.EarlyStopping(
        monitor=f"{train_config.tag}_validation/pcc",
        mode="max",
        patience=5,
        min_delta=0.01,
        verbose=True,
    )

    refresh_rate = max(1, int(len(data_module.train_dataloader()) / 10))

    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        log_every_n_steps=1,
        limit_val_batches=train_config.limit_val_batches,
        val_check_interval=train_config.val_check_interval,
        accelerator="gpu",
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        fast_dev_run=False,
        precision="bf16-mixed",
        strategy="auto",
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
            callback_ckpt,
            early_stop,
            TQDMProgressBar(refresh_rate=refresh_rate),
        ],
        logger=pl.loggers.TensorBoardLogger(f"./{train_odir}/lightning_logs", name="cell_specific"),
    )
    trainer.fit(train_module, data_module)
    return data_module, model_config



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


def plot_regulator_subnetwork(G, target_reg, odir, k_hop=1, threshold=None, quantile=None):
    if target_reg not in G:
        print(f"[WARN] {target_reg} not found in graph (degree == 0)")
        return

    subG = nx.ego_graph(G, target_reg, radius=k_hop)
    print(f"Subnetwork for {target_reg}:")
    print("  nodes:", subG.number_of_nodes())
    print("  edges:", subG.number_of_edges())

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(subG, seed=42)

    node_colors = []
    node_sizes = []
    for n in subG.nodes():
        if n == target_reg:
            node_colors.append("red")
            node_sizes.append(500)
        else:
            node_colors.append("lightgray")
            node_sizes.append(500)

    edges = subG.edges(data=True)
    weights = [d.get("weight", 1.0) for (_, _, d) in edges]
    edge_widths = [1 + 3 * (w - min(weights)) / (max(weights) - min(weights) + 1e-8) for w in weights]

    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(subG, pos, width=edge_widths, alpha=0.7)
    nx.draw_networkx_labels(subG, pos, font_size=10)

    plt.axis("off")
    plt.title(f"Subnetwork of {target_reg} (k:{k_hop}, threshold:{threshold:.3f}, quantile:{quantile:.3f})")
    plt.tight_layout()
    plt.savefig(f"{odir}/subnetwork_{target_reg}_k{k_hop}.pdf")


def plot_trn(embs, regulators, focus_regulator, odir, quantile=0.99):
    cos_sim = cosine_similarity(embs)
    cos_sim_df = pd.DataFrame(cos_sim, index=regulators, columns=regulators)
    cos_sim_df.to_csv(f"{odir}/regulator_cosine_similarity.tsv", sep="\t", index=True)
    N = embs.shape[0]
    i_upper = np.triu_indices(N, k=1)
    threshold = np.quantile(cos_sim[i_upper], quantile)

    G = nx.Graph()
    edge_rows = []
    for i in range(N):
        for j in range(i + 1, N):
            w = cos_sim[i, j]
            if w >= threshold:
                n1 = regulators[i]
                n2 = regulators[j]
                G.add_edge(n1, n2, weight=w)
                edge_rows.append((n1, n2, w))

    df_edges = pd.DataFrame(edge_rows, columns=["node1", "node2", "cosine_similarity"])
    df_edges.to_csv(
        f"{odir}/total_graph_edge_threshold{threshold:.2f}_quantile{quantile:.2f}.tsv",
        sep="\t",
        index=False,
    )

    print("Number of nodes of total graph:", G.number_of_nodes())
    print(f"Number of edges of total graph (threshold={threshold:.3f}):", G.number_of_edges())
    if focus_regulator is not None:
        for reg in focus_regulator:
            plot_regulator_subnetwork(G, reg, odir, k_hop=1, threshold=threshold, quantile=quantile)

def run(args):
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    files_dict = resolve_paths(args)
    check_files(files_dict)

    d_odir = f"{odir}/dataset";  os.makedirs(d_odir, exist_ok=True)
    train_odir = f"{odir}/train"; os.makedirs(train_odir, exist_ok=True)
    results_odir = f"{odir}/results"; os.makedirs(results_odir, exist_ok=True)
    emb_odir = f"{odir}/emb"; os.makedirs(emb_odir, exist_ok=True)

    # 1) prepare dataset
    print("Stage 1: Praparing the dataset")
    make_dataset(args.cell_type_peak, args.cell_type_bw, d_odir, files_dict)
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
        data_module, model_config = model_train(d_odir, train_odir, args, files_dict)
        data_config = data_module.basic_config
        print("Finished stage 2: the important stage, Congratudate you get a cell-specific chrombert")
        print("Evaluating the finetuned model performance")
        model_tuned = model_eval(args, train_odir, data_module, model_config, cal_metrics_regression)

    # 3) embedding
    print("Stage 3: generate regulator embedding on different activity regions")
    model_emb = model_embedding(train_odir, model_config, ft_ckpt=args.ft_ckpt, model_tuned=model_tuned)
    up_emb = generate_emb(data_config, model_emb, f"{d_odir}/up_region.csv", emb_odir, "up")
    nochange_emb = generate_emb(data_config, model_emb, f"{d_odir}/nochange_region.csv", emb_odir, "nochange")
    print("Finished stage 3")

    # 4) key regulator
    print("Stage 4: find key regulator")
    cos_sim_df = factor_rank(up_emb, nochange_emb, model_emb.list_regulator, results_odir)
    print("Finished stage 4")


    # 5) TRN
    print("Stage 5: plot TRN")
    focus_regulator = cos_sim_df.head(n=25).factors.tolist()
    plot_trn(up_emb, model_emb.list_regulator, focus_regulator, results_odir, quantile=args.quantile)
    print("Finished stage 5")

    print("Finished all stages!")
    print(f"You get a cell-specific chrombert in {train_odir} or your provided {args.ft_ckpt}")
    print(f"You can get the most important regulators for this cell type in {results_odir}/factor_importance_rank.csv")
    print(f"You can get the trn edge list for this cell type in {results_odir}/total_graph_edge_threshold*_quantile*.tsv")
    print(f"You can get the subnetwork for each top25 regulator in {results_odir}/subnetwork_*.pdf")


@click.command(name="infer_cell_trn", context_settings={"help_option_names": ["-h", "--help"]})
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
              default=os.path.expanduser("~/.cache/chrombert/data"),
              show_default=True, type=click.Path(file_okay=False),
              help="ChromBERT cache directory (contains config/ anno/ checkpoint/ etc).")
@click.option("--quantile", default=0.99, show_default=True, type=float,
              help="Quantile threshold for cosine similarity.")
@click.option("--k-hop", "k_hop", default=1, show_default=True, type=int,
              help="k-hop for subnetwork (currently subnetwork plot uses k_hop=1 as in file1).")

def infer_cell_trn(cell_type_bw, cell_type_peak, ft_ckpt, odir, mode, batch_size,
                   chrombert_cache_dir, quantile, k_hop, genome, resolution):
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
        quantile=quantile,
        k_hop=k_hop,
    )
    run(args)


if __name__ == "__main__":
    infer_cell_trn()
