import os
import itertools
from collections import defaultdict
from typing import Optional

import click
from types import SimpleNamespace

import numpy as np
import pandas as pd
import networkx as nx
import nxviz as nv
from nxviz import annotate
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from .utils import resolve_paths, check_files, overlap_regulator_func, check_region_file
from .utils_interpret import (
    embed_pool_func,
    build_interpret_config,
    load_interpret_model,
)




def get_node_order(G, group_by, sort_by):
    groups = defaultdict(list)
    for node in G.nodes(data=True):
        group_key = node[1].get(group_by)
        sort_value = node[1].get(sort_by)
        groups[group_key].append((node[0], sort_value))

    sorted_nodes = []
    for group in sorted(groups.keys()):
        sorted_nodes.extend(sorted(groups[group], key=lambda x: x[1]))

    node_labels_in_order = [node[0] for node in sorted_nodes]
    return node_labels_in_order


def plot_dual_trn(
    results_odir,
    top_pairs,
    dual_regulator,
    df_results,
    df_cos_func1,
    df_cos_func2,
    thre_func1,
    thre_func2,
    differential_threshold=0.1,
):
    cccol = ["#CE0013", "#16557A"]
    G = nx.Graph()
    graph_regulators = np.concatenate([[dual_regulator], top_pairs])
    for i,j in itertools.combinations(np.arange(len(graph_regulators)), 2):
        if i == 0:
            G.add_nodes_from([(i, {'group':0, 'factor':graph_regulators[i], 'value': 1})])
            if df_results.loc[graph_regulators[j], 'diff'] > differential_threshold:
                G.add_nodes_from([(j, {'group':1, 'factor':graph_regulators[j], 'value':df_results.loc[graph_regulators[j], 'function1']})])
                G.add_edges_from([(i, j, {'color': cccol[0], 'edge_value': df_results.loc[graph_regulators[j], 'function1']})])
            elif df_results.loc[graph_regulators[j], 'diff'] < (-1 * differential_threshold):
                G.add_nodes_from([(j, {'group':2, 'factor':graph_regulators[j], 'value':df_results.loc[graph_regulators[j], 'function2']})])
                G.add_edges_from([(i, j, {'color': cccol[1], 'edge_value': df_results.loc[graph_regulators[j], 'function2']})])
        else:
            # print(graph_regulators[i], graph_regulators[j])
            if df_cos_func1.loc[graph_regulators[i], graph_regulators[j]] > thre_func1 or df_cos_func2.loc[graph_regulators[i], graph_regulators[j]] > thre_func2:
                G.add_edges_from([(i, j, {'color': 'lightgrey', 'edge_value': max(df_cos_func1.loc[graph_regulators[i], graph_regulators[j]], df_cos_func2.loc[graph_regulators[i], graph_regulators[j]])})])
    mapping = {i: str(graph_regulators[i]).upper() for i in range(len(graph_regulators))}
    G = nx.relabel_nodes(G, mapping)
    _ax = nv.circos(
        G,
        group_by="group",
        sort_by="value",
        node_color_by="group",
        edge_alpha_by="edge_value",
    )
    _node_order = get_node_order(G, "group", "value")
    annotate.circos_labels(G, group_by="group", sort_by="value", layout="rotate")
    plt.tight_layout()
    plt.savefig(f"{results_odir}/dual_regulator_{dual_regulator}_subnetwork.pdf")
    print(f"Dual regulator subnetwork plot saved: {results_odir}/dual_regulator_{dual_regulator}_subnetwork.pdf")
    print("Yellow color represents function1 subnetwork; blue color represents function2 subnetwork.")


def infer_driver_factor_trn(
    args,
    dual_regulators,
    embs_pool_region1,
    embs_pool_region2,
    regulators,
    pair_results_subdir: Optional[str] = None,
):
    differential_threshold = getattr(args, "threshold", 0.1)
    emb_odir = f"{args.odir}/emb"
    os.makedirs(emb_odir, exist_ok=True)
    base_results = os.path.join(args.odir, "results")
    if pair_results_subdir:
        results_odir = os.path.join(base_results, pair_results_subdir)
    else:
        results_odir = base_results
    os.makedirs(results_odir, exist_ok=True)
    
    cos_func1 = cosine_similarity(embs_pool_region1)
    cos_func2 = cosine_similarity(embs_pool_region2)
    df_cos_func1 = pd.DataFrame(cos_func1, columns=regulators, index=regulators)
    df_cos_func2 = pd.DataFrame(cos_func2, columns=regulators, index=regulators)

    df_cos_func1.to_csv(
        os.path.join(results_odir, "regulator_cosine_similarity_on_function1_regions.csv")
    )
    df_cos_func2.to_csv(
        os.path.join(results_odir, "regulator_cosine_similarity_on_function2_regions.csv"))

    if dual_regulators is not None:
        thre_func1 = np.percentile(cos_func1.flatten(), 95)
        thre_func2 = np.percentile(cos_func2.flatten(), 95)
        for _idx, dual_regulator in enumerate(dual_regulators):
            df_cos_reg = pd.DataFrame(
                index=regulators,
                data={
                    "function1": df_cos_func1.loc[dual_regulator, :],
                    "function2": df_cos_func2.loc[dual_regulator, :],
                },
            )
            df_cos_reg["diff"] = df_cos_reg["function1"] - df_cos_reg["function2"]
            df_candidate = df_cos_reg[df_cos_reg["diff"].abs() > differential_threshold]
            topN_pos = df_candidate.query("function1 > @thre_func1").index.values
            topN_neg = df_candidate.query("function2 > @thre_func2").index.values
            top_pairs = np.union1d(topN_pos, topN_neg)
            df_cos_reg_save = df_cos_reg.loc[top_pairs, :].reset_index().rename(
                                columns={
                                    "index": "candidate_cofactor",
                                    "function1": "embedding_similarity_region1",
                                    "function2": "embedding_similarity_region2",
                                    "diff": "embedding_similarity_difference"
                                }
                            )
            df_cos_reg_save.to_csv(os.path.join(results_odir, f"dual_regulator_{dual_regulator}_candidate_cofactors.csv"),index=False)
            if len(df_cos_reg_save) > 0:
                plot_dual_trn(
                    results_odir,
                    top_pairs,
                    dual_regulator,
                    df_cos_reg,
                    df_cos_func1,
                    df_cos_func2,
                    thre_func1,
                    thre_func2,
                    differential_threshold=differential_threshold,
                )
            else:
                print(f"No candidate context-dependent cofactors found for dual-regulator {dual_regulator}")

        print("Finished stage 4b: infer dual-functional regulator subnetworks")


def run(args):
    odir = args.odir
    os.makedirs(odir, exist_ok=True)
    files_dict = resolve_paths(args)
    required_keys=[
        "chrombert_region_file",
        "hdf5_file",
        # "pretrain_ckpt",
        # "mtx_mask",
        "meta_file"
    ]
    check_files(files_dict, required_keys=required_keys)

    # 1) make dataset
    d_odir = f"{odir}/dataset";  os.makedirs(d_odir, exist_ok=True)
    d_region1 = f"{d_odir}/region1";  os.makedirs(d_region1, exist_ok=True)
    d_region2 = f"{d_odir}/region2";  os.makedirs(d_region2, exist_ok=True)
    check_region_file(args.region1_file, files_dict, d_region1)
    check_region_file(args.region2_file, files_dict, d_region2)

    region1_file = f"{d_region1}/model_input.tsv"
    region2_file = f"{d_region2}/model_input.tsv"
    # train_odir = f"{odir}/train"; os.makedirs(train_odir, exist_ok=True)
    results_odir = f"{odir}/results"; os.makedirs(results_odir, exist_ok=True)
    emb_odir = f"{odir}/emb"; os.makedirs(emb_odir, exist_ok=True)


    # 2) get dual-regulator
    dual_regulator = args.dual_regulator
    overlap_dual_regulator = None  # avoid undefined
    if dual_regulator is not None:
        overlap_dual_regulator, _, _ = overlap_regulator_func(
            dual_regulator, files_dict["chrombert_regulator_file"]
        )

    # 3) load model (shared with interpret_regulator_interactions)
    data_config, model_config = build_interpret_config(
        args, files_dict, region1_file
    )
    _, model_emb = load_interpret_model(model_config)

    # 4) generate embeddings
    embs_pool_region1, regulators = embed_pool_func(data_config, model_emb, region1_file, emb_odir, "region1")
    embs_pool_region2, regulators = embed_pool_func(data_config, model_emb, region2_file, emb_odir, "region2")



    # 5) infer driver factor in different regions
    infer_driver_factor_trn(
        args, overlap_dual_regulator, embs_pool_region1, embs_pool_region2, regulators
    )
    print("Finished all stages!")
    if args.ft_ckpt is not None:
        print(f"Used fine-tuned ChromBERT checkpoint: {args.ft_ckpt}")
    else:
        print(f"Used pretrained ChromBERT")
    print(f"Dual-functional regulator subnetwork: {args.odir}/results/dual_regulator_*_subnetwork.pdf")
    print(f"Candidate cofactors: {args.odir}/results/dual_regulator_*_candidate_cofactors.csv")


@click.command(
    name="interpret_context_dependent_cofactor",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--region1-file",
    "region1_file",
    required=True,
    help="Different genomic regions for function1.",
)
@click.option(
    "--region2-file",
    "region2_file",
    required=True,
    help="Different genomic regions for function2. ",
)
@click.option(
    "--dual-regulator",
    "dual_regulator",
    required=True,
    help="Dual-functional regulator(s). Use ';' to separate multiple regulators.",
)
@click.option(
    "--ignore-regulator",
    "ignore_regulator",
    default=None,
    help="Regulators to ignore. Use ';' to separate multiple regulators.",
)
@click.option(
    "--threshold",
    "threshold",
    default=0.1,
    show_default=True,
    type=float,
    help="Threshold for the embedding similarity difference between two regions used to identify a regulator as a context-dependent cofactor.",
)
@click.option(
    "--odir",
    default="./output",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Output directory.",
)
@click.option("--gep", "gep", is_flag=True, default=False, show_default=True,
              help="Use GEP model (multi-flank-window). Default: False.")
@click.option("--flank-window", "flank_window",
              type=int,
              required=False, default=4, show_default=True,
              help="Flank window size for gep model.")
@click.option(
    "--genome",
    default="hg38",
    show_default=True,
    type=click.Choice(["hg38", "mm10"], case_sensitive=False),
    help="Genome version.",
)

@click.option(
    "--resolution", 
    default="1kb", 
    show_default=True,
    type=click.Choice(["200bp", "1kb", "2kb", "4kb"], case_sensitive=False),
    help="ChromBERT resolution."
)

@click.option(
    "--ft-ckpt",
    "ft_ckpt",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Fine-tuned ChromBERT checkpoint file.",
)

@click.option(
    "--batch-size",
    "batch_size",
    default=4,
    show_default=True,
    type=int,
    help="Batch size. Increase this value if you have sufficient GPU memory.",
)
@click.option(
    "--chrombert-cache-dir",
    "chrombert_cache_dir",
    default=os.path.expanduser("~/.cache/chrombert/data"),
    show_default=True,
    type=click.Path(file_okay=False),
    help="ChromBERT cache directory (containing config/ and anno/ subfolders).",
)
def interpret_context_dependent_cofactor(
    region1_file,
    region2_file,
    dual_regulator,
    ignore_regulator,
    threshold,
    odir, gep, flank_window, genome, resolution, ft_ckpt, batch_size, chrombert_cache_dir
):
    """
    Interpret context-dependent cofactors for dual-functional regulators.
    
    """
    args = SimpleNamespace(
        region1_file=region1_file,
        region2_file=region2_file,
        dual_regulator=dual_regulator,
        ignore_regulator=ignore_regulator,
        threshold=threshold,
        odir=odir,
        gep=gep,
        flank_window=flank_window,
        genome=genome,
        resolution=resolution,
        ft_ckpt=ft_ckpt,
        batch_size=batch_size,
        chrombert_cache_dir=chrombert_cache_dir,
    )
    run(args)


if __name__ == "__main__":
    interpret_context_dependent_cofactor()
