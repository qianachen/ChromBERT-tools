import os

import click
from types import SimpleNamespace

import pandas as pd

from .utils import resolve_paths, check_files, check_region_file
from .predict_region_activity_regression import make_acc_dataset, load_train_model_acc, _semicolon_paths
from .predict_gene_expression import make_exp_dataset, load_train_model_gep
from .interpret_regulator_effects_between_regions_groups import factor_rank_across_regions

def run(args):
    """Main execution function for finding driver factors in cell state transitions."""
    odir = args.odir
    os.makedirs(odir, exist_ok=True)
    
    files_dict = resolve_paths(args)
    required_keys = [
        "chrombert_region_file",
        "hdf5_file",
        # "pretrain_ckpt",
        # "mtx_mask",
        "gene_meta_tsv"
    ]
    check_files(files_dict, required_keys=required_keys)
    
    
    # 1. prepare dataset
   
    exp_bool = False
    acc_bool = False
    if args.exp_tpm1 is not None and args.exp_tpm2 is not None:
        exp_bool = True; print(f"Whether to train ChromBERT to predict expression changes in cell state transition: {exp_bool}")
        exp_odir = f'{odir}/exp'; os.makedirs(exp_odir, exist_ok=True)
        exp_data_odir = f'{exp_odir}/dataset'; os.makedirs(exp_data_odir, exist_ok=True)
        
        print("Stage 1 (exp): prepare dataset")
        args_exp, _ = make_exp_dataset(args, files_dict, exp_data_odir)
        print("Finished Stage 1 (exp)")
        
        print("Stage 2 (exp): train ChromBERT to predict expression changes in cell state transition")
        model_tuned, data_config = load_train_model_gep(args_exp, files_dict, exp_odir)
        print("Finished Stage 2 (exp)")
        
        print("Stage 3 (exp): infer driver factors in different expression activity genes")
        exp_emb_odir = f"{exp_odir}/emb"; os.makedirs(exp_emb_odir, exist_ok=True)
        exp_results_odir = f"{exp_odir}/results"; os.makedirs(exp_results_odir, exist_ok=True)
        cos_sim_df_exp = factor_rank_across_regions(data_config, model_tuned, exp_data_odir, files_dict, exp_emb_odir, exp_results_odir)
        print("Finished stage 3 (exp)")
        
    if args.acc_peak1 is not None and args.acc_peak2 is not None and args.acc_signal1 is not None and args.acc_signal2 is not None:
        acc_bool = True; print(f"Whether to train ChromBERT to predict chromatin accessibility changes in cell state transition: {acc_bool}")
        acc_odir = f'{odir}/acc'; os.makedirs(acc_odir, exist_ok=True)
        acc_data_odir = f'{acc_odir}/dataset'; os.makedirs(acc_data_odir, exist_ok=True)
        
        print("Stage 1 (acc): prepare dataset")
        args_acc, _ = make_acc_dataset(args, files_dict, acc_data_odir)
        print("Finished Stage 1 (acc)")
        
        print("Stage 2 (acc): train ChromBERT to predict chromatin accessibility changes in cell state transition")
        model_tuned, data_config = load_train_model_acc(args_acc, files_dict, acc_odir)
        print("Finished Stage 2 (acc)")
    
        print("Stage 3 (acc): infer driver factors in different chromatin accessibility activity genes")
        acc_emb_odir = f"{acc_odir}/emb"; os.makedirs(acc_emb_odir, exist_ok=True)
        acc_results_odir = f"{acc_odir}/results"; os.makedirs(acc_results_odir, exist_ok=True)
        cos_sim_df_acc = factor_rank_across_regions(data_config, model_tuned, acc_data_odir, files_dict, acc_emb_odir, acc_results_odir)
        print("Finished stage 3 (acc)")
        
        
        
    if acc_bool and exp_bool:
        merge_odir = os.path.join(odir, "merge"); os.makedirs(merge_odir, exist_ok=True)
        print("Merging factor ranks from expression and chromatin accessibility")
        merge_df = pd.merge(cos_sim_df_exp,cos_sim_df_acc,on='factors',how='inner',suffixes=['_exp','_acc'])
        merge_df['total_rank']=((merge_df['rank_exp']+merge_df['rank_acc'])/2).rank().astype(int)
        merge_df = merge_df.sort_values('total_rank').reset_index(drop=True)
        merge_df.to_csv(os.path.join(merge_odir, "factor_importance_rank.csv"), index=False)
        print("Finished merging factor ranks from expression and chromatin accessibility")
    
    print("Finished all stages!")
    if exp_bool:
        if hasattr(args, 'ft_ckpt_exp') and args.ft_ckpt_exp:
            print(f"Used fine-tuned ChromBERT checkpoint: {args.ft_ckpt_exp}")
        else:
            print(f"Fine-tuned ChromBERT model (for expression changes) saved in: {exp_odir}/train/")
        print(f"Driver factors for expression changes: {exp_results_odir}/factor_importance_rank.csv")
        
    if acc_bool:
        if hasattr(args, 'ft_ckpt_acc') and args.ft_ckpt_acc:
            print(f"Used fine-tuned ChromBERT checkpoint: {args.ft_ckpt_acc}")
        else:
            print(f"Fine-tuned ChromBERT model (for accessibility changes) saved in: {acc_odir}/train/")
        print(f"Driver factors for chromatin accessibility changes: {acc_results_odir}/factor_importance_rank.csv")
        
    if acc_bool and exp_bool:
        print(f"Integrated driver factor rankings: {merge_odir}/factor_importance_rank.csv")

@click.command(
    name="find_driver_in_transition",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--exp-tpm1",
    "exp_tpm1",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Expression (TPM) file for cell state 1. CSV format with 'gene_id' and 'tpm' columns.",
)
@click.option(
    "--exp-tpm2",
    "exp_tpm2",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Expression (TPM) file for cell state 2. CSV format with 'gene_id' and 'tpm' columns.",
)
@click.option(
    "--acc-peak1",
    "acc_peak1",
    default=None,
    required=False,
    help="BED peak file(s) for state 1; use ';' for multiple. Required for training unless predict-only "
         "(--ft-ckpt + --predict-file).",
)
@click.option(
    "--acc-peak2",
    "acc_peak2",
    required=False,
    default=None,
    help="Optional BED peak file(s) for state 2 (same ';' / overlap / union as state 1). Omit to use only state-1 peaks in the peak union.",
)
@click.option(
    "--acc-signal1",
    "acc_signal1",
    default=None,
    required=False,
    help="BigWig(s) for state 1; use ';' for replicates. Required for training unless predict-only.",
)
@click.option(
    "--acc-signal2",
    "acc_signal2",
    default=None,
    required=False,
    help="Optional. BigWig(s) for state 2 (';' for replicates). Omit for single-state mode (predict state-1 log2 signal only).",
)
@click.option(
    "--direction",
    default="2-1",
    show_default=True,
    type=click.Choice(["2-1", "1-2"], case_sensitive=False),
    help="Direction of cell state transition: '2-1' means from state 1 to state 2; '1-2' means from state 2 to state 1.",
)
@click.option(
    "--odir",
    default="./output",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Output directory.",
)
@click.option(
    "--genome",
    default="hg38",
    show_default=True,
    type=click.Choice(["hg38", "mm10"], case_sensitive=False),
    help="Reference genome.",
)
@click.option(
    "--resolution",
    default="1kb",
    show_default=True,
    type=click.Choice(["200bp", "1kb", "2kb", "4kb"], case_sensitive=False),
    help="ChromBERT resolution.",
)
@click.option(
    "--mode",
    default="fast",
    show_default=True,
    type=click.Choice(["fast", "full"], case_sensitive=False),
    help="Training mode: 'fast' downsamples to 20k regions for quicker training; 'full' uses all regions.",
)
@click.option(
    "--ft-ckpt-exp",
    "ft_ckpt_exp",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Fine-tuned ChromBERT checkpoint file for expression changes.",
)
@click.option(
    "--ft-ckpt-acc",
    "ft_ckpt_acc",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Fine-tuned ChromBERT checkpoint file for chromatin accessibility changes.",
)
@click.option(
    "--chrombert-cache-dir",
    "chrombert_cache_dir",
    default="~/.cache/chrombert/data",
    show_default=True,
    type=click.Path(file_okay=False),
    help="ChromBERT cache directory (containing config/ and anno/ subfolders).",
)
@click.option(
    "--batch-size",
    "batch_size",
    default=4,
    show_default=True,
    type=int,
    help="Batch size. Increase this value if you have sufficient GPU memory.",
)

def find_driver_in_transition(
    exp_tpm1,
    exp_tpm2,
    acc_peak1,
    acc_peak2,
    acc_signal1,
    acc_signal2,
    direction,
    odir,
    genome,
    resolution,
    mode,
    ft_ckpt_exp,
    ft_ckpt_acc,
    chrombert_cache_dir,
    batch_size
):
    """
    Find driver factors in cell state transitions.
    
    This tool identifies key transcription factors that drive cell state transitions
    by analyzing changes in gene expression and/or chromatin accessibility between
    two cell states.
    
    You must provide at least one of the following:
    - Expression data (--exp-tpm1 and --exp-tpm2)
    - Accessibility data (--acc-peak1, --acc-peak2, --acc-signal1, --acc-signal2)
    
    Providing both expression and accessibility data yields more confident results.
    """
    args = SimpleNamespace(
        exp_tpm1=exp_tpm1,
        exp_tpm2=exp_tpm2,
        acc_peak1=acc_peak1,
        acc_peak2=acc_peak2,
        acc_signal1=acc_signal1,
        acc_signal2=acc_signal2,
        direction=direction,
        odir=odir,
        genome=genome,
        resolution=resolution,
        mode=mode,
        ft_ckpt_exp=ft_ckpt_exp,
        ft_ckpt_acc=ft_ckpt_acc,
        chrombert_cache_dir=chrombert_cache_dir,
        batch_size = batch_size,
    )
    run(args)


if __name__ == "__main__":
    find_driver_in_transition()