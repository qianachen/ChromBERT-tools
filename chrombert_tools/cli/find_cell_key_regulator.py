import os
import glob
import click
from types import SimpleNamespace

from .utils import resolve_paths, check_files
from .utils import factor_rank
from .utils_embed import build_cell_model_emb
from .utils_interpret import embed_pool_func, build_interpret_config
from .interpret_regulators_across_regions import run as run_interpret_regulators_across_regions


def run(args):
    """
    Run key-regulator interpretation for a cell type or for two user-defined region sets.

    This command supports two main usage modes:

    1. Accessibility-driven mode
    Provide --cell-type-bw and --cell-type-peak.
    The workflow will build or load a cell-specific model, define region groups
    from the cell-type accessibility data, and rank candidate key regulators.

    2. Checkpoint-based mode
    Provide --ft-ckpt.

    a) If --region1-file and --region2-file are also provided,
        the workflow will directly compare regulator relevance between the two
        user-provided region sets.

    b) If --region1-file and --region2-file are not provided,
        you can provide --cell-type-bw and --cell-type-peak instead.
        The workflow will use the checkpoint together with the cell-type
        accessibility data to identify region groups automatically and then rank
        candidate key regulators.

    Output
    ------
    Ranked candidate regulators are saved to:
        {odir}/results/factor_importance_rank.csv
    """
    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    # The workflow needs either:
    #   (a) an existing fine-tuned checkpoint, or
    #   (b) cell-type accessibility inputs to build a cell-specific model.
    if args.ft_ckpt is None and (args.cell_type_bw is None or args.cell_type_peak is None):
        raise ValueError(
            "Please provide either --ft-ckpt, or both --cell-type-bw and --cell-type-peak."
        )

    # If the user already has a checkpoint and two region sets, run the direct
    # region-vs-region interpretation workflow and stop here.
    if (
        args.ft_ckpt is not None
        and args.region1_file is not None
        and args.region2_file is not None
    ):
        run_interpret_regulators_across_regions(args)
        return

    # Otherwise, run the default cell-type-based workflow.
    files_dict = resolve_paths(args)
    required_keys = [
        "chrombert_region_file",
        "hdf5_file",
    ]
    check_files(files_dict, required_keys=required_keys)

    # Organize outputs by stage so users can inspect intermediate files more easily.
    d_odir = f"{odir}/dataset"
    os.makedirs(d_odir, exist_ok=True)

    train_odir = f"{odir}/train"
    os.makedirs(train_odir, exist_ok=True)

    results_odir = f"{odir}/results"
    os.makedirs(results_odir, exist_ok=True)

    emb_odir = f"{odir}/emb"
    os.makedirs(emb_odir, exist_ok=True)

    # Step 1. Build or load a cell-specific embedding model.
    # If no checkpoint is given, this step will prepare the required model state
    # from the provided cell-type accessibility data.
    print("Step 1/4: Building or loading a cell-specific model...")
    model_emb = build_cell_model_emb(args, files_dict, odir)

    # Step 2. Build the interpretation dataset.
    # In the default workflow, the method contrasts highly accessible regions
    # against background regions for regulator ranking.
    print("Step 2/4: Preparing region groups for interpretation...")
    data_config, _ = build_interpret_config(
        args, files_dict, f"{d_odir}/highly_accessible_region.csv"
    )

    # Step 3. Generate pooled regulator embeddings for each region group.
    # These pooled embeddings are then compared to identify candidate regulators
    # associated with cell-type-specific accessible regions.
    print("Step 3/4: Generating regulator embeddings for region groups...")
    embs_pool_up_region, regulators = embed_pool_func(
        data_config,
        model_emb,
        f"{d_odir}/highly_accessible_region.csv",
        emb_odir,
        "highly_accessible_region",
    )
    embs_pool_nochange_region, regulators = embed_pool_func(
        data_config,
        model_emb,
        f"{d_odir}/background_region.csv",
        emb_odir,
        "background_region",
    )

    # Step 4. Rank regulators by comparing the two region groups.
    # Higher-ranked regulators are stronger candidates for explaining the
    # difference between highly accessible and background regions.
    print("Step 4/4: Ranking candidate key regulators...")
    cos_sim_df = factor_rank(
        embs_pool_up_region,
        embs_pool_nochange_region,
        regulators,
        results_odir,
    )

    print("Top 25 candidate regulators:")
    print(cos_sim_df.head(n=25))

    print("Analysis finished.")

    if args.ft_ckpt is not None:
        print(f"Checkpoint used for interpretation: {args.ft_ckpt}")
    else:
        ckpts = glob.glob(f"{train_odir}/**/checkpoints/*.ckpt", recursive=True)
        if not ckpts:
            raise FileNotFoundError(
                f"No checkpoint found under {train_odir}/**/checkpoints/*.ckpt after training."
            )
        ft_ckpt = os.path.abspath(max(ckpts, key=os.path.getmtime))
        print(f"Checkpoint used for interpretation: {ft_ckpt}")

    print(f"Ranked regulators saved to: {results_odir}/factor_importance_rank.csv")


@click.command(
    name="find_cell_key_regulator",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--cell-type-bw",
    "cell_type_bw",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=False,
    help=(
        "BigWig file of chromatin accessibility for the target cell type. "
        "Use this together with --cell-type-peak when you do not already have "
        "a fine-tuned checkpoint."
    ),
)
@click.option(
    "--cell-type-peak",
    "cell_type_peak",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=False,
    help=(
        "BED file of accessible peaks for the target cell type. "
        "Use this together with --cell-type-bw."
    ),
)
@click.option(
    "--ft-ckpt",
    "ft_ckpt",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=False,
    default=None,
    show_default=True,
    help=(
        "Path to a fine-tuned ChromBERT checkpoint. "
        "If provided, the workflow skips model fine-tuning and uses this checkpoint directly."
    ),
)
@click.option(
    "--region1-file",
    "region1_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=False,
    help=(
        "BED file for region set 1. "
        "Use together with --region2-file and --ft-ckpt for direct comparison between two region groups."
    ),
)
@click.option(
    "--region2-file",
    "region2_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=False,
    help=(
        "BED file for region set 2. "
        "Use together with --region1-file and --ft-ckpt for direct comparison between two region groups."
    ),
)
@click.option(
    "--genome",
    default="hg38",
    show_default=True,
    type=click.Choice(["hg38", "mm10"], case_sensitive=False),
    help="Reference genome used by ChromBERT resources.",
)
@click.option(
    "--resolution",
    default="1kb",
    show_default=True,
    type=click.Choice(["200bp", "1kb", "2kb", "4kb"], case_sensitive=False),
    help="Resolution of the ChromBERT representation.",
)
@click.option(
    "--odir",
    default="./output",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Output directory for intermediate files, embeddings, and final ranking results.",
)
@click.option(
    "--mode",
    default="fast",
    show_default=True,
    type=click.Choice(["fast", "full"], case_sensitive=False),
    help=(
        "Run mode. "
        "'fast' uses a subset of regions for quicker execution, while "
        "'full' uses all eligible regions."
    ),
)
@click.option(
    "--batch-size",
    "batch_size",
    default=4,
    show_default=True,
    type=int,
    help="Batch size used for embedding generation and model inference.",
)
@click.option(
    "--chrombert-cache-dir",
    "chrombert_cache_dir",
    default="~/.cache/chrombert/data",
    show_default=True,
    type=click.Path(file_okay=False),
    help=(
        "Directory containing ChromBERT cached resources, such as configs, "
        "annotations, checkpoints, and reference embeddings."
    ),
)
def find_cell_key_regulator(
    cell_type_bw,
    cell_type_peak,
    region1_file,
    region2_file,
    ft_ckpt,
    odir,
    mode,
    batch_size,
    chrombert_cache_dir,
    genome,
    resolution,
):
    """
    Find candidate key regulators for a target cell type or between two region sets.

    Typical usage
    
    1. Start from accessibility data
    
       Provide:
         --cell-type-bw
         --cell-type-peak

       This mode builds or loads a cell-specific model and identifies candidate regulators associated with accessible regions.

    2. Start from an existing fine-tuned checkpoint
    
        Provide --ft-ckpt.

        a) If --region1-file and --region2-file are also provided, the workflow will directly compare regulator relevance between the two
            user-provided region sets.

        b) If --region1-file and --region2-file are not provided,
            you can provide --cell-type-bw and --cell-type-peak instead.The workflow will use the checkpoint together with the cell-type
            accessibility data to identify region groups automatically and then rank candidate key regulators.
    """
    args = SimpleNamespace(
        cell_type_bw=cell_type_bw,
        cell_type_peak=cell_type_peak,
        region1_file=region1_file,
        region2_file=region2_file,
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
    find_cell_key_regulator()