import os
import click
from types import SimpleNamespace
import numpy as np
import pandas as pd
import chrombert
from chrombert import ChromBERTFTConfig, DatasetConfig
import torch
from tqdm import tqdm
from .utils import resolve_paths, check_files, check_region_file

    
def model_emb_func(args,files_dict,odir):
    # init datamodule
    data_config = DatasetConfig(
        kind = "GeneralDataset",
        supervised_file = f"{odir}/model_input.tsv",
        hdf5_file = files_dict["hdf5_file"],
        batch_size = 4,
        num_workers = 8,
    )
    dl = data_config.init_dataloader()
    ds = data_config.init_dataset()

    # init chrombert
    model_config = ChromBERTFTConfig(
        genome = args.genome,
        dropout = 0,
        task = "general",
        pretrain_ckpt = files_dict["pretrain_ckpt"],
        mtx_mask = files_dict["mtx_mask"],
    )
    model_emb = model_config.init_model().get_embedding_manager().cuda().bfloat16()
    return ds, dl, model_emb


def cal_sim_tss_region_pairs(
    regions: pd.DataFrame,
    tss_df: pd.DataFrame,
    union_idx: np.ndarray,
    union_emb: np.ndarray,
    window: int = 250_000,
    chunk_size: int = 2_000_000,
    eps: float = 1e-12,
    pre_normalize: bool = True,
    out_col: str = "cos_sim",
) -> pd.DataFrame:
    """
    regions: overlap_bed_h1_chr1, columns: chrom,start,end,build_region_index
    tss_df : gene_tss_chr1, columns: chrom,start,end,build_region_index,tss,gene_name,gene_id
    union_idx: sorted unique global build_region_index, the order of union_idx must be consistent with the union_emb
    union_emb: embeddings aligned with union_idx, shape (len(union_idx), dim)

    return: pairs DataFrame with columns:
        chrom gene_id gene_name tss tss_build_region_index start end build_region_index dist dist_bin cos_sim
    """

    # ---------- Part 1: build pairs ----------
    r = regions[['chrom', 'start', 'end', 'build_region_index']].copy()
    t = tss_df[['chrom', 'tss', 'build_region_index', 'gene_name', 'gene_id']].copy()
    t = t.rename(columns={'build_region_index': 'tss_build_region_index'})

    r['start'] = r['start'].astype(np.int64)
    r['end']   = r['end'].astype(np.int64)
    r['build_region_index'] = r['build_region_index'].astype(np.int64)
    t['tss']   = t['tss'].astype(np.int64)
    t['tss_build_region_index'] = t['tss_build_region_index'].astype(np.int64)

    out = []

    for chrom, t_chr in t.groupby('chrom', sort=False):
        r_chr = r[r['chrom'] == chrom]
        if r_chr.empty:
            continue

        # sort once per chrom
        r_s = r_chr.sort_values('start').reset_index(drop=True)
        r_e = r_chr.sort_values('end').reset_index(drop=True)

        starts = r_s['start'].to_numpy()
        ends   = r_e['end'].to_numpy()
        n = len(r_chr)

        for row in t_chr.itertuples(index=False):
            tt = row.tss
            lo = tt - window
            hi = tt + window

            # candidate A: start <= hi  => r_s[:right]
            right = np.searchsorted(starts, hi, side='right')
            sizeA = right

            # candidate B: end >= lo  => r_e[pos:]
            pos = np.searchsorted(ends, lo, side='left')
            sizeB = n - pos

            if sizeA <= sizeB:
                cand = r_s.iloc[:right].copy()
                cand = cand[cand['end'].to_numpy() >= lo]
            else:
                cand = r_e.iloc[pos:].copy()
                cand = cand[cand['start'].to_numpy() <= hi]

            if cand.empty:
                continue

            s = cand['start'].to_numpy()
            e = cand['end'].to_numpy()

            # signed distance: downstream positive, upstream negative, inside 0
            dist = np.where(
                tt < s, s - tt,
                np.where(tt > e, e - tt, 0)
            )

            cand['tss'] = tt
            cand['tss_build_region_index'] = row.tss_build_region_index
            cand['dist'] = dist
            cand['gene_name'] = row.gene_name
            cand['gene_id'] = row.gene_id

            out.append(cand)

    if not out:
        return pd.DataFrame(columns=[
            'chrom', 'gene_id', 'gene_name', 'tss', 'tss_build_region_index', 'distal_region_start', 'distal_region_end', 'distal_region_build_region_index',
            'dist', 'dist_bin', out_col
        ])

    pairs = pd.concat(out, ignore_index=True)[['chrom', 'gene_id', 'gene_name', 'tss', 'tss_build_region_index', 'start', 'end', 'build_region_index', 'dist']].rename(columns={'start': 'distal_region_start', 'end': 'distal_region_end', 'build_region_index': 'distal_region_build_region_index'})
    pairs = pairs[np.abs(pairs['dist']) <= window].reset_index(drop=True)

    pairs['dist_bin'] = pairs['distal_region_build_region_index'].astype(np.int64) - pairs['tss_build_region_index'].astype(np.int64)

    # ---------- Part 2: cosine similarity with union_idx/union_emb ----------
    union_idx = np.asarray(union_idx, dtype=np.int64)
    E = union_emb

    if pre_normalize:
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + eps)

    idx_r = pairs['distal_region_build_region_index'].to_numpy(np.int64)
    idx_t = pairs['tss_build_region_index'].to_numpy(np.int64)

    pos_r = np.searchsorted(union_idx, idx_r)
    pos_t = np.searchsorted(union_idx, idx_t)

    # safety checks
    if (pos_r >= len(union_idx)).any() or (union_idx[pos_r] != idx_r).any():
        raise ValueError("Some region indices are not found in union_idx (check union_idx/union_emb alignment).")
    if (pos_t >= len(union_idx)).any() or (union_idx[pos_t] != idx_t).any():
        raise ValueError("Some tss indices are not found in union_idx (check union_idx/union_emb alignment).")

    cos_all = np.zeros(len(pairs), dtype=np.float32)

    for s0 in range(0, len(pairs), chunk_size):
        e0 = min(s0 + chunk_size, len(pairs))
        Vr = E[pos_r[s0:e0]]
        Vt = E[pos_t[s0:e0]]

        if pre_normalize:
            cos_all[s0:e0] = np.einsum("ij,ij->i", Vr, Vt).astype(np.float32)
        else:
            dot = np.einsum("ij,ij->i", Vr, Vt)
            nr  = np.linalg.norm(Vr, axis=1)
            nt  = np.linalg.norm(Vt, axis=1)
            cos_all[s0:e0] = (dot / (nr * nt + eps)).astype(np.float32)

    pairs[out_col] = cos_all
    pairs = pairs.query("dist_bin!=0").reset_index(drop=True)
    return pairs


def run(args, return_data=False):

    odir = args.odir
    os.makedirs(odir, exist_ok=True)

    focus_region = args.region
    files_dict = resolve_paths(args)
    check_files(files_dict, required_keys=[
        "chrombert_region_file",
        "hdf5_file",
        "pretrain_ckpt",
        "gene_meta_tsv"])

    # ---------- overlapping focus regions ----------
    overlap_bed = check_region_file(focus_region,files_dict,odir)
    
    # ---------- gene tss ----------
    gene_tss = pd.read_csv(files_dict["gene_meta_tsv"],sep='\t')
    gene_tss = gene_tss[['chrom','start','end','build_region_index','tss','gene_name','gene_id']]
    gene_tss = gene_tss.sort_values(by='build_region_index').reset_index(drop=True)
    gene_tss

    #---------- union idx (for pick embedding together)----------
    model_input = pd.concat([overlap_bed,gene_tss[['chrom','start','end','build_region_index']]]).drop_duplicates(subset='build_region_index').sort_values(by='build_region_index').reset_index(drop=True)
    model_input.to_csv(f"{odir}/model_input.tsv", sep="\t", index=False)
    use_idx = model_input.build_region_index.values

    # ---------- focus region embeddings ----------
    emb_npy_path = files_dict["region_emb_npy"]
    if os.path.exists(emb_npy_path):
        all_emb = np.load(emb_npy_path)
        region_embs = all_emb[use_idx]
        np.save(f"{odir}/use_region_emb.npy", region_embs)
    else:
        print(f"ChromBERT region embedding file not found: {emb_npy_path}, and not directly pick region embedding from cache dir.")
        print("Load model ChromBERT to embed focus regions.")
        ds, dl, model_emb = model_emb_func(args,files_dict,odir)
    
        region_embs = []
        with torch.no_grad():
            for batch in tqdm(dl, total = len(dl)):
                for k,v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()
                model_emb(batch) # initialize the cache 
                region_embs.append(model_emb.get_region_embedding().float().cpu().detach())
        region_embs = torch.cat(region_embs,axis=0).numpy()
        np.save(f"{odir}/use_region_emb.npy", region_embs)
        

    #-----------------calculate cosine similarity between tss and region pairs-----------------
    pairs_cos = cal_sim_tss_region_pairs(overlap_bed,gene_tss,use_idx,region_embs)
    pairs_cos.to_csv(f"{odir}/tss_region_pairs_cos.tsv", sep="\t", index=False)
    print("Finished!")
    print(f"Cosine similarity between tss and region pairs saved to: {odir}/tss_region_pairs_cos.tsv")
    
    if return_data:
        return pairs_cos

@click.command(name="infer_ep",
               context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--region", "region",
              type=click.Path(exists=True, dir_okay=False, readable=True),
              required=True, help="Region file.")
@click.option("--odir", default="./output", show_default=True,
              type=click.Path(file_okay=False), help="Output directory.")
@click.option("--genome", default="hg38", show_default=True,
              type=click.Choice(["hg38", "mm10"], case_sensitive=False), help="Genome.")
@click.option("--resolution", default="1kb", show_default=True,
              type=click.Choice(["1kb", "200bp", "2kb", "4kb"], case_sensitive=False), help="Resolution. Mouse only supports 1kb resolution.")

@click.option("--chrombert-cache-dir", "chrombert_cache_dir",
              default="~/.cache/chrombert/data",
              show_default=True,
              type=click.Path(file_okay=False),
              help="ChromBERT cache dir. ")

@click.option("--chrombert-region-file", "chrombert_region_file",
              default=None,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help="ChromBERT region BED file. If not provided, use the default hg38_6k_1kb_region.bed in the cache dir.")
@click.option("--chrombert-region-emb-file", "chrombert_region_emb_file",
              default=None,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help="ChromBERT region embedding file. If not provided, use the default hm_1kb_all_region_emb.npy in the cache dir.")



def infer_ep(region, odir, genome, resolution, chrombert_cache_dir,chrombert_region_file, chrombert_region_emb_file):      
    '''
    Infer enhancer-promoter loop
    '''
    args = SimpleNamespace(
        region=region,
        odir=odir,
        genome=genome,
        resolution=resolution,
        chrombert_cache_dir=chrombert_cache_dir,
        chrombert_region_file=chrombert_region_file,
        chrombert_region_emb_file=chrombert_region_emb_file
    )
    run(args)