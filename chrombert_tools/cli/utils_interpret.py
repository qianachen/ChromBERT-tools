"""
Shared helpers for interpret_* CLIs (regulator embeddings, DatasetConfig, ChromBERTFTConfig).
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json
from chrombert_hf import ChromBERTFTConfig, DatasetConfig

from .utils import get_model_name, overlap_regulator_func

def resolve_ignore_object(
    ignore_regulator: Optional[str], chrombert_regulator_file: str
) -> Tuple[bool, Optional[str]]:
    """
    Map CLI ignore string to dataset/model ignore_object.

    Returns:
        (ignore, ignore_object) where ignore_object is ';'-joined ChromBERT names or None.
    """
    if ignore_regulator is None:
        return False, None
    overlap_ignore, _, _ = overlap_regulator_func(
        ignore_regulator, chrombert_regulator_file
    )
    ignore_object = ";".join(overlap_ignore) if overlap_ignore else None
    return bool(ignore_object), ignore_object


def build_interpret_config(
    args: Any,
    files_dict: dict,
    supervised_file_for_ignore_idx: str,
    gep: bool = False,
    flank_window: int = 4,
    ignore_regulator: Optional[str] = None,
) -> Tuple[DatasetConfig, ChromBERTFTConfig]:
    """
    Build DatasetConfig + ChromBERTFTConfig like interpret_regulators_across_regions.

    args must provide: genome, resolution, batch_size, ft_ckpt.
    gep / flank_window: taken from args if present, else use the function parameters
    (defaults: gep=False, flank_window=4). Optional: ignore_regulator on args.

    supervised_file_for_ignore_idx: TSV used only to init_dataset when computing ignore_index.
    """
    if args.model_config is not None and args.data_config is not None:
        with open(args.model_config, "r") as f:
            model_config_dict = json.load(f)
        with open(args.data_config, "r") as f:
            data_config_dict = json.load(f)
        model_config = ChromBERTFTConfig(**model_config_dict)
        data_config = DatasetConfig(**data_config_dict)
    else:
        ignore_regulator = getattr(args, "ignore_regulator", ignore_regulator)
        ignore, ignore_object = resolve_ignore_object(
            ignore_regulator,
            files_dict["chrombert_regulator_file"],
        )
        ignore_index = None
        gep = getattr(args, "gep", gep)
        flank_window = getattr(args, "flank_window", flank_window)
        if not gep:
            data_config = DatasetConfig(
                kind="GeneralDataset",
                supervised_file=None,
                hdf5_file=files_dict["hdf5_file"],
                batch_size=args.batch_size,
                num_workers=8,
                meta_file=files_dict["meta_file"],
            )
            if ignore:
                data_config.ignore = ignore
                data_config.ignore_object = ignore_object
                ds0 = data_config.init_dataset(supervised_file=supervised_file_for_ignore_idx)
                ignore_index = ds0[0]["ignore_index"]

            model_config = ChromBERTFTConfig(
                genome=args.genome,
                task="general",
                dropout=0,
                pretrained_model_name_or_path=get_model_name(args.genome, args.resolution),
                pretrain_ckpt=files_dict["pretrain_ckpt"],
                mtx_mask=files_dict["mtx_mask"],
                finetune_ckpt=args.ft_ckpt,
                ignore=ignore,
                ignore_index=ignore_index,
            )
        else:
            data_config = DatasetConfig(
                kind="MultiFlankwindowDataset",
                supervised_file=None,
                hdf5_file=files_dict["hdf5_file"],
                batch_size=args.batch_size,
                num_workers=2,
                meta_file=files_dict["meta_file"],
                flank_window=flank_window,
            )
            if ignore:
                data_config.ignore = ignore
                data_config.ignore_object = ignore_object
                ds0 = data_config.init_dataset(supervised_file=supervised_file_for_ignore_idx)
                ignore_index = ds0[0]["ignore_index"]

            model_config = ChromBERTFTConfig(
                genome=args.genome,
                task="gep",
                dropout=0,
                pretrained_model_name_or_path=get_model_name(args.genome, args.resolution),
                pretrain_ckpt=files_dict["pretrain_ckpt"],
                mtx_mask=files_dict["mtx_mask"],
                finetune_ckpt=args.ft_ckpt,
                gep_flank_window=flank_window,
                ignore=ignore,
                ignore_index=ignore_index,
            )

    return data_config, model_config


def load_interpret_model(model_config: ChromBERTFTConfig):
    """init_model on CUDA, eval, return (model_tuned, bfloat16 embedding manager on CUDA)."""
    model_tuned = model_config.init_model().cuda().eval()
    model_emb = model_tuned.get_embedding_manager().cuda().bfloat16()
    return model_tuned, model_emb


def batch_num_regions(batch: dict) -> int:
    """Batch size key for GeneralDataset ('region') vs GEP ('center_region')."""
    if "region" in batch:
        return batch["region"].shape[0]
    return batch["center_region"].shape[0]


def embed_pool_func(
    data_config: DatasetConfig,
    model_emb: torch.nn.Module,
    sup_file: str,
    emb_odir: str,
    region_name: str,
    show_progress: bool = True,
):
    """
    Sum regulator embeddings over all samples in dataloader(supervised_file=sup_file).

    region_name: optional name for error messages (e.g. 'region1').
    """
    regulators = model_emb.regulator_names
    dl = data_config.init_dataloader(supervised_file=sup_file)
    embs_sum = np.zeros((len(regulators), 768), dtype=np.float64)
    total_counts = 0

    iterator = tqdm(dl) if show_progress else dl
    with torch.no_grad():
        for batch in iterator:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            emb = model_emb(batch)
            emb_np = emb.float().cpu().numpy()
            total_counts += batch_num_regions(batch)
            embs_sum += emb_np.sum(axis=0)

    if total_counts == 0:
        loc = f"region {region_name!r} " if region_name else ""
        raise ValueError(
            f"No batches for {loc}(supervised_file={sup_file!r}). "
            "Check region overlap and model_input.tsv."
        )
    regulator_idx_dict = {regulator: idx for idx, regulator in enumerate(regulators)}
    embs_pool = embs_sum / total_counts
    embs_pool_dict = {
        regulator: embs_pool[regulator_idx_dict[regulator]] for regulator in regulators
    }
    out_pkl = os.path.join(emb_odir, f"{region_name}_regulator_embs_dict.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(embs_pool_dict, f)
    return embs_pool, regulators


def coerce_region_interpret_args(args: Any) -> None:
    """Defaults for interpret_region_interactions / infer_ep when using build_interpret_config."""
    defaults = {
        "batch_size": 4,
        "gep": False,
        "flank_window": 4,
        "ft_ckpt": None,
        "ignore_regulator": None,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)


def collect_region_embeddings_from_dataloader(
    data_config: DatasetConfig,
    model_emb: torch.nn.Module,
    sup_file: str,
    show_progress: bool = True,
) -> np.ndarray:
    """Region-level vectors [N, D] in the same order as rows in supervised_file."""
    dl = data_config.init_dataloader(supervised_file=sup_file)
    chunks: list[np.ndarray] = []
    iterator = tqdm(dl) if show_progress else dl
    with torch.no_grad():
        for batch in iterator:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            model_emb(batch)
            chunks.append(model_emb.get_region_embedding().float().cpu().numpy())
    if not chunks:
        raise ValueError(
            f"No batches for region embedding (supervised_file={sup_file!r}). "
            "Check model_input.tsv."
        )
    return np.concatenate(chunks, axis=0)


def load_union_region_embeddings(
    files_dict: dict,
    model_input: pd.DataFrame,
    odir: str,
    args: Any,
    save_npy: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort model_input by build_region_index, write model_input.tsv, return (union_idx, emb)
    with emb[i] matching union_idx[i].
    """
    model_input = (
        model_input[["chrom", "start", "end", "build_region_index"]]
        .copy()
        .sort_values(by="build_region_index")
        .reset_index(drop=True)
    )
    os.makedirs(odir, exist_ok=True)
    sup_file = os.path.join(odir, "model_input.tsv")
    model_input.to_csv(sup_file, sep="\t", index=False)
    union_idx = model_input["build_region_index"].to_numpy(dtype=np.int64)
    emb_path = files_dict["region_emb_npy"]
    if os.path.exists(emb_path):
        all_emb = np.load(emb_path)
        region_embs = all_emb[union_idx]
    else:
        print(
            f"No cached region embedding at {emb_path}; "
            "running ChromBERT embedding manager (same stack as interpret_regulator_interactions)."
        )
        coerce_region_interpret_args(args)
        data_config, model_config = build_interpret_config(args, files_dict, sup_file)
        _, model_emb = load_interpret_model(model_config)
        region_embs = collect_region_embeddings_from_dataloader(
            data_config, model_emb, sup_file, show_progress=True
        )
    if save_npy:
        np.save(os.path.join(odir, "use_region_emb.npy"), region_embs)
    return union_idx, region_embs


def cal_sim_tss_region_pairs(
    regions: pd.DataFrame,
    tss_df: pd.DataFrame,
    union_idx: np.ndarray,
    union_emb: np.ndarray,
    distance_min: int = 0,
    distance_max: int = 250_000,
    chunk_size: int = 2_000_000,
    eps: float = 1e-12,
    pre_normalize: bool = True,
    out_col: str = "cos_sim",
    window: Optional[int] = None,
) -> pd.DataFrame:
    """
    Enhancer–promoter style: pairs of TSS regions vs distal regions, cosine sim.

    Pairs kept when ``distance_min <= |dist| <= distance_max`` (bp), where ``dist`` is the
    signed shortest gap between the TSS point and the distal interval (negative when the
    TSS lies downstream of the interval, positive upstream, 0 when TSS is inside).
    Direction is ignored: filtering uses the absolute value.

    regions: columns chrom,start,end,build_region_index
    tss_df: chrom,start,end,build_region_index,tss,gene_name,gene_id
    union_idx / union_emb: aligned global region index and embeddings.
    window: deprecated alias for ``distance_max`` (with ``distance_min=0``); kept for
    backward compatibility.
    """
    if window is not None:
        distance_max = int(window)
        distance_min = 0

    distance_min = int(abs(distance_min))
    distance_max = int(abs(distance_max))
    if distance_min > distance_max:
        raise ValueError(
            f"distance_min ({distance_min}) must be <= distance_max ({distance_max})."
        )

    r = regions[["chrom", "start", "end", "build_region_index"]].copy()
    t = tss_df[["chrom", "tss", "build_region_index", "gene_name", "gene_id"]].copy()
    t = t.rename(columns={"build_region_index": "tss_build_region_index"})

    r["start"] = r["start"].astype(np.int64)
    r["end"] = r["end"].astype(np.int64)
    r["build_region_index"] = r["build_region_index"].astype(np.int64)
    t["tss"] = t["tss"].astype(np.int64)
    t["tss_build_region_index"] = t["tss_build_region_index"].astype(np.int64)

    out = []

    for chrom, t_chr in t.groupby("chrom", sort=False):
        r_chr = r[r["chrom"] == chrom]
        if r_chr.empty:
            continue

        r_s = r_chr.sort_values("start").reset_index(drop=True)
        r_e = r_chr.sort_values("end").reset_index(drop=True)

        starts = r_s["start"].to_numpy()
        ends = r_e["end"].to_numpy()
        n = len(r_chr)

        for row in t_chr.itertuples(index=False):
            tt = row.tss
            lo = tt - distance_max
            hi = tt + distance_max

            right = np.searchsorted(starts, hi, side="right")
            sizeA = right

            pos = np.searchsorted(ends, lo, side="left")
            sizeB = n - pos

            if sizeA <= sizeB:
                cand = r_s.iloc[:right].copy()
                cand = cand[cand["end"].to_numpy() >= lo]
            else:
                cand = r_e.iloc[pos:].copy()
                cand = cand[cand["start"].to_numpy() <= hi]

            if cand.empty:
                continue

            s = cand["start"].to_numpy()
            e = cand["end"].to_numpy()

            dist = np.where(
                tt < s,
                s - tt,
                np.where(tt > e, e - tt, 0),
            )

            cand["tss"] = tt
            cand["tss_build_region_index"] = row.tss_build_region_index
            cand["dist"] = dist
            cand["gene_name"] = row.gene_name
            cand["gene_id"] = row.gene_id

            out.append(cand)

    if not out:
        return pd.DataFrame(
            columns=[
                "chrom",
                "gene_id",
                "gene_name",
                "tss",
                "tss_build_region_index",
                "distal_region_start",
                "distal_region_end",
                "distal_region_build_region_index",
                "dist",
                "dist_bin",
                out_col,
            ]
        )

    pairs = pd.concat(out, ignore_index=True)[
        [
            "chrom",
            "gene_id",
            "gene_name",
            "tss",
            "tss_build_region_index",
            "start",
            "end",
            "build_region_index",
            "dist",
        ]
    ].rename(
        columns={
            "start": "distal_region_start",
            "end": "distal_region_end",
            "build_region_index": "distal_region_build_region_index",
        }
    )
    abs_dist = np.abs(pairs["dist"].to_numpy())
    keep = (abs_dist >= distance_min) & (abs_dist <= distance_max)
    pairs = pairs[keep].reset_index(drop=True)

    pairs["dist_bin"] = (
        pairs["distal_region_build_region_index"].astype(np.int64)
        - pairs["tss_build_region_index"].astype(np.int64)
    )

    union_idx = np.asarray(union_idx, dtype=np.int64)
    E = union_emb

    if pre_normalize:
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + eps)

    idx_r = pairs["distal_region_build_region_index"].to_numpy(np.int64)
    idx_t = pairs["tss_build_region_index"].to_numpy(np.int64)

    pos_r = np.searchsorted(union_idx, idx_r)
    pos_t = np.searchsorted(union_idx, idx_t)

    if (pos_r >= len(union_idx)).any() or (union_idx[pos_r] != idx_r).any():
        raise ValueError(
            "Some distal indices are not found in union_idx (check union_idx/union_emb alignment)."
        )
    if (pos_t >= len(union_idx)).any() or (union_idx[pos_t] != idx_t).any():
        raise ValueError(
            "Some TSS indices are not found in union_idx (check union_idx/union_emb alignment)."
        )

    cos_all = np.zeros(len(pairs), dtype=np.float32)

    for s0 in range(0, len(pairs), chunk_size):
        e0 = min(s0 + chunk_size, len(pairs))
        Vr = E[pos_r[s0:e0]]
        Vt = E[pos_t[s0:e0]]

        if pre_normalize:
            cos_all[s0:e0] = np.einsum("ij,ij->i", Vr, Vt).astype(np.float32)
        else:
            dot = np.einsum("ij,ij->i", Vr, Vt)
            nr = np.linalg.norm(Vr, axis=1)
            nt = np.linalg.norm(Vt, axis=1)
            cos_all[s0:e0] = (dot / (nr * nt + eps)).astype(np.float32)

    pairs[out_col] = cos_all
    pairs = pairs.query("dist_bin!=0").reset_index(drop=True)
    return pairs


def interval_min_separation_bp(s1: int, e1: int, s2: int, e2: int) -> int:
    """
    Shortest distance in bp between any two points on two closed intervals (same chromosome).
    0 if they overlap or touch.
    """
    if max(s1, s2) <= min(e1, e2):
        return 0
    if e1 < s2:
        return int(s2 - e1)
    if e2 < s1:
        return int(s1 - e2)
    return 0


def cross_region_set_cosine_pairs(
    overlap_a: pd.DataFrame,
    overlap_b: pd.DataFrame,
    union_idx: np.ndarray,
    union_emb: np.ndarray,
    distance_min: int = 0,
    distance_max: Optional[int] = None,
    eps: float = 1e-12,
    max_genomic_dist_bp: Optional[int] = None,
) -> pd.DataFrame:
    """
    Pairs (row in overlap_a) x (row in overlap_b): cosine between region embeddings.

    If ``distance_max`` is set: only same-chromosome pairs whose interval separation
    (the minimum unsigned gap between the two intervals, 0 if overlapping) lies in
    ``[distance_min, distance_max]`` (bp). The separation is always non-negative, so
    direction (upstream vs downstream of set1) is ignored. Different chromosomes are
    skipped and the column ``genomic_dist_bp`` is added.

    Implemented by sorting set2 by start and, per set1 interval [s1,e1], taking
    candidates with s2 <= e1+distance_max and e2 >= s1-distance_max, then batching
    cosine dots and applying the lower bound ``distance_min``.

    If ``distance_max`` is None: full Cartesian product (all chromosome pairs);
    ``distance_min`` is ignored.

    ``max_genomic_dist_bp`` is a deprecated alias of ``distance_max`` (with
    ``distance_min=0``); kept for backward compatibility.
    """
    if max_genomic_dist_bp is not None and distance_max is None:
        distance_max = int(max_genomic_dist_bp)
        distance_min = 0

    if distance_max is not None:
        distance_min = int(abs(distance_min))
        distance_max = int(abs(distance_max))
        if distance_min > distance_max:
            raise ValueError(
                f"distance_min ({distance_min}) must be <= distance_max ({distance_max})."
            )

    oa = overlap_a.reset_index(drop=True)
    ob = overlap_b.reset_index(drop=True)
    union_idx = np.asarray(union_idx, dtype=np.int64)
    idx_to_row = {int(u): k for k, u in enumerate(union_idx)}

    i1 = oa["build_region_index"].astype(np.int64).to_numpy()
    i2 = ob["build_region_index"].astype(np.int64).to_numpy()
    for ix in np.unique(np.concatenate([i1, i2])):
        if int(ix) not in idx_to_row:
            raise ValueError(
                f"build_region_index {ix} not in union_idx (check overlaps vs model_input union)."
            )

    p1 = np.array([idx_to_row[int(x)] for x in i1])
    p2 = np.array([idx_to_row[int(x)] for x in i2])
    E1 = union_emb[p1].astype(np.float64)
    E2 = union_emb[p2].astype(np.float64)
    E1n = E1 / (np.linalg.norm(E1, axis=1, keepdims=True) + eps)
    E2n = E2 / (np.linalg.norm(E2, axis=1, keepdims=True) + eps)
    if distance_max is None:
        cos_mat = (E1n @ E2n.T).astype(np.float32)
        n1, n2 = cos_mat.shape
        ri, cj = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
        ri = ri.ravel()
        cj = cj.ravel()
        return pd.DataFrame(
            {
                "set1_chrom": oa["chrom"].iloc[ri].to_numpy(),
                "set1_start": oa["start"].iloc[ri].to_numpy(),
                "set1_end": oa["end"].iloc[ri].to_numpy(),
                "set1_build_region_index": oa["build_region_index"].iloc[ri].to_numpy(),
                "set2_chrom": ob["chrom"].iloc[cj].to_numpy(),
                "set2_start": ob["start"].iloc[cj].to_numpy(),
                "set2_end": ob["end"].iloc[cj].to_numpy(),
                "set2_build_region_index": ob["build_region_index"].iloc[cj].to_numpy(),
                "cos_sim": cos_mat.ravel(),
            }
        )

    # Same-chrom pairs with distance_min <= sep <= distance_max:
    # sep >= 0 by construction (min unsigned gap), so the direction (upstream/downstream)
    # of set2 relative to set1 doesn't matter. Sort set2 by start; for each interval in
    # set1 use searchsorted + end filter (candidates are [s2,e2] intersecting
    # [s1-Dmax, e1+Dmax]); apply the lower bound after computing sep.
    Dmax = int(distance_max)
    Dmin = int(distance_min)
    c1 = oa["chrom"].astype(np.int64).to_numpy()
    s1a = oa["start"].astype(np.int64).to_numpy()
    e1a = oa["end"].astype(np.int64).to_numpy()
    bri1 = oa["build_region_index"].astype(np.int64).to_numpy()

    c2 = ob["chrom"].astype(np.int64).to_numpy()
    s2b = ob["start"].astype(np.int64).to_numpy()
    e2b = ob["end"].astype(np.int64).to_numpy()
    bri2 = ob["build_region_index"].astype(np.int64).to_numpy()

    chroms = np.intersect1d(np.unique(c1), np.unique(c2))
    rows: list[dict] = []

    for chrom in chroms:
        idx_b = np.flatnonzero(c2 == chrom)
        if idx_b.size == 0:
            continue
        order = np.argsort(s2b[idx_b], kind="mergesort")
        idx_bs = idx_b[order]
        Sb = s2b[idx_bs]
        Eb = e2b[idx_bs]
        E2s = E2n[idx_bs]
        bri2s = bri2[idx_bs]

        idx_a = np.flatnonzero(c1 == chrom)
        for i in idx_a:
            s1, e1 = int(s1a[i]), int(e1a[i])
            L = s1 - Dmax
            R = e1 + Dmax
            j_end = int(np.searchsorted(Sb, R, side="right"))
            if j_end == 0:
                continue
            sel = np.nonzero(Eb[:j_end] >= L)[0]
            if sel.size == 0:
                continue

            Sbb = Sb[sel]
            Ebb = Eb[sel]
            ov = (np.maximum(s1, Sbb) <= np.minimum(e1, Ebb))
            not_ov = ~ov
            sep = np.zeros(sel.size, dtype=np.int64)
            sep[ov] = 0
            right = not_ov & (e1 < Sbb)
            left = not_ov & (Ebb < s1)
            sep[right] = Sbb[right] - e1
            sep[left] = s1 - Ebb[left]

            keep = (sep >= Dmin) & (sep <= Dmax)
            if not keep.any():
                continue

            cos_batch = (E2s[sel] @ E1n[i]).astype(np.float32)
            ch_out = oa["chrom"].iloc[int(i)]

            kept_idx = np.nonzero(keep)[0]
            for k in kept_idx:
                rows.append(
                    {
                        "set1_chrom": ch_out,
                        "set1_start": s1,
                        "set1_end": e1,
                        "set1_build_region_index": int(bri1[i]),
                        "set2_chrom": ch_out,
                        "set2_start": int(Sbb[k]),
                        "set2_end": int(Ebb[k]),
                        "set2_build_region_index": int(bri2s[sel[k]]),
                        "genomic_dist_bp": int(sep[k]),
                        "cos_sim": cos_batch[k],
                    }
                )

    return pd.DataFrame(rows)