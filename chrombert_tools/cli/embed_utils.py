import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from chrombert_hf import ChromBERTFTConfig, DatasetConfig, ChromBERTConfig
from transformers import AutoModel
from .utils import (
    get_model_name,
    cal_metrics_regression,
    HDF5Manager
)
from .utils_train_cell import make_dataset, retry_train



def is_cell_specific(args):
    '''
    Check if the embedding is cell-specific
    '''
    return (
        args.ft_ckpt is not None
        or (args.cell_type_bw is not None and args.cell_type_peak is not None)
    )



def get_required_keys(args):
    """
    Get required cache keys for current mode.
    """
    required = [
        "chrombert_region_file",
        "chrombert_regulator_file",
        "hdf5_file",
    ]
    return required

def build_dataloader(supervised_file, hdf5_file, batch_size, num_workers=8):
    '''
    Build dataloader
    '''
    data_config = DatasetConfig(
        kind="GeneralDataset",
        supervised_file=supervised_file,
        hdf5_file=hdf5_file,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dl = data_config.init_dataloader()
    ds = data_config.init_dataset()
    return ds, dl


def build_model_emb(args,files_dict):
    '''
    Build pretrained model and embedding manager
    '''
    lite = getattr(args, "lite", False)
    if files_dict["pretrain_ckpt"] is not None and os.path.exists(files_dict["pretrain_ckpt"]) and os.path.exists(files_dict["mtx_mask"]):
        model_config = ChromBERTConfig(
                ckpt=files_dict["pretrain_ckpt"],
                genome=args.genome,
                mask_matrix=files_dict["mtx_mask"],
                dropout=0,
                lite=lite,
            )
        model_emb = model_config.init_model().get_embedding_manager().cuda().bfloat16()
    else:
        model_emb = AutoModel.from_pretrained(get_model_name(args.genome, args.resolution, lite), trust_remote_code=True).get_embedding_manager().cuda().bfloat16()
    # model_emb = AutoModel.from_pretrained(get_model_name(args.genome, args.resolution), trust_remote_code=True).get_embedding_manager().cuda().bfloat16()
    
    return model_emb


def build_cell_model_emb(args, files_dict,odir):
    '''
    Build cell-specific model and embedding manager
    '''
    lite = getattr(args, "lite", False)
    # 1) load ft ckpt if provided
    if args.ft_ckpt is not None:
        print(f"Using provided fine-tuned checkpoint: {args.ft_ckpt}")
        model_config = ChromBERTFTConfig(
            pretrained_model_name_or_path = get_model_name(args.genome, args.resolution, lite),
            finetune_ckpt=args.ft_ckpt,
            pretrain_ckpt=files_dict["pretrain_ckpt"],
            mtx_mask=files_dict["mtx_mask"],
            dropout=0,
            lite=lite,
        )
        model_emb = model_config.init_model().get_embedding_manager().cuda().bfloat16()
        return model_emb

    # 2) no ft ckpt, train cell-specific model on the fly
    d_odir = f"{odir}/dataset"
    os.makedirs(d_odir, exist_ok=True)
    train_odir = f"{odir}/train"
    os.makedirs(train_odir, exist_ok=True)

    print("Preparing dataset for cell-specific model...")
    make_dataset(args.cell_type_peak, args.cell_type_bw, d_odir, files_dict, args.mode)

    print("Fine-tuning cell-specific model...")
    model_tuned, train_odir, model_config, data_config = retry_train(
        args,
        files_dict,
        cal_metrics_regression,
        metcic="pearsonr",
        min_threshold=0.4,
    )
    model_emb = model_tuned.get_embedding_manager().cuda().bfloat16()
    return model_emb


def generate_embeddings(dl, model_emb):
    '''
    Generate embeddings from chrombert (forward pass)
    '''
    region_embs = []
    with torch.no_grad():
        for batch in tqdm(dl, total=len(dl)):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            model_emb(batch)
            region_embs.append(model_emb.get_region_embedding().float().cpu().detach())

    return torch.cat(region_embs, dim=0).numpy()


def generate_regulator_embeddings(ds, dl, model_emb, overlap_bed, regulator_idx_dict, odir, oname, return_data):
    '''
    Generate regulator embeddings from chrombert (forward pass)
    '''
    shapes = {f"emb/{k}": [(len(ds), 768), np.float16] for k in regulator_idx_dict}
    reg_sums = {name: np.zeros(768, dtype=np.float64) for name in regulator_idx_dict}
    reg_emb_dict = {}
    total_counts = 0

    out_h5 = f"{odir}/region_aware_{oname}.hdf5"
    with HDF5Manager(out_h5, region=[(len(ds), 4), np.int64], **shapes) as h5:
        with torch.no_grad():
            for batch in tqdm(dl, total=len(dl), desc="Computing regulator embeddings"):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()

                model_emb(batch)

                bs = batch["region"].shape[0]
                start_idx = total_counts
                total_counts += bs
                end_idx = total_counts

                batch_index = batch["build_region_index"].long().cpu().numpy().reshape(-1)
                region = overlap_bed.iloc[start_idx:end_idx].values # the region of users' interest
                assert (batch_index == region[:, -1].reshape(-1)).all(), "Batch index and region index do not match"

                embs = {
                    f"emb/{k}": model_emb.get_regulator_embedding(k).float().cpu().numpy()
                    for k in regulator_idx_dict
                }
                h5.insert(region=region, **embs)

                for reg_name in regulator_idx_dict:
                    emb_np = model_emb.get_regulator_embedding(reg_name).float().cpu().numpy()
                    reg_sums[reg_name] += emb_np.sum(axis=0)
                    if return_data:
                        if reg_name not in reg_emb_dict:
                            reg_emb_dict[reg_name] = []
                        reg_emb_dict[reg_name].append(emb_np)

    if return_data:
        for key in reg_emb_dict:
            reg_emb_dict[key] = np.concatenate(reg_emb_dict[key], axis=0)
    else:
        reg_emb_dict = None

    reg_means = {
        reg_name: (sum_vec / total_counts)
        for reg_name, sum_vec in reg_sums.items()
    }
    out_pkl = os.path.join(odir, f"mean_{oname}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(reg_means, f)
    return reg_means, reg_emb_dict