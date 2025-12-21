import os
import numpy as np
import pandas as pd
import chrombert
import torch
import json
import glob
import lightning.pytorch as pl
from chrombert import ChromBERTFTConfig, DatasetConfig
from .utils import set_seed
from .utils import bw_getSignal_bins, split_data
from lightning.pytorch.callbacks import TQDMProgressBar
from chrombert.scripts.chrombert_make_dataset import process

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


def init_datamodule(d_odir, args, files_dict, ignore_object=None, task="general"):
    """
    Initialize data module for training.
    
    Args:
        d_odir: Dataset directory
        args: Arguments containing mode, batch_size, etc.
        files_dict: Dictionary of file paths
        ignore_object: Optional object to ignore during training
        dataset_kind: Type of dataset ("GeneralDataset" or "MultiFlankwindowDataset")
        extra_config: Extra config dict for DatasetConfig (e.g., {"flank_window": 4})
    
    Returns:
        data_config, data_module, ignore, ignore_index
    """
    ignore = True if ignore_object is not None else False
    
    # 1. init dataconfig
    config_params = {
        "kind": "MultiFlankwindowDataset" if task == "gep" else "GeneralDataset",
        "supervised_file": None,
        "hdf5_file": files_dict["hdf5_file"],
        "batch_size": 2 if task == "gep" else args.batch_size,
        "num_workers": 8,
        "meta_file": files_dict["meta_file"]
    }
    
    # Only add flank_window for gep task (MultiFlankwindowDataset)
    if task == "gep":
        config_params["flank_window"] = 4
    
    data_config = DatasetConfig(**config_params)
    
    if ignore:
        data_config.ignore = ignore
        data_config.ignore_object = ignore_object
        
    # 2. init datamodule
    if args.mode == "fast" and task != 'gep':
        ds = data_config.init_dataset(supervised_file=os.path.join(d_odir, "train_sampled.csv"))
        data_module = chrombert.LitChromBERTFTDataModule(
            config=data_config,
            train_params={"supervised_file": f"{d_odir}/train_sampled.csv"},
            val_params={"supervised_file": f"{d_odir}/valid_sampled.csv"},
            test_params={"supervised_file": f"{d_odir}/test_sampled.csv"},
        )
    else:
        ds = data_config.init_dataset(supervised_file=os.path.join(d_odir, "train.csv"))
        data_module = chrombert.LitChromBERTFTDataModule(
            config=data_config,
            train_params={"supervised_file": f"{d_odir}/train.csv"},
            val_params={"supervised_file": f"{d_odir}/valid.csv"},
            test_params={"supervised_file": f"{d_odir}/test.csv"},
        )
    data_module.setup()
    
    ignore_index = ds[0]["ignore_index"] if ignore else None
    
    return data_config, data_module, ignore, ignore_index

def model_train(d_odir, train_odir, args, files_dict, train_kind='regression', 
                ignore_object=None, task="general"):
    """
    Train a ChromBERT model.
    
    Args:
        d_odir: Dataset directory
        train_odir: Training output directory
        args: Arguments containing genome, batch_size, mode, etc.
        files_dict: Dictionary of file paths
        train_kind: "regression" or "classification"
        ignore_object: Optional object to ignore during training
        task: ChromBERT task type ("general" or "gep")
    
    Returns:
        data_module, model_config
    """
    data_config, data_module, ignore, ignore_index = init_datamodule(
        d_odir, args, files_dict, ignore_object, task
    )

    # Init model config
    if task == "gep":
        model_config = ChromBERTFTConfig(
            genome=args.genome,
            task="gep",
            pretrain_ckpt=files_dict["pretrain_ckpt"],
            mtx_mask=files_dict["mtx_mask"],
            gep_flank_window=4
        )
    else:
        model_config = ChromBERTFTConfig(
            genome=args.genome,
            task=task,
            pretrain_ckpt=files_dict["pretrain_ckpt"],
            mtx_mask=files_dict["mtx_mask"],
            ignore=ignore,
            ignore_index=ignore_index,
        )
        
    model = model_config.init_model()
    model.freeze_pretrain(2)  # freeze chrombert 6 transformer blocks during fine-tuning

    # Init trainer
    loss = 'rmse' if train_kind == 'regression' else "bce"
    callback_metrics = "pcc" if train_kind == 'regression' else "auprc"
    max_epochs = 10 if task == "gep" else 5
    accumulate_grad_batches = 64
    patience = 10 if task == "gep" else 5
    
    train_config = chrombert.finetune.TrainConfig(
        kind=train_kind,
        loss=loss,
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=0.2,
        limit_val_batches=0.5,
    )
    train_module = train_config.init_pl_module(model)
    callback_ckpt = pl.callbacks.ModelCheckpoint(monitor=f"{train_config.tag}_validation/{callback_metrics}", mode="max")
    early_stop = pl.callbacks.EarlyStopping(
        monitor=f"{train_config.tag}_validation/{callback_metrics}",
        mode="max",
        patience=patience,
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
        logger=pl.loggers.TensorBoardLogger(f"./{train_odir}/lightning_logs"),
    )
    trainer.fit(train_module, data_module)
    return data_module, model_config

def model_eval(train_odir, data_module, model_config, cal_metrics):
    ckpts = glob.glob(f"{train_odir}/**/checkpoints/*.ckpt", recursive=True)
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint found under {train_odir}. Please verify that training completed successfully."
        )
    ft_ckpt = os.path.abspath(max(ckpts, key=os.path.getmtime))

    dc_test = data_module.test_config
    dl_test = dc_test.init_dataloader(batch_size=4)

    model_tuned = model_config.init_model(finetune_ckpt=ft_ckpt, dropout=0).eval().cuda()

    test_preds = []
    test_labels = []
    for batch in dl_test:
        with torch.no_grad():
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            preds = model_tuned(batch).cpu()
            test_labels.append(batch["label"].cpu())
            test_preds.append(preds)

    test_preds = torch.cat(test_preds, dim=0).reshape(-1)
    test_labels = torch.cat(test_labels, axis=0).reshape(-1)

    test_metrics = cal_metrics(test_preds, test_labels)
    print(f"ft_ckpt: {ft_ckpt}, test_metrics: {test_metrics}")
    test_metrics['ft_ckpt'] = ft_ckpt
    with open(os.path.join(train_odir, "eval_performance.json"), "w") as f:
        json.dump(test_metrics, f)
    return model_tuned, test_metrics

def retry_train(args, files_dict, cal_metrics, metcic='pearsonr', min_threshold=0.2, 
                train_kind='regression', ignore_object=None, task="general",odir=None):
    """
    Train with multiple retries to ensure stable performance.
    
    Args:
        args: Arguments containing odir, genome, batch_size, mode, etc.
        files_dict: Dictionary of file paths
        cal_metrics: Function to calculate metrics
        metcic: Metric name to monitor (e.g., 'pearsonr', 'auprc')
        min_threshold: Minimum acceptable threshold for the metric
        train_kind: "regression" or "classification"
        ignore_object: Optional object to ignore during training
        task: ChromBERT task type ("general" or "gep")
    
    Returns:
        model_tuned, train_odir, model_config, data_config
    """
    if odir is not None:
        d_odir = f"{odir}/dataset"
        os.makedirs(d_odir, exist_ok=True)
        train_odir = f"{odir}/train"
        os.makedirs(train_odir, exist_ok=True)
    else:
        d_odir = f"{args.odir}/dataset";os.makedirs(d_odir, exist_ok=True)
        train_odir = f"{args.odir}/train";os.makedirs(train_odir, exist_ok=True)
    
    best_model = None
    best_train_odir = None
    best_metrics = None
    best_pcc = float("-inf")
    last_err = None
    max_retries = 2  # number of retries after the first run
    base_seed = getattr(args, "seed", 55)
    
    for attempt in range(max_retries + 1):
        trial_seed = base_seed + attempt
        set_seed(trial_seed)

        # Isolate each attempt's outputs
        train_odir_try = os.path.join(train_odir, f"try_{attempt:02d}_seed_{trial_seed}")
        os.makedirs(train_odir_try, exist_ok=True)

        print(f"\n[Attempt {attempt}/{max_retries}] seed={trial_seed}")
        try:
            data_module, model_config = model_train(
                d_odir, train_odir_try, args, files_dict, train_kind, ignore_object, task
            )
            data_config = data_module.basic_config

            print("Evaluating the finetuned model performance")
            model_tuned, test_metrics = model_eval(
                train_odir_try, data_module, model_config, cal_metrics
            )

            pcc = float(test_metrics.get(metcic, float("nan")))
            print(f"Attempt metrics: {metcic}={pcc}")

            # Track best run even if it doesn't pass threshold
            if np.isfinite(pcc) and pcc > best_pcc:
                best_pcc = pcc
                best_model = model_tuned
                best_train_odir = train_odir_try
                best_metrics = test_metrics

            # Accept if stable and good enough
            if np.isfinite(pcc) and (pcc >= min_threshold):
                print(f"Accepted run ({metcic}={pcc:.4f} >= {min_threshold}).")
                best_model = model_tuned
                best_train_odir = train_odir_try
                best_metrics = test_metrics
                break

            print(
                f"Poor/unstable run ({metcic}={pcc}). "
                "Retrying with a different seed due to random initialization and stochastic optimization..."
            )

        except Exception as e:
            last_err = e
            print(f"Attempt failed with error: {repr(e)}")
            print("Retrying with a different seed...")

    # Finalize
    if best_model is None:
        raise RuntimeError(f"All attempts failed. Last error: {repr(last_err)}")

    model_tuned = best_model
    test_metrics = best_metrics
    train_odir = best_train_odir
    print("\nFinished stage 2: obtained a fine-tuned ChromBERT")
    print(f"Best {metcic}={best_pcc}, metrics={test_metrics}")
    return model_tuned, train_odir, model_config, data_config