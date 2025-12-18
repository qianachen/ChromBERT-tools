import numpy as np
import pandas as pd
import chrombert
import torchmetrics as tm
import lightning.pytorch as pl
from chrombert import ChromBERTFTConfig, DatasetConfig
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