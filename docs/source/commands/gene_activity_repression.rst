==========================
gene_activity_repression
==========================

Fine-tune or load a ChromBERT **GEP** (gene expression) regression model from TPM tables, then predict **log1p TPM** (single state) or **log1p fold change** (two states) at gene-linked ChromBERT windows.

Overview
========

The command runs three **stages** (see ``run`` in ``gene_activity_repression.py``):

1. **Stage 1 — Expression dataset** (skipped in predict-only mode): read TPM CSVs, merge with protein-coding **gene meta** (TSS / ``build_region_index``), build ``total.csv`` and **train / test / valid** splits under ``<odir>/dataset/``.
2. **Stage 2 — Model**: either **load** ``--ft-ckpt`` or **train** via ``retry_train`` (regression, metric Pearson, task ``gep``).
3. **Stage 3 — Predict**: run the model on ``--predict-file`` or, if omitted, on ``<odir>/dataset/test.csv``; write ``<odir>/predict/predictions.csv``.

**Predict-only mode** (``_is_predict_only``): both ``--ft-ckpt`` and ``--predict-file`` are set. Skips TPM dataset preparation and training; loads the GEP checkpoint with ``_load_model_for_predict_gep`` and runs **Stage 3** only. ``gene_meta_tsv`` is **not** in the required file list in that branch.

**Single-state vs two-state**

* **Single-state:** only ``--exp-tpm1`` (one or more paths, ``;``-separated). Label = **log1p(mean TPM)** per gene after replicate merging.
* **Two-state:** ``--exp-tpm1`` and ``--exp-tpm2``. Genes are **inner-joined** across states; label = ``log1p(TPM2) - log1p(TPM1)``, optionally flipped by ``--direction`` (``2-1`` vs ``1-2``).

Replicate CSVs per state: each file must have columns ``gene_id`` and ``tpm`` (case-insensitive headers). Multiple paths use **inner join** on ``gene_id`` across files, then **mean TPM** (``mean_tpm_by_gene``).

Basic Usage
===========

Train from one state (log1p TPM as target)
------------------------------------------

.. code-block:: bash

   chrombert-tools gene_activity_repression \
     --exp-tpm1 state1_tpm.csv \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Train from two states (fold change)
-----------------------------------

.. code-block:: bash

   chrombert-tools gene_activity_repression \
     --exp-tpm1 s1.csv \
     --exp-tpm2 s2.csv \
     --direction 2-1 \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Replicates (inner-join genes across files, then mean TPM)
---------------------------------------------------------

.. code-block:: bash

   chrombert-tools gene_activity_repression \
     --exp-tpm1 "rep1.csv;rep2.csv;rep3.csv" \
     --genome hg38 \
     --odir output

Predict-only (no TPM, no training)
----------------------------------

.. code-block:: bash

   chrombert-tools gene_activity_repression \
     --ft-ckpt path/to/gep_finetuned.ckpt \
     --predict-file regions_genes.tsv \
     --genome hg38 \
     --resolution 1kb \
     --odir output_predict

Parameters
==========

``--exp-tpm1``, ``--exp-tpm2``
   TPM CSV path(s). Use ``;`` for multiple replicates per state. Required for training unless predict-only.

``--direction``
   Only two-state: ``2-1`` (default) or ``1-2`` flips the sign of the fold-change label.

``--predict-file``
   Table with ``chrom``, ``start``, ``end``, ``build_region_index``, ``gene_id``, ``tss``; optional ``label``. Passed through ``check_region_file`` into ``<odir>/predict/model_input.tsv``. With ``--ft-ckpt``, enables predict-only. If omitted after training, predictions use ``<odir>/dataset/test.csv``.

``--ft-ckpt``
   Fine-tuned GEP checkpoint: **skips** ``retry_train`` when training from TPM is otherwise run; together with ``--predict-file`` selects predict-only.

``--flank-window``
   GEP multi-flank window size (default ``4``); forwarded to ``DatasetConfig`` / ``ChromBERTFTConfig.gep_flank_window``.

``--mode``
   Documented in Click as **reserved for parity**; expression splits always use full ``train.csv`` / ``test.csv`` / ``valid.csv`` (no separate “fast sampled” split logic in this module).

``--genome``, ``--resolution``, ``--batch-size``, ``--chrombert-cache-dir``, ``--odir``
   Standard ChromBERT options.

Required cache files (training path)
------------------------------------

``check_files`` requires ``chrombert_region_file``, ``hdf5_file``, ``meta_file``, and **``gene_meta_tsv``** unless predict-only. Pretrain checkpoint and mask paths come from ``resolve_paths`` for model initialization.

Outputs
=======

Under ``--odir``:

* **``dataset/``** — ``total.csv``, ``train.csv``, ``test.csv``, ``valid.csv``; if two-state, also ``up.csv`` / ``nochange.csv`` subsamples; ``dataset_meta.json`` with ``dual_state``.
* **``train/``** — training attempts when ``retry_train`` runs (best checkpoint path recorded in ``eval_performance.json`` under a ``try_*_seed_*`` subdirectory).
* **``predict/predictions.csv``** — columns from the input meta plus ``predicted_value`` and ``true_label`` when labels were present in the batch.
* **``model_config.json``**, **``dataset_config.json``** — serialized configs after the run.

The CLI prints the checkpoint path used and ``Predictions: <odir>/predict/predictions.csv``.

Tips
====

1. If ``<odir>/dataset/total.csv`` already exists, Stage 1 short-circuits (see ``make_exp_dataset``) and ``dual_state`` may be read from ``dataset_meta.json``.
2. For full option text, run ``chrombert-tools gene_activity_repression -h``.
