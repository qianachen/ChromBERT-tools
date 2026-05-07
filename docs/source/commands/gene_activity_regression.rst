========================
gene_activity_regression
========================

Fine-tune a ChromBERT gene activity regression model using TPM expression tables, or use an existing checkpoint to predict gene activity from TSS-centered representations that include the promoter region and upstream/downstream flanking regions.

The command supports two prediction targets:

* **single-state gene activity**: ``log1p(TPM)``
* **two-state gene activity change**: ``log1p(TPM2) - log1p(TPM1)``

Overview
========

``gene_activity_regression`` can be used in two ways:

* **training + prediction**: train a regression model from TPM tables, then predict gene
  activity
* **predict-only**: load an existing fine-tuned checkpoint and predict directly

During training, ChromBERT-tools prepares the expression dataset, trains or loads a
regression model, and writes predictions to ``<odir>/predict/predictions.csv``.

Input expression tables
=======================

Each TPM file should contain at least two columns:

* ``gene_id``
* ``tpm``

Column names are matched case-insensitively.

Single-state mode
-----------------

Use only ``--exp-tpm1``.

The prediction target is:

``log1p(mean TPM)``

If multiple TPM files are provided, they are treated as replicates and averaged by gene.

Two-state mode
--------------

Use both ``--exp-tpm1`` and ``--exp-tpm2``.

The prediction target is:

``log1p(TPM2) - log1p(TPM1)``

Use ``--direction`` to control the direction of the fold change:

* ``2-1``: keep ``log1p(TPM2) - log1p(TPM1)``
* ``1-2``: use ``log1p(TPM1) - log1p(TPM2)``

Replicates
----------

Multiple TPM files can be provided with semicolons:

.. code-block:: bash

   --exp-tpm1 "rep1.csv;rep2.csv;rep3.csv"

Within each state, ChromBERT-tools keeps genes present in all replicate files and averages
their TPM values.

Basic Usage
===========

Train from one state
--------------------

.. code-block:: bash

   chrombert-tools gene_activity_regression \
     --exp-tpm1 state1_tpm.csv \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Train from two states
---------------------

.. code-block:: bash

   chrombert-tools gene_activity_regression \
     --exp-tpm1 state1_tpm.csv \
     --exp-tpm2 state2_tpm.csv \
     --direction 2-1 \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Train with replicates
---------------------

.. code-block:: bash

   chrombert-tools gene_activity_regression \
     --exp-tpm1 "rep1.csv;rep2.csv;rep3.csv" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Predict only
------------

Use this mode when you already have a fine-tuned gene activity regression checkpoint.

.. code-block:: bash

   chrombert-tools gene_activity_regression \
     --ft-ckpt path/to/gep_finetuned.ckpt \
     --predict-file regions_genes.tsv \
     --genome hg38 \
     --resolution 1kb \
     --odir output_predict

Run with Apptainer
------------------

Use ``--nv`` to enable GPU access.

.. code-block:: bash

   apptainer exec --nv /path/to/chrombert-tools.sif chrombert-tools gene_activity_regression \
     --exp-tpm1 state1_tpm.csv \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Expression inputs
-----------------

``--exp-tpm1`` *(file path or semicolon-separated paths)*
   State-1 TPM table(s). Required for training.

``--exp-tpm2`` *(file path or semicolon-separated paths, optional)*
   State-2 TPM table(s). Provide this option for two-state gene activity change
   prediction.

``--direction`` *(2-1 | 1-2, default: 2-1)*
   Direction of the two-state target.

   ``2-1`` means ``log1p(TPM2) - log1p(TPM1)``.

   ``1-2`` means ``log1p(TPM1) - log1p(TPM2)``.

Prediction inputs
-----------------

``--predict-file`` *(file path, optional)*
   Regions or genes used for prediction.

   The file should contain the following columns:

   * ``chrom``
   * ``start``
   * ``end``
   * ``build_region_index``
   * ``gene_id``
   * ``tss``

   If this option is not provided after training, ChromBERT-tools predicts on the test
   split generated during dataset preparation.

``--ft-ckpt`` *(file path, optional)*
   Fine-tuned gene activity regression checkpoint.

   When both ``--ft-ckpt`` and ``--predict-file`` are provided, ChromBERT-tools runs in
   predict-only mode and skips training.

Model options
-------------

``--flank-window`` *(int, default: 4)*
   Number of flanking ChromBERT windows used for gene activity prediction.

``--batch-size`` *(int, default: 4)*
   Batch size for training and prediction.

``--mode`` *(fast | full, default: fast)*
   Reserved for consistency with other ChromBERT-tools commands. This command always uses
   the prepared train, test, and validation splits.

Reference and output options
----------------------------

``--genome`` *(hg38 | mm10, default: hg38)*
   Reference genome.

``--resolution`` *(200bp | 1kb | 2kb | 4kb, default: 1kb)*
   ChromBERT bin resolution. For ``mm10``, only ``1kb`` is currently supported.

``--odir`` *(directory, default: ./output)*
   Output directory. It will be created automatically if needed.

``--chrombert-cache-dir`` *(directory, default: ~/.cache/chrombert/data)*
   Directory for ChromBERT reference files, model files, and cached data.

Required cache files
====================

The command uses the following ChromBERT cache files:

* ChromBERT reference region file
* ChromBERT HDF5 feature file
* metadata file
* gene metadata file

The gene metadata file is required for training and test-split prediction. It is not
required in predict-only mode when both ``--ft-ckpt`` and ``--predict-file`` are provided.

Outputs
=======

The following files are written under ``--odir``.

``dataset/``
   Created during training. It contains the processed expression dataset and train, test,
   and validation splits.

``train/``
   Created during training. It contains model training outputs and the selected checkpoint.

``predict/model_input.tsv``
   Processed input table used for prediction.

``predict/predictions.csv``
   Main prediction output.

   The file contains gene and region metadata, predicted values, and optionally true labels
   if labels are available in the prediction input.

``model_config.json``
   Model configuration used for the run.

``dataset_config.json``
   Dataset configuration used for the run.

Predict-only mode
=================

Predict-only mode is used when both of the following are provided:

* ``--ft-ckpt``
* ``--predict-file``

In this mode, ChromBERT-tools does not require TPM tables and does not train a model. It
loads the checkpoint and writes predictions directly to ``<odir>/predict/predictions.csv``.

Tips
====

1. Use ``--exp-tpm1`` for single-state gene activity prediction.
2. Use both ``--exp-tpm1`` and ``--exp-tpm2`` for two-state gene activity change
   prediction.
3. Use semicolons to provide replicate TPM files.
4. Use ``--ft-ckpt`` together with ``--predict-file`` for predict-only mode.
5. ``--mode`` is kept for consistency with other commands but does not change the training
   behavior of this command.
6. To see all options, run:

.. code-block:: bash

   chrombert-tools gene_activity_regression -h

Tutorials
=========

.. toctree::
   :maxdepth: 1

   Python API example <../examples/api/gene_activity_regression>