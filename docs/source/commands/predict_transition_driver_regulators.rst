====================================
predict_transition_driver_regulators
====================================

Rank candidate driver regulators for a cell-state transition.

This command identifies regulators whose embeddings differ most between strongly changed
regions or genes and near-zero-change background regions or genes.

It supports two data modalities:

* **chromatin accessibility change**
* **gene expression change**

When both modalities are provided, ChromBERT-tools also generates an integrated regulator
ranking.

Overview
========

``predict_transition_driver_regulators`` runs an end-to-end workflow:

1. build transition datasets from accessibility and/or expression data
2. fine-tune ChromBERT to predict state changes, or load existing checkpoints
3. define strongly changed and near-zero-change groups
4. compare regulator embeddings between these groups
5. rank regulators by embedding shift

The main results are written to:

.. code-block:: text

   <odir>/acc/results/factor_importance_rank.csv
   <odir>/exp/results/factor_importance_rank.csv
   <odir>/merge/factor_importance_rank.csv

The merged result is generated only when both accessibility and expression branches are
run.

Input modalities
================

Accessibility branch
--------------------

Use this branch to identify drivers of chromatin accessibility changes.

Required inputs:

* ``--acc-peak1``
* ``--acc-signal1``
* ``--acc-signal2``

``--acc-peak2`` is optional.

The accessibility change is calculated as:

.. code-block:: text

   log2(1 + state2 signal) - log2(1 + state1 signal)

Expression branch
-----------------

Use this branch to identify drivers of gene expression changes.

Required inputs:

* ``--exp-tpm1``
* ``--exp-tpm2``

The expression change is calculated as:

.. code-block:: text

   log1p(TPM2) - log1p(TPM1)

Direction
---------

Use ``--direction`` to control the direction of change:

* ``2-1``: state 2 minus state 1
* ``1-2``: state 1 minus state 2

At least one modality must be provided.

Basic Usage
===========

Accessibility-only transition
-----------------------------

.. code-block:: bash

   chrombert-tools predict_transition_driver_regulators \
     --acc-peak1 state1_peaks.bed \
     --acc-signal1 state1.bw \
     --acc-signal2 state2.bw \
     --direction 2-1 \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Expression-only transition
--------------------------

.. code-block:: bash

   chrombert-tools predict_transition_driver_regulators \
     --exp-tpm1 state1_tpm.csv \
     --exp-tpm2 state2_tpm.csv \
     --direction 2-1 \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use both modalities
-------------------

.. code-block:: bash

   chrombert-tools predict_transition_driver_regulators \
     --acc-peak1 state1_peaks.bed \
     --acc-signal1 state1.bw \
     --acc-signal2 state2.bw \
     --exp-tpm1 state1_tpm.csv \
     --exp-tpm2 state2_tpm.csv \
     --direction 2-1 \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Reuse existing checkpoints
--------------------------

Use existing checkpoints to skip fine-tuning.

.. code-block:: bash

   chrombert-tools predict_transition_driver_regulators \
     --acc-peak1 state1_peaks.bed \
     --acc-signal1 state1.bw \
     --acc-signal2 state2.bw \
     --exp-tpm1 state1_tpm.csv \
     --exp-tpm2 state2_tpm.csv \
     --ft-ckpt-acc path/to/acc_ft.ckpt \
     --ft-ckpt-exp path/to/exp_ft.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Run with Apptainer
------------------

Use ``--nv`` to enable GPU access.

.. code-block:: bash

   apptainer exec --nv /path/to/chrombert-tools.sif chrombert-tools predict_transition_driver_regulators \
     --acc-peak1 state1_peaks.bed \
     --acc-signal1 state1.bw \
     --acc-signal2 state2.bw \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Accessibility inputs
--------------------

``--acc-peak1`` *(file path or semicolon-separated paths)*
   State-1 peak BED file(s). Required for the accessibility branch.

``--acc-peak2`` *(file path or semicolon-separated paths, optional)*
   State-2 peak BED file(s). If omitted, state-1 peaks are used to define the candidate
   regions.

``--acc-signal1`` *(file path or semicolon-separated paths)*
   State-1 accessibility BigWig file(s). Required for the accessibility branch.

``--acc-signal2`` *(file path or semicolon-separated paths)*
   State-2 accessibility BigWig file(s). Required for the accessibility branch.

Expression inputs
-----------------

``--exp-tpm1`` *(file path)*
   State-1 TPM table. The file should contain ``gene_id`` and ``tpm`` columns.

``--exp-tpm2`` *(file path)*
   State-2 TPM table. The file should contain ``gene_id`` and ``tpm`` columns.

Both ``--exp-tpm1`` and ``--exp-tpm2`` are required for the expression branch.

Direction and background options
--------------------------------

``--direction`` *(2-1 | 1-2, default: 2-1)*
   Direction of the transition change.

   ``2-1`` means state 2 minus state 1.

   ``1-2`` means state 1 minus state 2.

``--include-tss-background`` *(true | false, default: true)*
   Whether to add TSS-centered background regions to the accessibility branch.

``--tss-flank`` *(int, default: 10000)*
   Flanking distance in base pairs around each TSS when TSS background is enabled.

Checkpoint options
------------------

``--ft-ckpt-acc`` *(file path, optional)*
   Fine-tuned checkpoint for the accessibility branch.

   If provided, ChromBERT-tools skips accessibility model fine-tuning.

``--ft-ckpt-exp`` *(file path, optional)*
   Fine-tuned checkpoint for the expression branch.

   If provided, ChromBERT-tools skips expression model fine-tuning.

Training options
----------------

``--mode`` *(fast | full, default: fast)*
   Training mode.

   ``fast`` uses fewer regions for quicker training.

   ``full`` uses all available rows.

``--flank-window`` *(int, default: 4)*
   Multi-flank window size for the expression model.

``--batch-size`` *(int, default: 4)*
   Batch size used for training and embedding generation.

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
* ChromBERT gene metadata file
* metadata file
* pre-trained ChromBERT checkpoint
* mask matrix

Outputs
=======

The following files are written under ``<odir>``.

Accessibility branch
--------------------

Created when accessibility inputs are provided.

``acc/dataset/``
   Processed accessibility transition dataset, including strongly changed and
   near-zero-change region groups.

``acc/train/``
   Accessibility model fine-tuning outputs and checkpoints.

   This directory is skipped when ``--ft-ckpt-acc`` is provided.

``acc/emb/``
   Mean regulator embeddings for accessibility-based region groups.

``acc/results/factor_importance_rank.csv``
   Accessibility-based driver regulator ranking.

Expression branch
-----------------

Created when expression inputs are provided.

``exp/dataset/``
   Processed expression transition dataset, including strongly changed and
   near-zero-change gene groups.

``exp/train/``
   Expression model fine-tuning outputs and checkpoints.

   This directory is skipped when ``--ft-ckpt-exp`` is provided.

``exp/emb/``
   Mean regulator embeddings for expression-based gene groups.

``exp/results/factor_importance_rank.csv``
   Expression-based driver regulator ranking.

Integrated ranking
------------------

Created only when both branches are completed.

``merge/factor_importance_rank.csv``
   Integrated driver regulator ranking.

   This file combines the accessibility and expression rankings by regulator name and
   ranks regulators using the average rank across modalities.

Main ranking columns
--------------------

The modality-specific ranking files include:

* ``factors``: regulator name
* ``similarity``: cosine similarity between embeddings from strongly changed and
  near-zero-change groups
* ``embedding_shift``: ``1 - similarity``
* ``rank``: regulator rank; 1 means the largest embedding shift

The integrated ranking includes modality-specific ranks and a combined ``total_rank``.

Interpretation
==============

Regulators with larger ``embedding_shift`` values show stronger differences between
strongly changed and near-zero-change regions or genes.

Top-ranked regulators are candidate drivers of the cell-state transition.

When both modalities are used, regulators with high ranks in both accessibility and
expression analyses provide stronger evidence as transition drivers.

Tips
====

1. Use accessibility inputs to identify drivers of chromatin accessibility changes.
2. Use expression inputs to identify drivers of gene expression changes.
3. Provide both modalities to generate an integrated ranking.
4. Use ``--ft-ckpt-acc`` or ``--ft-ckpt-exp`` to reuse checkpoints and skip fine-tuning.
5. Use ``--include-tss-background false`` to disable TSS background regions in the
   accessibility branch.
6. To see all options, run:

.. code-block:: bash

   chrombert-tools predict_transition_driver_regulators -h

Tutorials
=========

.. toctree::
   :maxdepth: 1

   CLI example <../examples/cli/predict_transition_driver_regulators>