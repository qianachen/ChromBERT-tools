===================================
predict_cell_type_master_regulators
===================================

Rank candidate master regulators for a target cell type.

This command uses cell-type chromatin accessibility data to identify regulators whose
embeddings differ most between highly accessible regions and background regions.

Top-ranked regulators are candidate cell-type-specific master regulators.

Overview
========

``predict_cell_type_master_regulators`` runs an end-to-end workflow:

1. define highly accessible regions and background regions from the input accessibility data
2. fine-tune a cell-type-specific ChromBERT model, or load an existing checkpoint
3. compare regulator embeddings between the two region groups
4. rank regulators by embedding shift

The main result is written to:

.. code-block:: text

   <odir>/results/factor_importance_rank.csv

Required inputs
===============

You must provide both:

* ``--cell-type-bw``: chromatin accessibility signal for the target cell type
* ``--cell-type-peak``: accessible peaks for the target cell type

These two files are required even when ``--ft-ckpt`` is provided, because they are used to
define the highly accessible and background regions.

Basic Usage
===========

Fine-tune from accessibility data
---------------------------------

.. code-block:: bash

   chrombert-tools predict_cell_type_master_regulators \
     --cell-type-bw cell.bw \
     --cell-type-peak cell_peaks.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Reuse an existing fine-tuned checkpoint
---------------------------------------

Use ``--ft-ckpt`` to skip fine-tuning.

.. code-block:: bash

   chrombert-tools predict_cell_type_master_regulators \
     --cell-type-bw cell.bw \
     --cell-type-peak cell_peaks.bed \
     --ft-ckpt path/to/cell_finetuned.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Run with Apptainer
------------------

Use ``--nv`` to enable GPU access.

.. code-block:: bash

   apptainer exec --nv /path/to/chrombert-tools.sif chrombert-tools predict_cell_type_master_regulators \
     --cell-type-bw cell.bw \
     --cell-type-peak cell_peaks.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Cell-type inputs
----------------

``--cell-type-bw`` *(file path, required)*
   BigWig file containing chromatin accessibility signal for the target cell type.

``--cell-type-peak`` *(file path, required)*
   BED file containing accessible peaks for the target cell type.

``--ft-ckpt`` *(file path, optional)*
   Fine-tuned ChromBERT checkpoint.

   If provided, ChromBERT-tools loads this checkpoint and skips fine-tuning.

Run options
-----------

``--mode`` *(fast | full, default: fast)*
   Fine-tuning mode.

   ``fast`` uses fewer regions for quicker training.

   ``full`` uses all eligible regions and may give more stable results.

``--batch-size`` *(int, default: 4)*
   Batch size used for fine-tuning and embedding generation.

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

``--model-config`` *(file path, optional)*
   Custom model configuration file.

``--region-config`` *(file path, optional)*
   Custom region configuration file.

Required cache files
====================

The command uses the following ChromBERT cache files:

* ChromBERT reference region file
* ChromBERT HDF5 feature file
* pre-trained ChromBERT checkpoint
* mask matrix

Outputs
=======

The following files are written under ``<odir>``.

``dataset/highly_accessible_region.csv``
   Regions with high accessibility signal in the target cell type.

``dataset/background_region.csv``
   Background regions used for comparison.

``train/``
   Fine-tuning outputs and checkpoints.

   This directory is created only when ``--ft-ckpt`` is not provided.

``emb/mean_regulator_emb_region1.pkl``
   Mean regulator embeddings for highly accessible regions.

``emb/mean_regulator_emb_region2.pkl``
   Mean regulator embeddings for background regions.

``results/factor_importance_rank.csv``
   Main output file.

   Each row is one regulator. Regulators are sorted by embedding shift.

   Main columns:

   * ``factors``: regulator name
   * ``similarity``: cosine similarity between the regulator embeddings from highly
     accessible and background regions
   * ``embedding_shift``: ``1 - similarity``
   * ``rank``: regulator rank; 1 means the largest embedding shift

Interpretation
==============

Regulators with larger ``embedding_shift`` values show stronger differences between
highly accessible regions and background regions.

These top-ranked regulators are candidate master regulators for the target cell type.

Tips
====

1. Use ``--ft-ckpt`` to reuse an existing fine-tuned checkpoint and skip fine-tuning.
2. Use ``--mode fast`` for quick runs and ``--mode full`` for more stable rankings.
3. The two mean regulator embedding files can be reused for downstream analyses.
4. To see all options, run:

.. code-block:: bash

   chrombert-tools predict_cell_type_master_regulators -h

Tutorials
=========

.. toctree::
   :maxdepth: 1

   CLI example <../examples/cli/predict_cell_type_master_regulators>