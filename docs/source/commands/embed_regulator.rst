===============
embed_regulator
===============

Generate **768-dimensional regulator embeddings** for user-provided genomic regions.

This command can use either the pre-trained ChromBERT model or a cell-type-specific model.
For each requested regulator, it outputs both region-aware regulator embeddings and mean
regulator embeddings.

Overview
========

``embed_regulator`` requires two inputs:

* ``--region``: genomic regions of interest
* ``--regulator``: regulators of interest

For each regulator, ChromBERT-tools generates:

* **region-aware regulator embeddings** for all overlapping ChromBERT bins
* **mean regulator embeddings** averaged across the input regions

Modes
=====

General mode
------------

General mode is used when no cell-type-specific information is provided.

In this mode, ChromBERT-tools uses the pre-trained ChromBERT model to generate regulator
embeddings for the input regions.

Cell-type-specific mode
-----------------------

Cell-type-specific mode is used when either of the following is provided:

* ``--ft-ckpt``: a fine-tuned checkpoint
* both ``--cell-type-bw`` and ``--cell-type-peak``: cell-type accessibility signal and peaks

If ``--ft-ckpt`` is provided, ChromBERT-tools loads the checkpoint directly and skips
fine-tuning.

If ``--cell-type-bw`` and ``--cell-type-peak`` are provided without ``--ft-ckpt``,
ChromBERT-tools first fine-tunes a cell-type-specific model, then uses it to generate
regulator embeddings.

Basic Usage
===========

General mode
------------

.. code-block:: bash

   chrombert-tools embed_regulator \
     --region regions.bed \
     --regulator "CTCF;BRD4;MYC" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Cell-type-specific mode from accessibility data
-----------------------------------------------

.. code-block:: bash

   chrombert-tools embed_regulator \
     --region regions.bed \
     --regulator "CTCF;BRD4" \
     --cell-type-bw cell_accessibility.bigwig \
     --cell-type-peak cell_peaks.bed \
     --genome hg38 \
     --resolution 1kb \
     --mode fast \
     --odir output_cell_specific

Cell-type-specific mode from a checkpoint
-----------------------------------------

.. code-block:: bash

   chrombert-tools embed_regulator \
     --region regions.bed \
     --regulator "CTCF;BRD4" \
     --ft-ckpt path/to/finetuned.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output_from_ckpt

Run with Apptainer
------------------

Use ``--nv`` to enable GPU access:

.. code-block:: bash

   apptainer exec --nv /path/to/chrombert-tools.sif chrombert-tools embed_regulator \
     --region regions.bed \
     --regulator "CTCF;BRD4" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required parameters
-------------------

``--region`` *(file path, required)*
   Input genomic regions. The file should contain at least ``chrom``, ``start``, and
   ``end`` columns.

``--regulator`` *(string, required)*
   Regulators of interest, separated by semicolons. For example:

   ``"EZH2;BRD4;CTCF"``

   Regulator names are matched against the ChromBERT regulator list. Matching is
   case-insensitive.

Cell-type-specific options
--------------------------

``--cell-type-bw`` *(file path, optional)*
   Cell-type-specific accessibility signal in BigWig format. This option must be used
   together with ``--cell-type-peak`` unless ``--ft-ckpt`` is provided.

``--cell-type-peak`` *(file path, optional)*
   Cell-type-specific accessibility peaks in BED format. This option must be used
   together with ``--cell-type-bw`` unless ``--ft-ckpt`` is provided.

``--ft-ckpt`` *(file path, optional)*
   Fine-tuned checkpoint. When provided, ChromBERT-tools loads this checkpoint directly
   and does not perform fine-tuning.

``--mode`` *(fast | full, default: fast)*
   Fine-tuning mode. This option is only used when training a new cell-type-specific
   model from ``--cell-type-bw`` and ``--cell-type-peak``.

Reference and output options
----------------------------

``--genome`` *(hg38 | mm10, default: hg38)*
   Reference genome.

``--resolution`` *(1kb | 200bp | 2kb | 4kb, default: 1kb)*
   ChromBERT bin resolution. For ``mm10``, only ``1kb`` is currently supported.

``--batch-size`` *(int, default: 4)*
   Batch size used for model inference.

``--num-workers`` *(int, default: 8)*
   Number of dataloader workers.

``--odir`` *(directory, default: ./output)*
   Output directory. It will be created automatically if it does not exist.

``--oname`` *(string, default: regulator_emb)*
   Output file name prefix.

Cache option
------------

``--chrombert-cache-dir`` *(directory, default: ~/.cache/chrombert/data)*
   Directory containing ChromBERT reference files, regulator lists, model files, and
   cached data.

Output Files
============

The following files are written to ``--odir``.

``region_aware_<oname>.hdf5``
   Region-aware regulator embeddings.

   Each regulator is stored as one dataset under the ``emb/`` group. The dataset has
   shape ``(n_regions, 768)``, where ``n_regions`` is the number of input regions
   overlapping ChromBERT bins.

``mean_<oname>.pkl``
   A Python dictionary mapping each matched regulator to its 768-dimensional mean
   embedding.

``overlap_region.bed``
   Input regions that overlap ChromBERT reference bins.

``no_overlap_region.bed``
   Input regions that do not overlap ChromBERT reference bins.

``model_input.tsv``
   Processed input table used for model inference.

Load outputs in Python
======================

Load region-aware regulator embeddings
--------------------------------------

.. code-block:: python

   import h5py

   with h5py.File("output/region_aware_regulator_emb.hdf5", "r") as f:
       ctcf_emb = f["emb/ctcf"][:]
       brd4_emb = f["emb/brd4"][:]

Load mean regulator embeddings
------------------------------

.. code-block:: python

   import pickle

   with open("output/mean_regulator_emb.pkl", "rb") as f:
       mean_emb = pickle.load(f)

   ctcf_mean = mean_emb["ctcf"]

Tips
====

1. Regulator names are matched case-insensitively, but output keys are stored in lowercase.
2. If no requested regulator matches the ChromBERT regulator list, the command stops before
   model inference.
3. To generate cell-type-specific embeddings, provide either ``--ft-ckpt`` or both
   ``--cell-type-bw`` and ``--cell-type-peak``.
4. If you already have a fine-tuned checkpoint, use ``--ft-ckpt`` directly. BigWig and peak
   files are not required.
5. To see all available options, run:

.. code-block:: bash

   chrombert-tools embed_regulator -h

Tutorials
=========

.. toctree::
   :maxdepth: 1

   CLI example <../examples/cli/embed_regulator>
   Python API example <../examples/api/embed_regulator>