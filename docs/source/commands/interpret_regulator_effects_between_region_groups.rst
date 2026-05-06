=================================================
interpret_regulator_effects_between_region_groups
=================================================

Rank regulators by comparing their embeddings between two genomic region sets.

For each regulator, ChromBERT-tools calculates its mean embedding in region set 1 and
region set 2, then compares the two embeddings using cosine similarity.

A larger embedding shift suggests that the regulator has more different regulatory
contexts between the two region sets.

Overview
========

``interpret_regulator_effects_between_region_groups`` helps identify regulators whose
regulatory roles may differ between two sets of regions.

The command:

1. overlaps each input region set with ChromBERT bins
2. generates regulator embeddings for each region set
3. averages embeddings for each regulator within each set
4. compares the two mean embeddings for each regulator
5. ranks regulators by embedding shift

The main result is written to:

.. code-block:: text

   <odir>/results/factor_importance_rank.csv

Basic Usage
===========

Compare two region sets
-----------------------

.. code-block:: bash

   chrombert-tools interpret_regulator_effects_between_region_groups \
     --region1-file set1.bed \
     --region2-file set2.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use a fine-tuned checkpoint
---------------------------

Use ``--ft-ckpt`` to compute regulator embeddings with a fine-tuned model.

.. code-block:: bash

   chrombert-tools interpret_regulator_effects_between_region_groups \
     --region1-file set1.bed \
     --region2-file set2.bed \
     --ft-ckpt path/to/finetuned.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Run with Apptainer
------------------

Use ``--nv`` to enable GPU access.

.. code-block:: bash

   apptainer exec --nv /path/to/chrombert-tools.sif chrombert-tools interpret_regulator_effects_between_region_groups \
     --region1-file set1.bed \
     --region2-file set2.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required inputs
---------------

``--region1-file`` *(file path, required)*
   First region set. The file should contain at least ``chrom``, ``start``, and ``end``
   columns.

``--region2-file`` *(file path, required)*
   Second region set. The file should use the same format as ``--region1-file``.

Embedding options
-----------------

``--ft-ckpt`` *(file path, optional)*
   Fine-tuned checkpoint used to generate regulator embeddings.

   If this option is not provided, ChromBERT-tools uses the pre-trained ChromBERT model.

``--ignore-regulator`` *(string, optional)*
   Regulators to mask during embedding generation, separated by semicolons.

``--gep`` *(flag, default: False)*
   Use the GEP multi-flank-window model.

``--flank-window`` *(int, default: 4)*
   Flank window size used with ``--gep``.

``--model-config`` *(file path, optional)*
   Custom model configuration file.

``--data-config`` *(file path, optional)*
   Custom dataset configuration file.

Reference and output options
----------------------------

``--genome`` *(hg38 | mm10, default: hg38)*
   Reference genome.

``--resolution`` *(200bp | 1kb | 2kb | 4kb, default: 1kb)*
   ChromBERT bin resolution. For ``mm10``, only ``1kb`` is currently supported.

``--batch-size`` *(int, default: 4)*
   Batch size used for model inference.

``--odir`` *(directory, default: ./output)*
   Output directory. It will be created automatically if needed.

``--chrombert-cache-dir`` *(directory, default: ~/.cache/chrombert/data)*
   Directory for ChromBERT reference files, model files, and cached data.

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

``dataset/region1/``
   Overlap results for the first region set.

``dataset/region2/``
   Overlap results for the second region set.

``emb/mean_regulator_emb_region1.pkl``
   Mean regulator embeddings for region set 1.

``emb/mean_regulator_emb_region2.pkl``
   Mean regulator embeddings for region set 2.

``results/factor_importance_rank.csv``
   Main output file.

   Each row is one regulator. The table is sorted from the largest embedding shift to the
   smallest embedding shift.

   Main columns:

   * ``factors``: regulator name
   * ``similarity``: cosine similarity between the regulator embeddings from the two
     region sets
   * ``embedding_shift``: ``1 - similarity``
   * ``rank``: regulator rank; 1 means the largest embedding shift

Interpretation
==============

Regulators with larger ``embedding_shift`` values show stronger differences between the
two region sets.

These top-ranked regulators may represent context-specific regulators, cofactors, or
candidate drivers that distinguish the two region groups.

Tips
====

1. Use biologically comparable region sets when possible.
2. Region sets with similar size and genomic scale usually give cleaner rankings.
3. Use ``--ft-ckpt`` when you want cell-type-specific or task-specific regulator
   embeddings.
4. The mean regulator embedding files can be reused for downstream analyses.
5. To see all options, run:

.. code-block:: bash

   chrombert-tools interpret_regulator_effects_between_region_groups -h

Tutorials
=========

.. toctree::
   :maxdepth: 1

   CLI example <../examples/cli/interpret_regulator_effects_between_region_groups>
   Python API example <../examples/api/interpret_regulator_effects_between_region_groups>