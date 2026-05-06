====================================
interpret_region_region_interactions
====================================

Measure similarity between genomic regions using ChromBERT region embeddings.

This command computes cosine similarity between region embeddings. Higher similarity
suggests that two regions have more similar regulatory contexts.

Overview
========

``interpret_region_region_interactions`` supports two modes:

* **Enhancer-promoter mode**: compare input regions with nearby gene TSS regions
* **Two-region-set mode**: compare regions from two input files

Only same-chromosome region pairs are considered. Region pairs are kept if their genomic
distance falls within the range defined by ``--distance-min`` and ``--distance-max``.

The default distance range is 0 to 250 kb.

Modes
=====

Enhancer-promoter mode
----------------------

Use this mode by providing only ``--region``.

ChromBERT-tools compares each input region with nearby gene TSS regions from the
ChromBERT gene metadata.

The output is:

.. code-block:: text

   <odir>/tss_region_pairs_cos.tsv

Two-region-set mode
-------------------

Use this mode by providing both ``--region`` and ``--region2``.

ChromBERT-tools compares regions from the first file with regions from the second file.

The output is:

.. code-block:: text

   <odir>/region_set_pairs_cos.tsv

Basic Usage
===========

Enhancer-promoter mode
----------------------

.. code-block:: bash

   chrombert-tools interpret_region_region_interactions \
     --region candidates.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Two-region-set mode
-------------------

.. code-block:: bash

   chrombert-tools interpret_region_region_interactions \
     --region set1.bed \
     --region2 set2.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Restrict to specific genes
--------------------------

This option is available only in enhancer-promoter mode.

Use ``--gene`` or ``--gene-id`` to compare input regions only with selected gene TSSs.
The two filters can be used together.

.. code-block:: bash

   chrombert-tools interpret_region_region_interactions \
     --region candidates.bed \
     --gene "MYC;TP53" \
     --gene-id "ENSG00000136997;ENSG00000141510" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use a custom distance range
---------------------------

.. code-block:: bash

   chrombert-tools interpret_region_region_interactions \
     --region candidates.bed \
     --distance-min 50000 \
     --distance-max 250000 \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Run with Apptainer
------------------

Use ``--nv`` to enable GPU access.

.. code-block:: bash

   apptainer exec --nv /path/to/chrombert-tools.sif chrombert-tools interpret_region_region_interactions \
     --region candidates.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Input options
-------------

``--region`` *(file path, required)*
   First input region file. The file should contain at least ``chrom``, ``start``, and
   ``end`` columns.

``--region2`` *(file path, optional)*
   Second input region file. When provided, the command runs in two-region-set mode.

``--gene`` *(string, optional)*
   Gene symbols separated by semicolons. Used to restrict TSS regions in
   enhancer-promoter mode.

``--gene-id`` *(string, optional)*
   Gene IDs separated by semicolons. Used to restrict TSS regions in enhancer-promoter
   mode.

Distance options
----------------

``--distance-min`` *(int, default: 0)*
   Minimum genomic distance in base pairs.

``--distance-max`` *(int, default: 250000)*
   Maximum genomic distance in base pairs.

Only same-chromosome pairs within this distance range are kept.

Embedding options
-----------------

``--ft-ckpt`` *(file path, optional)*
   Fine-tuned checkpoint used to generate region embeddings.

   If this option is not provided, ChromBERT-tools uses cached precomputed embeddings when
   available, or the pre-trained ChromBERT model otherwise.

``--gep`` *(flag, default: False)*
   Use the GEP multi-flank-window model to compute embeddings.

``--flank-window`` *(int, default: 4)*
   Flank window size used with ``--gep``.

``--ignore-regulator`` *(string, optional)*
   Regulators to mask during embedding generation, separated by semicolons.

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

``--chrombert-region-file`` *(file path, optional)*
   Custom ChromBERT reference region BED file.

``--chrombert-region-emb-file`` *(file path, optional)*
   Custom precomputed region embedding file.

Required cache files
====================

The command uses the following ChromBERT cache files:

* ChromBERT reference region file
* ChromBERT HDF5 feature file

Enhancer-promoter mode also requires:

* ChromBERT gene metadata file

Outputs
=======

Enhancer-promoter mode
----------------------

The following files are written under ``--odir``.

``model_input.tsv``
   Regions used to compute embeddings, including input regions and selected TSS regions.

``tss_region_pairs_cos.tsv``
   Similarity scores between TSS regions and input regions.

   Main columns include:

   * ``chrom``
   * ``gene_id``
   * ``gene_name``
   * ``tss``
   * ``tss_build_region_index``
   * ``distal_region_start``
   * ``distal_region_end``
   * ``distal_region_build_region_index``
   * ``dist``
   * ``dist_bin``
   * ``cos_sim``

Two-region-set mode
-------------------

The following files are written under ``--odir``.

``dataset/region1/``
   Overlap results for the first input region file.

``dataset/region2/``
   Overlap results for the second input region file.

``model_input.tsv``
   Regions used to compute embeddings.

``region_set_pairs_cos.tsv``
   Similarity scores between regions from the two input files.

   Main columns include:

   * ``set1_chrom``
   * ``set1_start``
   * ``set1_end``
   * ``set1_build_region_index``
   * ``set2_chrom``
   * ``set2_start``
   * ``set2_end``
   * ``set2_build_region_index``
   * ``genomic_dist_bp``
   * ``cos_sim``

Tips
====

1. Use enhancer-promoter mode when you want to compare candidate distal regions with gene
   TSS regions.
2. Use two-region-set mode when you want to compare two custom sets of genomic regions.
3. Use ``--distance-min`` and ``--distance-max`` to control the genomic distance range.
4. Use ``--gene`` or ``--gene-id`` to focus on specific genes in enhancer-promoter mode.
5. Use ``--ft-ckpt`` if you want to use cell-type-specific or task-specific embeddings.
6. To see all options, run:

.. code-block:: bash

   chrombert-tools interpret_region_region_interactions -h

Tutorials
=========

.. toctree::
   :maxdepth: 1

   CLI example <../examples/cli/interpret_region_region_interactions>
   Python API example <../examples/api/interpret_region_region_interactions>