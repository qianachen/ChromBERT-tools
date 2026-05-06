==========================================
interpret_regulator_regulator_interactions
==========================================

Build a regulator-regulator network from ChromBERT regulator embeddings.

This command compares regulator embeddings on a user-provided region set. Regulator pairs
with high cosine similarity are connected in the network.

It can also draw subnetworks around regulators of interest.

Overview
========

``interpret_regulator_regulator_interactions`` identifies regulators with similar
regulatory contexts in the input regions.

The command:

1. overlaps input regions with ChromBERT bins
2. generates regulator embeddings for the input regions
3. averages each regulator's embeddings across regions
4. computes pairwise cosine similarity between regulators
5. builds a network using the most similar regulator pairs
6. optionally draws k-hop subnetworks for selected regulators

The main outputs include:

* a regulator cosine similarity matrix
* a thresholded regulator-regulator edge list
* optional PDF subnetwork figures

Basic Usage
===========

Build the full regulator network
--------------------------------

.. code-block:: bash

   chrombert-tools interpret_regulator_regulator_interactions \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Plot subnetworks for selected regulators
----------------------------------------

.. code-block:: bash

   chrombert-tools interpret_regulator_regulator_interactions \
     --region regions.bed \
     --regulator "EZH2;BRD4;CTCF" \
     --k-hop 1 \
     --quantile 0.98 \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use a fine-tuned checkpoint
---------------------------

Use ``--ft-ckpt`` to build a cell-type-specific or task-specific regulator network.

.. code-block:: bash

   chrombert-tools interpret_regulator_regulator_interactions \
     --region regions.bed \
     --ft-ckpt path/to/finetuned.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Run with Apptainer
------------------

Use ``--nv`` to enable GPU access.

.. code-block:: bash

   apptainer exec --nv /path/to/chrombert-tools.sif chrombert-tools interpret_regulator_regulator_interactions \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Input options
-------------

``--region`` *(file path, required)*
   Input genomic regions. The file should contain at least ``chrom``, ``start``, and
   ``end`` columns.

``--regulator`` *(string, optional)*
   Regulators used for subnetwork plotting, separated by semicolons. For example:

   ``"EZH2;BRD4;CTCF"``

   If this option is not provided, ChromBERT-tools still builds the full network but does
   not generate subnetwork PDF figures.

Graph options
-------------

``--quantile`` *(float, default: 0.98)*
   Quantile used to select network edges.

   The default value ``0.98`` keeps approximately the top 2% most similar regulator pairs.
   Use a higher value for a sparser network and a lower value for a denser network.

``--k-hop`` *(int, default: 1)*
   Size of the subnetwork drawn around each selected regulator.

   ``1`` includes direct neighbors.

   ``2`` also includes neighbors of neighbors.

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
   Reference genome. For ``mm10``, only ``1kb`` resolution is currently supported.

``--resolution`` *(200bp | 1kb | 2kb | 4kb, default: 1kb)*
   ChromBERT bin resolution.

``--batch-size`` *(int, default: 64)*
   Batch size used for model inference.

``--odir`` *(directory, default: ./output)*
   Output directory. It will be created automatically if needed.

``--chrombert-cache-dir`` *(directory, default: ~/.cache/chrombert/data)*
   Directory for ChromBERT reference files, model files, and cached data.

Required cache files
====================

The command uses the following ChromBERT cache files:

* ChromBERT reference region file
* ChromBERT regulator list
* ChromBERT HDF5 feature file
* pre-trained ChromBERT checkpoint
* mask matrix

Outputs
=======

The following files are written under ``<odir>``.

``model_input.tsv``
   Processed input regions used for embedding generation.

``overlap_region.bed``
   Input regions that overlap ChromBERT reference bins.

``no_overlap_region.bed``
   Input regions that do not overlap ChromBERT reference bins.

``emb/mean_regulator_emb_region.pkl``
   Mean regulator embeddings across the input regions.

``regulator_cosine_similarity.tsv``
   Pairwise regulator cosine similarity matrix.

``total_graph_edge_threshold<thr>_quantile<q>.tsv``
   Edge list of the regulator-regulator network.

   Main columns:

   * ``node1``
   * ``node2``
   * ``cosine_similarity``

``subnetwork_<reg>_k<k>_q<q>_thr<thr>.pdf``
   PDF subnetwork figure for each matched regulator provided by ``--regulator``.

   This file is generated only when ``--regulator`` is provided.

Interpretation
==============

Regulator pairs with higher cosine similarity have more similar regulatory contexts in
the input regions.

The thresholded network can be used to identify co-associated regulators or candidate
co-regulatory modules.

Subnetwork PDFs help visualize regulators closely connected to a selected regulator of
interest.

Tips
====

1. Use ``--quantile`` to control network density.
2. The default ``--quantile 0.98`` keeps only the strongest regulator pairs and is useful
   for cleaner figures.
3. Leave ``--regulator`` unset if you only need the full cosine matrix and edge list.
4. Use ``--ft-ckpt`` when you want a cell-type-specific or task-specific network.
5. The mean regulator embedding file can be reused for downstream analyses.
6. To see all options, run:

.. code-block:: bash

   chrombert-tools interpret_regulator_regulator_interactions -h

Tutorials
=========

.. toctree::
   :maxdepth: 1

   CLI example <../examples/cli/interpret_regulator_regulator_interactions>
   Python API example <../examples/api/interpret_regulator_regulator_interactions>