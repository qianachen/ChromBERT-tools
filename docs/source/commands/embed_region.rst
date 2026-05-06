============
embed_region
============

Generate **768-dimensional region embeddings** and/or **gene embeddings** using ChromBERT.

By default, this command uses the pre-trained ChromBERT model. It can also generate
cell-type-specific embeddings when a fine-tuned checkpoint or cell-type accessibility
data are provided.

Overview
========

You must provide at least one of the following inputs:

* ``--region``: a BED file of genomic regions
* ``--gene``: gene symbols or Ensembl IDs

If both are provided, ChromBERT-tools will generate both region and gene embeddings in
the same run.

Modes
=====

General mode
------------

General mode is used when no cell-type-specific information is provided.

In this mode, ChromBERT-tools uses the pre-trained ChromBERT model. If precomputed
genome-wide region embeddings are available, the command directly extracts the requested
rows from the cached embeddings. Otherwise, it loads the model and computes embeddings.

Cell-type-specific mode
-----------------------

Cell-type-specific mode is used when either of the following is provided:

* ``--ft-ckpt``: a fine-tuned checkpoint
* both ``--cell-type-bw`` and ``--cell-type-peak``: cell-type accessibility signal and peaks

If ``--ft-ckpt`` is provided, ChromBERT-tools loads the checkpoint directly and does not
perform fine-tuning.

If ``--cell-type-bw`` and ``--cell-type-peak`` are provided without ``--ft-ckpt``,
ChromBERT-tools first fine-tunes a cell-type-specific model, then uses it to generate
embeddings.

Basic Usage
===========

Regions only
------------

.. code-block:: bash

   chrombert-tools embed_region \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Genes only
----------

.. code-block:: bash

   chrombert-tools embed_region \
     --gene "TP53;BRD4" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Regions and genes
-----------------

.. code-block:: bash

   chrombert-tools embed_region \
     --region regions.bed \
     --gene "TP53;BRD4" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Generate cell-type-specific embeddings from accessibility data
--------------------------------------------------------------

.. code-block:: bash

   chrombert-tools embed_region \
     --region regions.bed \
     --cell-type-bw cell_accessibility.bigwig \
     --cell-type-peak cell_peaks.bed \
     --genome hg38 \
     --resolution 1kb \
     --mode fast \
     --odir output_cell

Generate cell-type-specific embeddings from a checkpoint
--------------------------------------------------------

.. code-block:: bash

   chrombert-tools embed_region \
     --region regions.bed \
     --ft-ckpt path/to/finetuned.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output_ckpt

Run with Apptainer
------------------

Use ``--nv`` to enable GPU access:

.. code-block:: bash

   apptainer exec --nv /path/to/chrombert-tools.sif chrombert-tools embed_region \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required input
--------------

At least one of the following is required.

``--region`` *(file path)*
   BED file containing genomic regions. The file should include at least ``chrom``,
   ``start``, and ``end`` columns.

``--gene`` *(string)*
   Gene symbols or Ensembl IDs separated by semicolons, for example
   ``"TP53;BRD4"``.

Cell-type-specific options
--------------------------

``--cell-type-bw`` *(file path)*
   Cell-type-specific accessibility signal in BigWig format. Must be used together with
   ``--cell-type-peak`` unless ``--ft-ckpt`` is provided.

``--cell-type-peak`` *(file path)*
   Cell-type-specific accessibility peaks in BED format. Must be used together with
   ``--cell-type-bw`` unless ``--ft-ckpt`` is provided.

``--ft-ckpt`` *(file path, optional)*
   Path to a fine-tuned checkpoint. When provided, ChromBERT-tools loads this checkpoint
   directly and skips fine-tuning.

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
   Batch size used during model inference. This has no effect when cached embeddings are
   used directly.

``--odir`` *(directory, default: ./output)*
   Output directory. It will be created automatically if it does not exist.

``--oname`` *(string, default: embedding)*
   Output file name prefix.

Advanced options
----------------

``--chrombert-cache-dir`` *(directory, default: ~/.cache/chrombert/data)*
   Directory for ChromBERT reference files and cached data.

``--chrombert-region-file`` *(file path, optional)*
   Custom ChromBERT reference region BED file.

``--chrombert-region-emb-file`` *(file path, optional)*
   Custom precomputed genome-wide region embedding file.

``--chrombert-gene-meta`` *(file path, optional)*
   Custom gene metadata file.

Output Files
============

When ``--region`` is used
-------------------------

The following files are written to ``--odir``:

``region_emb_<oname>.npy``
   Region embedding array with shape ``(n_regions, 768)``. Each row corresponds to one
   ChromBERT bin overlapping the input regions.

``overlap_region.bed``
   Input regions that overlap ChromBERT bins.

``no_overlap_region.bed``
   Input regions that do not overlap ChromBERT bins.

``model_input.tsv``
   Model input table, generated when model inference is needed.

When ``--gene`` is used
-----------------------

The following files are written to ``--odir``:

``gene_emb_<oname>.pkl``
   Python dictionary mapping each matched gene to a 768-dimensional embedding.

``overlap_genes_meta.tsv``
   Metadata for genes matched in the ChromBERT gene annotation.

``model_input_gene.tsv``
   Model input table for gene-associated regions.

Gene embeddings are calculated by averaging the embeddings of ChromBERT bins associated
with each gene promoter.

Load outputs in Python
======================

.. code-block:: python

   import numpy as np
   import pickle

   region_emb = np.load("output/region_emb_embedding.npy")
   print(region_emb.shape)

   with open("output/gene_emb_embedding.pkl", "rb") as f:
       gene_emb = pickle.load(f)

Tips
====

1. If you already have a fine-tuned checkpoint, use ``--ft-ckpt`` directly. You do not
   need to provide BigWig or peak files.
2. If you need both region and gene embeddings in cell-type-specific mode, provide
   ``--region`` and ``--gene`` in the same command. The model will be loaded or trained
   only once.
3. For first-time gene embedding runs, ChromBERT-tools may download the required gene
   metadata automatically.
4. In general mode, cached genome-wide embeddings may be used when available. In this
   case, no model forward pass is needed.
5. To see all available options, run:

.. code-block:: bash

   chrombert-tools embed_region -h

Tutorials
=========

.. toctree::
   :maxdepth: 1

   CLI example <../examples/cli/embed_region>
   Python API example <../examples/api/embed_region>