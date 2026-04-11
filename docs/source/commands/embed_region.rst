==============
embed_region
==============

Extract **768-dimensional region embeddings** and/or **gene-level embeddings** (promoter pooling) from ChromBERT, using the **pre-trained** model by default or a **cell-type-specific** model when you pass accessibility tracks or a fine-tuned checkpoint.

Overview
========

You must provide **at least one** of ``--region`` or ``--gene`` (``validate_args``).

**Mode** (``is_cell_specific`` in ``utils_embed.py``):

* **General:** neither ``--ft-ckpt`` nor **both** ``--cell-type-bw`` and ``--cell-type-peak``. Uses pre-trained weights; may **slice** precomputed ``region_emb_npy`` from the cache when available (see ``run_region_general`` / ``run_gene_general``).
* **Cell-type-specific:** ``--ft-ckpt`` is set **or** **both** ``--cell-type-bw`` and ``--cell-type-peak`` are set. Then ``build_cell_model_emb`` runs once; embeddings are computed with ``run_region_cell`` and/or ``run_gene_cell``.

In cell-type mode you must supply **either** ``--ft-ckpt`` **or** **both** BigWig and peak files (same rule as ``embed_regulator``).

Only the branches you request run: e.g. ``--gene`` alone skips ``run_region_*``; ``--region`` alone skips ``run_gene_*``.

Basic Usage
===========

Regions only (general)
----------------------

.. code-block:: bash

   chrombert-tools embed_region \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Genes only (general)
--------------------

.. code-block:: bash

   chrombert-tools embed_region \
     --gene "TP53;BRD4" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Regions + genes in one run (general)
------------------------------------

.. code-block:: bash

   chrombert-tools embed_region \
     --region regions.bed \
     --gene "TP53;BRD4" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Cell-type-specific: fine-tune from accessibility
----------------------------------------------

.. code-block:: bash

   chrombert-tools embed_region \
     --region regions.bed \
     --cell-type-bw cell_accessibility.bigwig \
     --cell-type-peak cell_peaks.bed \
     --genome hg38 \
     --resolution 1kb \
     --mode fast \
     --odir output_cell

Cell-type-specific: reuse checkpoint
------------------------------------

.. code-block:: bash

   chrombert-tools embed_region \
     --region regions.bed \
     --ft-ckpt path/to/finetuned.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output_ckpt

Singularity
-----------

Prefix with ``singularity exec --nv /path/to/chrombert.sif`` as for other ``chrombert-tools`` commands.

Parameters
==========

At least one required
----------------------

``--region``
   BED or CSV/TSV with ``chrom``, ``start``, ``end``. Drives ``check_region_file`` and ``run_region_*``.

``--gene``
   Gene symbols or Ensembl IDs, ``;``-separated; lowercased for matching (``parse_focus_genes`` / gene meta).

Cell-type-specific (optional; selects mode)
-------------------------------------------

``--cell-type-bw``
   Cell-type accessibility **BigWig**.

``--cell-type-peak``
   Cell-type **peak BED**; use with ``--cell-type-bw`` to fine-tune when ``--ft-ckpt`` is absent.

``--ft-ckpt``
   Fine-tuned checkpoint; enables cell-type path **without** BigWig/peaks and **skips** fine-tuning.

``--mode``
   ``fast`` or ``full``: used when **training** the cell-specific model (BigWig + peak, no ``--ft-ckpt``). See Click help text in ``embed_region.py``.

Other options
-------------

``--odir``
   Output directory (default ``./output``).

``--oname``
   Name stem (default ``embedding``). Region array: ``region_emb_<oname>.npy``; gene pickle: ``gene_emb_<oname>.pkl``.

``--genome``, ``--resolution``, ``--batch-size``
   Reference genome, ChromBERT resolution, and dataloader batch size (default batch ``4``).

``--chrombert-cache-dir``
   Data cache root (default ``~/.cache/chrombert/data``).

``--chrombert-region-file``, ``--chrombert-region-emb-file``, ``--chrombert-gene-meta``
   Optional overrides for reference BED, cached region ``.npy``, and gene meta TSV (passed through ``resolve_paths``).

Output Files
============

When ``--region`` is used
-------------------------

* ``region_emb_<oname>.npy`` — array shape ``(n_overlap_rows, 768)`` (one row per overlapping ChromBERT bin in ``overlap_bed``; a single input interval can yield multiple rows).
* ``overlap_region.bed`` / ``no_overlap_region.bed`` — overlap diagnostics (from ``check_region_file`` / ``overlap_region`` pipeline).
* Log line ``Embedding type: general`` or ``cell-specific`` from ``report_region``.

``report_region`` counts lines in ``args.region`` for the “total focus regions” summary; it requires ``--region`` to be set (gene-only runs do not call ``report_region``).

When ``--gene`` is used
-----------------------

* ``gene_emb_<oname>.pkl`` — ``dict`` mapping matched gene keys to ``(768,)`` vectors (mean over promoter-associated bins in ``pool_gene_embeddings``).
* ``overlap_genes_meta.tsv`` — matched gene metadata (see ``report_gene``).
* Intermediate ``model_input_gene.tsv`` under ``--odir`` for the gene bin table.
* ``Embedding type: general`` or ``cell-specific`` from ``report_gene``.

General path may print **“Using cached region embeddings…”** when ``region_emb_npy`` exists on disk and is used for slicing (no forward pass), unless an ``ignore_regulator`` attribute is set on ``args`` (reserved for API callers).

.. code-block:: python

   import numpy as np
   import pickle

   reg = np.load("output/region_emb_embedding.npy")
   print(reg.shape)

   with open("output/gene_emb_embedding.pkl", "rb") as f:
       gene_emb = pickle.load(f)

Tips
====

1. **Cell-type mode:** supply **either** ``--ft-ckpt`` **or** **both** ``--cell-type-bw`` and ``--cell-type-peak``.
2. **Gene meta:** if the default gene meta TSV is missing, the tool may **download** ChromBERT data into ``--chrombert-cache-dir`` (``load_gene_meta``).
3. For full flag text, run ``chrombert-tools embed_region -h``.
