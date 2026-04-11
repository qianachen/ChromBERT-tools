======================================
interpret_region_region_interactions
======================================

Quantify **region-region embedding similarity** (enhancer-promoter style with gene TSS, or two BED sets).

Overview
========

With a single ``--region`` BED, the command pools embeddings for overlapping ChromBERT bins and gene TSS windows, then reports cosine similarities within a genomic distance window (default 250 kb). With ``--region2``, it compares two region sets on the same chromosome within ``--distance-window``.

Basic Usage
===========

Enhancer-promoter / regions vs TSS (omit ``--region2``):

.. code-block:: bash

   chrombert-tools interpret_region_region_interactions \
     --region candidates.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Two region sets:

.. code-block:: bash

   chrombert-tools interpret_region_region_interactions \
     --region set1.bed \
     --region2 set2.bed \
     --genome hg38 \
     --odir output

Parameters (summary)
====================

``--region`` (required)
   First BED (or CSV/TSV with ``chrom``, ``start``, ``end``).

``--region2``
   Optional second BED; if set, outputs ``region_set_pairs_cos.tsv`` instead of TSS-focused pairs.

``--distance-window``
   Max separation in bp (same chromosome; cross-chrom dropped).

``--ft-ckpt``, ``--gep``, ``--flank-window``, ``--ignore-regulator``, ``--model-config``, ``--data-config``
   Optional fine-tuning / GEP / JSON configs (see ``-h``).

``--genome``, ``--resolution``, ``--batch-size``, ``--chrombert-cache-dir``, ``--chrombert-region-file``, ``--chrombert-region-emb-file``, ``--odir``
   Standard options.

Output
======

* Single BED: ``tss_region_pairs_cos.tsv``
* Two BEDs: ``region_set_pairs_cos.tsv``

Also writes ``model_input.tsv`` (and per-set overlap artifacts under subfolders when using two BEDs).
