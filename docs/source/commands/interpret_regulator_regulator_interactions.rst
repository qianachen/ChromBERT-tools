==========================================
interpret_regulator_regulator_interactions
==========================================

Infer **regulator–regulator co-association** (cosine similarity graph) on user regions.

Overview
========

The ``interpret_regulator_regulator_interactions`` command embeds regulators across a focus region set, computes pairwise cosine similarities, thresholds edges (quantile), and optionally extracts k-hop subnetworks for listed regulators.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools interpret_regulator_regulator_interactions \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

With explicit regulators for subnetwork focus:

.. code-block:: bash

   chrombert-tools interpret_regulator_regulator_interactions \
     --region regions.bed \
     --regulator "EZH2;BRD4;CTCF" \
     --genome hg38 \
     --odir output

Parameters (summary)
====================

``--region`` (required)
   Focus regions (BED or CSV/TSV with ``chrom``, ``start``, ``end``).

``--regulator``
   Optional ``;``-separated names for subnetwork output.

``--ft-ckpt``
   Fine-tuned checkpoint for embeddings.

``--ignore-regulator``, ``--gep``, ``--flank-window``
   Ignore list; GEP multi-flank mode.

``--quantile``, ``--k-hop``
   Edge thresholding and subgraph radius.

``--model-config``, ``--data-config``
   Optional JSON configs instead of auto-built configs.

``--genome``, ``--resolution``, ``--batch-size``, ``--chrombert-cache-dir``, ``--odir``
   Standard options.

Output
======

* ``regulator_cosine_similarity.tsv``
* ``total_graph_edge_threshold*_quantile*.tsv``

Paths are printed at the end of the run.
