=================================================
interpret_regulator_effects_between_region_groups
=================================================

Compare **regulator embedding shifts** between two region sets.

Overview
========

The ``interpret_regulator_effects_between_region_groups`` command builds embeddings for regulators on ``--region1-file`` and ``--region2-file``, then summarizes differences (e.g. cosine / shift metrics) to highlight regulators that behave differently between the two genomic contexts.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools interpret_regulator_effects_between_region_groups \
     --region1-file set1.bed \
     --region2-file set2.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters (summary)
====================

``--region1-file``, ``--region2-file`` (required)
   Two region files (BED or compatible tables).

``--ft-ckpt``
   Fine-tuned checkpoint for embedding extraction.

``--ignore-regulator``, ``--gep``, ``--flank-window``
   Same semantics as other interpret commands.

``--model-config``, ``--data-config``
   Optional JSON configs.

``--genome``, ``--resolution``, ``--batch-size``, ``--chrombert-cache-dir``, ``--odir``
   Standard options.

Output
======

Tabular results under ``--odir`` (exact filenames printed by the CLI). Run ``chrombert-tools interpret_regulator_effects_between_region_groups -h`` for details.
