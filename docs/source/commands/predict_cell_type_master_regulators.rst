===================================
predict_cell_type_master_regulators
===================================

Rank **cell-type–specific master regulators** from chromatin accessibility (fine-tune or reuse checkpoint).

Overview
========

Either provide ``--cell-type-bw`` and ``--cell-type-peak`` to fine-tune ChromBERT on the target cell type, or pass ``--ft-ckpt`` to skip training. Outputs a ranked list of regulator importance (e.g. ``factor_importance_rank.csv``).

Basic Usage
===========

Fine-tune from accessibility tracks:

.. code-block:: bash

   chrombert-tools predict_cell_type_master_regulators \
     --cell-type-bw cell.bw \
     --cell-type-peak cell_peaks.bed \
     --genome hg38 \
     --odir output

Reuse a checkpoint:

.. code-block:: bash

   chrombert-tools predict_cell_type_master_regulators \
     --ft-ckpt path/to/model.ckpt \
     --genome hg38 \
     --odir output

Parameters (summary)
====================

``--cell-type-bw``, ``--cell-type-peak``
   BigWig + peaks for the target cell type (required if ``--ft-ckpt`` is not set).

``--ft-ckpt``
   Existing fine-tuned checkpoint (skips fine-tuning).

``--mode``
   ``fast`` or ``full`` region subsampling.

``--genome``, ``--resolution``, ``--batch-size``, ``--chrombert-cache-dir``, ``--odir``
   Standard options.

Output
======

Rankings under ``--odir`` (e.g. ``factor_importance_rank.csv``). The CLI prints the final path when finished.
