==============================
region_function_classification
==============================

Train or run a ChromBERT fine-tuned classifier for **N functional region classes** (binary or multiclass).

Overview
========

The ``region_function_classification`` command builds a supervised dataset from BED-defined region classes, optionally fine-tunes ChromBERT, and predicts labels for a held-out or user-supplied region table.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools region_function_classification \
     --function-bed class_a.bed \
     --function-bed class_b.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Singularity example:

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools region_function_classification \
     --function-bed class_a.bed \
     --function-bed class_b.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

``--function-bed`` (repeatable)
   One BED (or ``;``-joined BED list) per class. Omit when both ``--ft-ckpt`` and ``--predict-file`` are set (predict-only).

``--function-mode`` (repeatable)
   Per class: ``and`` (intersection) or ``or`` (union) when merging multiple BEDs in one ``--function-bed``. Default: ``and``.

``--function-name`` (repeatable)
   Display name per class (default: ``function_0``, ``function_1``, …).

``--predict-file``
   TSV/CSV with ``chrom``, ``start``, ``end``, ``build_region_index``, ``label`` for prediction.

``--ft-ckpt``
   Fine-tuned checkpoint; skips training when set (often with ``--predict-file``).

``--ignore-regulator``
   Regulators to ignore; ``;``-separated.

``--mode``
   ``fast`` (downsample ~20k regions) or ``full``.

``--genome``, ``--resolution``, ``--batch-size``, ``--chrombert-cache-dir``, ``--odir``
   Standard ChromBERT options (see ``chrombert-tools region_function_classification -h``).

Output
======

Written under ``--odir`` (training logs, checkpoints, and prediction tables). Use ``chrombert-tools region_function_classification -h`` for the exact layout after your run.
