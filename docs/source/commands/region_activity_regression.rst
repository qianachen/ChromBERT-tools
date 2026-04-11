============================
region_activity_regression
============================

Predict **chromatin accessibility** (log2 signal or fold change between two states) at ChromBERT regions.

Overview
========

The ``region_activity_regression`` command prepares peak + BigWig inputs, fine-tunes a ChromBERT regression head (unless ``--ft-ckpt`` is given), and writes predictions. **Predict-only:** pass ``--ft-ckpt`` and ``--predict-file`` to skip training.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools region_activity_regression \
     --acc-peak1 state1_peaks.bed \
     --acc-signal1 state1.bw \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Two-state fold change (example):

.. code-block:: bash

   chrombert-tools region_activity_regression \
     --acc-peak1 s1.bed --acc-signal1 s1.bw \
     --acc-peak2 s2.bed --acc-signal2 s2.bw \
     --direction 2-1 \
     --genome hg38 \
     --odir output

Parameters (summary)
====================

``--acc-peak1``, ``--acc-signal1``
   State 1 peaks and BigWig(s); ``;`` for replicates. Required for training unless predict-only.

``--acc-peak2``, ``--acc-signal2``
   Optional state 2 (omit for single-state prediction).

``--predict-file``
   Regions TSV/CSV (``chrom``, ``start``, ``end``, ``build_region_index``; optional ``label``). With ``--ft-ckpt``, skips training.

``--direction``
   ``2-1`` or ``1-2`` (two-state fold-change sign).

``--mode``
   ``fast`` or ``full``.

``--include-tss-background``, ``--tss-flank``
   Optional TSS±flank background bins.

``--ft-ckpt``, ``--genome``, ``--resolution``, ``--batch-size``, ``--chrombert-cache-dir``, ``--odir``
   See ``chrombert-tools region_activity_regression -h``.

Output
======

Typically ``{odir}/predict/predictions.csv`` after training or predict-only; checkpoints and dataset under ``--odir``. See CLI help for details.
