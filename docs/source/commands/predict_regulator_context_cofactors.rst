=====================================
predict_regulator_context_cofactors
=====================================

Identify **context-dependent cofactors** for dual-functional regulators across region classes.

Overview
========

Define two or more functional region classes with repeated ``--function-bed``, specify ``--dual-regulator`` (``;``-separated), then train or load a classifier and compare cofactor patterns between class pairs. Results live under ``{odir}/results/<pair>/``.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools predict_regulator_context_cofactors \
     --function-bed active_enh.bed \
     --function-bed poised_enh.bed \
     --dual-regulator "CTCF;BRD4" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters (summary)
====================

``--function-bed`` (repeatable)
   One class per repeat; ``;`` to merge multiple BEDs in one class.

``--function-mode``, ``--function-name``
   Per-class merge mode (``and``/``or``) and labels.

``--dual-regulator`` (required)
   Regulators of interest (``;``-separated).

``--ignore-regulator``
   Excluded regulators.

``--threshold``, ``--quantile``
   Cofactor detection and graph edge settings.

``--ft-ckpt``
   Reuse a fine-tuned checkpoint instead of training.

``--mode``, ``--genome``, ``--resolution``, ``--batch-size``, ``--chrombert-cache-dir``, ``--odir``
   Standard options.

Output
======

``{odir}/results/<label_pair_subdir>/`` for each class pair (printed at end of run).
