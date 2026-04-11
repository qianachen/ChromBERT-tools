====================================
predict_transition_driver_regulators
====================================

Find **driver regulators** in cell-state transitions (expression and/or chromatin accessibility).

Overview
========

Provide **expression** (``--exp-tpm1`` / ``--exp-tpm2``), **accessibility** (peaks + BigWigs), or both. The command trains or loads fine-tuned models (``--ft-ckpt-exp``, ``--ft-ckpt-acc``) and writes factor importance rankings. Merged integrated rankings appear when both modalities complete.

Basic Usage
===========

Accessibility-only transition:

.. code-block:: bash

   chrombert-tools predict_transition_driver_regulators \
     --acc-peak1 s1.bed --acc-signal1 s1.bw \
     --acc-peak2 s2.bed --acc-signal2 s2.bw \
     --direction 2-1 \
     --genome hg38 \
     --odir output

Expression transition:

.. code-block:: bash

   chrombert-tools predict_transition_driver_regulators \
     --exp-tpm1 s1_tpm.csv \
     --exp-tpm2 s2_tpm.csv \
     --direction 2-1 \
     --genome hg38 \
     --odir output

Parameters (summary)
====================

``--exp-tpm1``, ``--exp-tpm2``
   CSV with ``gene_id`` and ``tpm``.

``--acc-peak1``, ``--acc-peak2``, ``--acc-signal1``, ``--acc-signal2``
   Peaks and BigWigs; ``;`` for replicates where supported.

``--direction``
   ``2-1`` or ``1-2``.

``--mode``, ``--flank-window``, ``--include-tss-background``, ``--tss-flank``
   Training / dataset options (see ``-h``).

``--ft-ckpt-exp``, ``--ft-ckpt-acc``
   Skip training for expression or accessibility branch.

``--genome``, ``--resolution``, ``--batch-size``, ``--chrombert-cache-dir``, ``--odir``
   Standard options.

Output
======

CSV rankings per modality under ``--odir``; integrated merge path printed when applicable.
