==========================
predict_tf_binding_regions
==========================

**Impute / predict TF binding** (cistrome) on regions for given factor:cell combinations (**1 kb human only**).

Overview
========

The ``predict_tf_binding_regions`` command takes a region BED and a ``--cistrome`` string such as ``CTCF:K562;BRD4:MCF7``, then runs the TF-binding imputation path and writes outputs under ``--odir`` with prefix ``--oname``.

Basic Usage
===========

.. code-block:: bash

   chrombert-tools predict_tf_binding_regions \
     --region regions.bed \
     --cistrome "CTCF:GM12878;BRD4:MCF7" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters (summary)
====================

``--region`` (required)
   Region BED.

``--cistrome`` (required)
   ``factor:celltype`` entries separated by ``;``.

``--resolution``
   Must be ``1kb`` (human TF-binding task).

``--oname``
   Output name prefix (default: ``cistrome_impute``).

``--batch-size``, ``--num-workers``, ``--genome``, ``--chrombert-cache-dir``, ``--odir``
   Standard options.

Output
======

Imputation tables and side files under ``--odir`` (see CLI log). Run ``chrombert-tools predict_tf_binding_regions -h`` for details.
