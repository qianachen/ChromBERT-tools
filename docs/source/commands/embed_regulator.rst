================
embed_regulator
================

Extract **regulator embeddings** on user regions with the **pre-trained** ChromBERT model, or with a **cell-type-specific** model (fine-tuned from accessibility tracks or loaded from a checkpoint).

Overview
========

``chrombert-tools embed_regulator`` runs a forward pass through ChromBERT's regulator head on every overlapping ChromBERT bin in ``--region``, and writes:

* **region-aware** embeddings per regulator (HDF5),
* **mean** embeddings per regulator (pickle).

**Mode selection** (implemented in ``is_cell_specific`` in ``utils_embed.py``):

* **General (pre-trained):** default when you do **not** pass ``--ft-ckpt`` and do **not** pass **both** ``--cell-type-bw`` and ``--cell-type-peak``.
* **Cell-type-specific:** when **either**

  * ``--ft-ckpt`` is set (reuse a fine-tuned checkpoint, no training in this command), **or**
  * **both** ``--cell-type-bw`` and ``--cell-type-peak`` are set (fine-tune on that cell type's accessibility).

If you are in cell-type mode, you must provide **either** ``--ft-ckpt`` **or** **both** BigWig and peak files; otherwise the CLI raises the same error as ``validate_args`` in ``embed_regulator.py``.

Basic Usage
===========

General (pre-trained model)
---------------------------

.. code-block:: bash

   chrombert-tools embed_regulator \
     --regulator "CTCF;BRD4;MYC" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Cell-type-specific: fine-tune from accessibility
------------------------------------------------

.. code-block:: bash

   chrombert-tools embed_regulator \
     --regulator "CTCF;BRD4" \
     --region regions.bed \
     --cell-type-bw cell_accessibility.bigwig \
     --cell-type-peak cell_peaks.bed \
     --genome hg38 \
     --resolution 1kb \
     --mode fast \
     --odir output_cell_specific

Cell-type-specific: reuse a checkpoint (skip fine-tuning)
---------------------------------------------------------

.. code-block:: bash

   chrombert-tools embed_regulator \
     --regulator "CTCF;BRD4" \
     --region regions.bed \
     --ft-ckpt path/to/finetuned.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output_from_ckpt

Singularity
-----------

.. code-block:: bash

   singularity exec --nv /path/to/chrombert.sif chrombert-tools embed_regulator \
     --regulator "CTCF;BRD4" \
     --region regions.bed \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required
--------

``--region``
   Region file (BED, or CSV/TSV with columns ``chrom``, ``start``, ``end``). Passed to ``check_region_file``.

``--regulator``
   Regulators of interest, ``;``-separated (e.g. ``EZH2;BRD4``). Names are lowercased and matched against ``<chrombert-cache-dir>/config/*regulators_list.txt`` (see ``overlap_regulator_func``).

Cell-type-specific inputs (optional, control mode)
--------------------------------------------------

``--cell-type-bw``
   Cell-type chromatin accessibility **BigWig**. If set together with ``--cell-type-peak`` (and without ``--ft-ckpt``), triggers fine-tuning for cell-specific embeddings.

``--cell-type-peak``
   Cell-type accessibility **peak BED**. Must be paired with ``--cell-type-bw`` for the training path.

``--ft-ckpt``
   Fine-tuned checkpoint path. If set, uses the cell-specific embedding path and **skips** fine-tuning. Also selects cell-type mode **without** requiring BigWig/peaks.

``--mode``
   ``fast`` or ``full``: used when **training** the cell-specific model (i.e. when using ``--cell-type-bw`` and ``--cell-type-peak`` without ``--ft-ckpt``). Ignored for the general pre-trained path in typical use.

Other options
-------------

``--odir``
   Output directory (default: ``./output``).

``--oname``
   Stem for output files (default: ``regulator_emb``). Produces ``mean_<oname>.pkl`` and ``region_aware_<oname>.hdf5``.

``--genome``
   ``hg38`` (default) or ``mm10``.

``--resolution``
   ``1kb`` (default), ``200bp``, ``2kb``, or ``4kb``.

``--batch-size``
   Dataloader batch size (default: ``4``).

``--num-workers``
   Dataloader workers (default: ``8``); exposed on the CLI (see ``embed_regulator.py``).

``--chrombert-cache-dir``
   ChromBERT data cache (default: ``~/.cache/chrombert/data``).

Output Files
============

Written under ``--odir``:

``region_aware_<oname>.hdf5``
   Region-aware regulator embeddings. Datasets are named ``emb/<regulator>`` with **lowercase** regulator keys (see ``generate_regulator_embeddings`` in ``utils_embed.py``).

   .. code-block:: python

      import h5py

      # Default --oname regulator_emb
      with h5py.File("output/region_aware_regulator_emb.hdf5", "r") as f:
          emb_ctcf = f["emb/ctcf"][:]
          emb_brd4 = f["emb/brd4"][:]

``mean_<oname>.pkl``
   ``dict`` mapping each matched regulator (lowercase) to a length-768 mean vector.

``overlap_region.bed`` / ``no_overlap_region.bed``
   Overlap of input regions with ChromBERT reference bins.

The run finishes with a line ``Embedding type: general`` or ``Embedding type: cell-specific`` matching ``report_regulator``.

Tips
====

1. **Regulator not found**

   * Names must appear in ChromBERT's regulator list (the CLI prints matched vs not-found counts).

2. **Cell-type mode errors**

   * Provide **either** ``--ft-ckpt`` **or** **both** ``--cell-type-bw`` and ``--cell-type-peak``.
