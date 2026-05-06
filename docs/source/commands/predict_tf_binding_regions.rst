==========================
predict_tf_binding_regions
==========================

Predict TF binding probabilities for user-provided genomic regions.

This command predicts binding probabilities for one or more ``factor:celltype``
combinations. It outputs both a prediction table and one BigWig track for each requested
cistrome.

This task currently supports only ``1kb`` resolution.

Overview
========

``predict_tf_binding_regions`` uses the ChromBERT cistrome-prompt model to predict TF
binding probabilities.

Required inputs:

* ``--region``: genomic regions to score
* ``--cistrome``: one or more ``factor:celltype`` combinations

For each matched ``factor:celltype`` pair, ChromBERT-tools predicts a probability between
0 and 1 for each input region that overlaps ChromBERT regions.

Basic Usage
===========

Predict one cistrome
--------------------

.. code-block:: bash

   chrombert-tools predict_tf_binding_regions \
     --region regions.bed \
     --cistrome "CTCF:K562" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Predict multiple cistromes
--------------------------

.. code-block:: bash

   chrombert-tools predict_tf_binding_regions \
     --region regions.bed \
     --cistrome "CTCF:GM12878;BRD4:MCF7;BCL11A:K562" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use explicit cistrome IDs
-------------------------

You can provide a ``GSM`` or ``ENC`` cistrome ID instead of a cell-type name.

.. code-block:: bash

   chrombert-tools predict_tf_binding_regions \
     --region regions.bed \
     --cistrome "CTCF:GSM2026781;BRD4:ENCFF000XYZ" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Run with Apptainer
------------------

Use ``--nv`` to enable GPU access.

.. code-block:: bash

   apptainer exec --nv /path/to/chrombert-tools.sif chrombert-tools predict_tf_binding_regions \
     --region regions.bed \
     --cistrome "CTCF:K562;BRD4:MCF7" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Required inputs
---------------

``--region`` *(file path, required)*
   Input genomic regions. The file should contain at least ``chrom``, ``start``, and
   ``end`` columns.

   Only regions overlapping ChromBERT reference bins are scored.

``--cistrome`` *(string, required)*
   One or more cistromes in ``factor:celltype`` format, separated by semicolons. For
   example:

   ``"CTCF:K562;BRD4:MCF7"``

   The ``celltype`` field can be either:

   * a cell-type name, such as ``K562``
   * a cistrome ID starting with ``GSM`` or ``ENC``

   Names are matched case-insensitively.

Reference and runtime options
-----------------------------

``--genome`` *(hg38 | mm10, default: hg38)*
   Reference genome.

``--resolution`` *(1kb, default: 1kb)*
   Resolution for prediction. Only ``1kb`` is currently supported for this command.

``--batch-size`` *(int, default: 4)*
   Batch size used for prediction.

``--num-workers`` *(int, default: 8)*
   Number of dataloader workers.

``--chrombert-cache-dir`` *(directory, default: ~/.cache/chrombert/data)*
   Directory for ChromBERT reference files, model files, and cached data.

Output options
--------------

``--odir`` *(directory, default: ./output)*
   Output directory. It will be created automatically if needed.

``--oname`` *(string, default: cistrome_impute)*
   Output name prefix. This option is currently reserved for future use.

Required cache files
====================

The command uses the following ChromBERT cache files:

* ChromBERT reference region file
* ChromBERT HDF5 feature file
* cistrome-prompt checkpoint
* pre-trained ChromBERT checkpoint
* mask matrix
* metadata file for cistrome matching

Outputs
=======

The following files are written under ``<odir>``.

``overlap_region.bed``
   Input regions that overlap ChromBERT reference bins.

``no_overlap_region.bed``
   Input regions that do not overlap ChromBERT reference bins. These regions are not
   scored.

``model_input.tsv``
   Processed input table used for model prediction.

``results_prob_df.csv``
   Main prediction table.

   It contains input region coordinates, matched ChromBERT region coordinates, and one
   probability column for each matched ``factor:celltype`` pair.

``<factor>_<celltype>.bw``
   BigWig probability track for each matched cistrome.

   Scores range from 0 to 1 and can be viewed in genome browsers such as IGV, UCSC, or
   WashU.

Interpretation
==============

Each prediction score is a TF binding probability for a given ``factor:celltype`` pair at
a given region.

Higher values indicate higher predicted binding probability.

Tips
====

1. Use ``factor:celltype`` format for each requested cistrome.
2. Separate multiple cistromes with semicolons.
3. Use explicit ``GSM`` or ``ENC`` IDs if a cell-type name is hard to match.
4. Check the console output for unmatched cistromes before using the results.
5. This command currently supports only ``1kb`` resolution.
6. To see all options, run:

.. code-block:: bash

   chrombert-tools predict_tf_binding_regions -h

Tutorials
=========

.. toctree::
   :maxdepth: 1

   CLI example <../examples/cli/predict_tf_binding_regions>
   Python API example <../examples/api/predict_tf_binding_regions>