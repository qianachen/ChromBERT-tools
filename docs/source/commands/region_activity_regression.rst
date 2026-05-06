==========================
region_activity_regression
==========================

Fine-tune a ChromBERT region activity regression model from chromatin accessibility data,
or use an existing checkpoint to predict region activity.

The command supports two prediction targets:

* **single-state accessibility**: ``log2(1 + accessibility signal)``
* **two-state accessibility change**: ``log2(1 + signal2) - log2(1 + signal1)``

Overview
========

``region_activity_regression`` can be used in two ways:

* **training + prediction**: build a training dataset from peaks and BigWig signals, train
  a regression model, and predict region activity
* **predict-only**: load an existing fine-tuned checkpoint and predict directly

During training, ChromBERT-tools prepares the accessibility dataset, trains or loads a
regression model, and writes predictions to:

.. code-block:: text

   <odir>/predict/predictions.csv

Input accessibility data
========================

The command uses two types of accessibility inputs:

* peak BED files, provided by ``--acc-peak1`` and optionally ``--acc-peak2``
* BigWig signal files, provided by ``--acc-signal1`` and optionally ``--acc-signal2``

Peaks define the regions used for training. BigWig files provide the accessibility signal
for those regions.

Single-state mode
-----------------

Use ``--acc-peak1`` and ``--acc-signal1``.

By default, the target is:

.. code-block:: text

   log2(1 + state1 signal) - log2(1 + reference baseline)

To use the raw accessibility signal without subtracting the reference baseline, add:

.. code-block:: bash

   --subtract-reference-baseline

In that case, the target becomes:

.. code-block:: text

   log2(1 + state1 signal)

Two-state mode
--------------

Use ``--acc-signal1`` and ``--acc-signal2``.

The default target is:

.. code-block:: text

   log2(1 + state2 signal) - log2(1 + state1 signal)

Use ``--direction`` to control the direction:

* ``2-1``: keep ``state2 - state1``
* ``1-2``: use ``state1 - state2``

Replicates
----------

Multiple peak or BigWig files can be provided with semicolons:

.. code-block:: bash

   --acc-peak1 "rep1.bed;rep2.bed"
   --acc-signal1 "rep1.bw;rep2.bw"

For peak replicates, ChromBERT-tools uses the union of overlapping ChromBERT bins.

For BigWig replicates, ChromBERT-tools averages the signal across replicates before
calculating labels.

Optional TSS background regions
===============================

For two-state or transition analyses, it can be useful to include regions around gene TSSs.

Add this option to include TSS-centered background regions:

.. code-block:: bash

   --include-tss-background

The TSS window size is controlled by:

.. code-block:: bash

   --tss-flank 10000

This adds protein-coding TSS ± ``--tss-flank`` regions to the training region set.

Basic Usage
===========

Train from one accessibility state
----------------------------------

.. code-block:: bash

   chrombert-tools region_activity_regression \
     --acc-peak1 state1_peaks.bed \
     --acc-signal1 state1.bw \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Train from one state without baseline subtraction
-------------------------------------------------

.. code-block:: bash

   chrombert-tools region_activity_regression \
     --acc-peak1 state1_peaks.bed \
     --acc-signal1 state1.bw \
     --subtract-reference-baseline \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Train from two accessibility states
-----------------------------------

.. code-block:: bash

   chrombert-tools region_activity_regression \
     --acc-peak1 state1_peaks.bed \
     --acc-signal1 state1.bw \
     --acc-peak2 state2_peaks.bed \
     --acc-signal2 state2.bw \
     --direction 2-1 \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Train from two states with TSS background regions
-------------------------------------------------

.. code-block:: bash

   chrombert-tools region_activity_regression \
     --acc-peak1 state1_peaks.bed \
     --acc-signal1 state1.bw \
     --acc-peak2 state2_peaks.bed \
     --acc-signal2 state2.bw \
     --direction 2-1 \
     --include-tss-background \
     --tss-flank 10000 \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Train with replicates
---------------------

.. code-block:: bash

   chrombert-tools region_activity_regression \
     --acc-peak1 "s1_rep1.bed;s1_rep2.bed" \
     --acc-signal1 "s1_rep1.bw;s1_rep2.bw" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use chromosome-level splits
---------------------------

.. code-block:: bash

   chrombert-tools region_activity_regression \
     --acc-peak1 state1_peaks.bed \
     --acc-signal1 state1.bw \
     --train-chr "chr1;chr2;chr3" \
     --valid-chr "chr8;chr9" \
     --test-chr "chr18;chr19" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Predict only
------------

Use this mode when you already have a fine-tuned region activity checkpoint.

.. code-block:: bash

   chrombert-tools region_activity_regression \
     --ft-ckpt path/to/region_activity_finetuned.ckpt \
     --predict-file regions.tsv \
     --genome hg38 \
     --resolution 1kb \
     --odir output_predict

Run with Apptainer
------------------

Use ``--nv`` to enable GPU access.

.. code-block:: bash

   apptainer exec --nv /path/to/chrombert-tools.sif chrombert-tools region_activity_regression \
     --acc-peak1 state1_peaks.bed \
     --acc-signal1 state1.bw \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Accessibility inputs
--------------------

``--acc-peak1`` *(file path or semicolon-separated paths)*
   State-1 peak BED file(s). Required for training.

``--acc-peak2`` *(file path or semicolon-separated paths, optional)*
   State-2 peak BED file(s). Used in two-state analyses.

``--acc-signal1`` *(file path or semicolon-separated paths)*
   State-1 accessibility BigWig file(s). Required for training.

``--acc-signal2`` *(file path or semicolon-separated paths, optional)*
   State-2 accessibility BigWig file(s). Providing this option enables two-state mode.

Prediction inputs
-----------------

``--predict-file`` *(file path, optional)*
   Regions used for prediction.

   The file should contain at least:

   * ``chrom``
   * ``start``
   * ``end``
   * ``build_region_index``

   If a ``label`` column is included, it will be copied to the output as ``true_label``.

   When ``--predict-file`` is not provided after training, ChromBERT-tools predicts on the
   test split generated during dataset preparation.

``--ft-ckpt`` *(file path, optional)*
   Fine-tuned region activity checkpoint.

   When both ``--ft-ckpt`` and ``--predict-file`` are provided, ChromBERT-tools runs in
   predict-only mode and skips dataset preparation and training.

Label options
-------------

``--direction`` *(2-1 | 1-2, default: 2-1)*
   Direction of the two-state accessibility change.

``--subtract-reference-baseline`` *(flag)*
   Single-state mode only.

   By default, single-state labels subtract the packaged reference accessibility baseline.
   Add this flag to use raw ``log2(1 + state1 signal)`` instead.

``--include-tss-background`` *(flag)*
   Add TSS-centered background regions to the training set.

``--tss-flank`` *(int, default: 10000)*
   Flanking distance in base pairs around each TSS. This option is used only when
   ``--include-tss-background`` is set.

Training options
----------------

``--mode`` *(fast | full, default: fast)*
   Training mode.

   In ``fast`` mode, the training region set is downsampled to 20,000 regions when the
   full set is larger than 20,000. If the full set has 20,000 regions or fewer, all
   regions are used.

``--train-chr`` *(string, optional)*
   Semicolon-separated chromosomes used for training.

``--valid-chr`` *(string, optional)*
   Semicolon-separated chromosomes used for validation.

``--test-chr`` *(string, optional)*
   Semicolon-separated chromosomes used for testing.

   ``--train-chr`` and ``--valid-chr`` must be provided together. If chromosome-level
   splits are not provided, ChromBERT-tools uses a random train, validation, and test split.

Reference and output options
----------------------------

``--genome`` *(hg38 | mm10, default: hg38)*
   Reference genome.

``--resolution`` *(200bp | 1kb | 2kb | 4kb, default: 1kb)*
   ChromBERT bin resolution. For ``mm10``, only ``1kb`` is currently supported.

``--batch-size`` *(int, default: 4)*
   Batch size for training and prediction.

``--odir`` *(directory, default: ./output)*
   Output directory. It will be created automatically if needed.

``--chrombert-cache-dir`` *(directory, default: ~/.cache/chrombert/data)*
   Directory for ChromBERT reference files, model files, and cached data.

Required cache files
====================

The command uses the following ChromBERT cache files:

* ChromBERT reference region file
* ChromBERT HDF5 feature file
* pre-trained ChromBERT checkpoint
* mask matrix

Additional files may be required in specific modes:

* gene metadata file: required when ``--include-tss-background`` is used
* reference accessibility baseline: required for single-state training with default
  baseline subtraction

Outputs
=======

The following files are written under ``--odir``.

``dataset/``
   Created during training. It contains processed accessibility datasets and train,
   validation, and test splits.

``train/``
   Created during training. It contains model training outputs and the selected checkpoint.

``predict/model_input.tsv``
   Processed input table used for prediction.

``predict/predictions.csv``
   Main prediction output.

   The file contains region metadata, predicted values, and optionally true labels if the
   input contains a ``label`` column.

``model_config.json``
   Model configuration used for the run.

``dataset_config.json``
   Dataset configuration used for the run.

Predict-only mode
=================

Predict-only mode is used when both of the following are provided:

* ``--ft-ckpt``
* ``--predict-file``

In this mode, ChromBERT-tools does not require peak or BigWig files and does not train a
model. It loads the checkpoint and writes predictions directly to:

.. code-block:: text

   <odir>/predict/predictions.csv

Tips
====

1. Use ``--acc-peak1`` and ``--acc-signal1`` for single-state accessibility prediction.
2. Use ``--acc-signal2`` for two-state accessibility change prediction.
3. Use semicolons to provide replicate peak or BigWig files.
4. Use ``--include-tss-background`` to add TSS-centered regions, especially for transition
   analyses.
5. Use ``--ft-ckpt`` together with ``--predict-file`` for predict-only mode.
6. To see all options, run:

.. code-block:: bash

   chrombert-tools region_activity_regression -h

Tutorials
=========

.. toctree::
   :maxdepth: 1

   Python API example <../examples/api/region_activity_regression>