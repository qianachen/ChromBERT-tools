==============================
region_function_classification
==============================

Train a ChromBERT classifier to assign genomic regions to functional classes, or use an
existing checkpoint to predict region classes.

The command supports three settings:

* **one-class classification**: provide one ``--function-bed``; a background class is
  generated automatically
* **binary classification**: provide two ``--function-bed`` files
* **multiclass classification**: provide three or more ``--function-bed`` files

Overview
========

``region_function_classification`` can be used in two ways:

* **training + prediction**: build a labeled dataset from BED files, train a classifier,
  and predict region classes
* **predict-only**: load an existing fine-tuned checkpoint and predict directly

The final predictions are written to:

.. code-block:: text

   <odir>/predict/predictions.csv

Basic Usage
===========

Binary classification
---------------------

.. code-block:: bash

   chrombert-tools region_function_classification \
     --function-bed enhancer.bed --function-name enhancer \
     --function-bed promoter.bed --function-name promoter \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Multiclass classification
-------------------------

.. code-block:: bash

   chrombert-tools region_function_classification \
     --function-bed enhancer.bed --function-name enhancer \
     --function-bed promoter.bed --function-name promoter \
     --function-bed silencer.bed --function-name silencer \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use multiple BED files for one class
------------------------------------

Multiple BED files can be combined into one class with semicolons.

Use ``--function-mode and`` to keep only regions shared by all BED files.

Use ``--function-mode or`` to use the union of all BED files.

.. code-block:: bash

   chrombert-tools region_function_classification \
     --function-bed "enh_rep1.bed;enh_rep2.bed" --function-mode and --function-name enhancer \
     --function-bed promoter.bed --function-name promoter \
     --function-bed silencer.bed --function-name silencer \
     --genome hg38 \
     --resolution 1kb \
     --odir output

One-class classification
------------------------

When only one class is provided, ChromBERT-tools automatically generates a background
class.

.. code-block:: bash

   chrombert-tools region_function_classification \
     --function-bed enhancer.bed \
     --function-name enhancer \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Use chromosome-level splits
---------------------------

Provide ``--train-chr`` and ``--valid-chr`` together. Regions from all other chromosomes
are used for testing.

.. code-block:: bash

   chrombert-tools region_function_classification \
     --function-bed enhancer.bed --function-name enhancer \
     --function-bed promoter.bed --function-name promoter \
     --train-chr "chr1;chr2;chr3" \
     --valid-chr "chr8;chr9" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Predict only
------------

Use this mode when you already have a fine-tuned region function classifier.

.. code-block:: bash

   chrombert-tools region_function_classification \
     --ft-ckpt path/to/region_function_finetuned.ckpt \
     --predict-file regions.tsv \
     --function-name enhancer \
     --function-name promoter \
     --genome hg38 \
     --resolution 1kb \
     --odir output_predict

Run with Apptainer
------------------

Use ``--nv`` to enable GPU access.

.. code-block:: bash

   apptainer exec --nv /path/to/chrombert-tools.sif chrombert-tools region_function_classification \
     --function-bed enhancer.bed \
     --function-name enhancer \
     --function-bed promoter.bed \
     --function-name promoter \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Class definition
----------------

``--function-bed`` *(file path or semicolon-separated paths)*
   BED file(s) defining one functional class.

   Repeat this option to provide multiple classes.

``--function-name`` *(string)*
   Name of each functional class.

   Class names are matched to ``--function-bed`` in the same order. In predict-only mode,
   ``--function-name`` is required so the output columns can be named correctly.

``--function-mode`` *(and | or, default: and)*
   How to combine multiple BED files within one class.

   ``and`` keeps shared regions.

   ``or`` keeps the union of regions.

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

   If this option is not provided after training, ChromBERT-tools predicts on the test
   split generated during dataset preparation.

``--ft-ckpt`` *(file path, optional)*
   Fine-tuned classifier checkpoint.

   When both ``--ft-ckpt`` and ``--predict-file`` are provided, ChromBERT-tools runs in
   predict-only mode and skips dataset preparation and training.

Training options
----------------

``--mode`` *(fast | full, default: fast)*
   Training mode.

   ``fast`` uses a balanced subset of regions from each class.

   ``full`` uses all labeled regions.

``--fast-max-total`` *(int, default: 20000)*
   Maximum total number of regions used in fast mode. The budget is divided evenly across
   classes.

``--ignore-regulator`` *(string, optional)*
   Regulators to mask during fine-tuning, separated by semicolons.

``--train-chr`` *(string, optional)*
   Semicolon-separated chromosomes used for training.

``--valid-chr`` *(string, optional)*
   Semicolon-separated chromosomes used for validation.

   ``--train-chr`` and ``--valid-chr`` must be provided together. If they are not
   provided, ChromBERT-tools uses a random train, validation, and test split.

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

Outputs
=======

The following files are written under ``--odir``.

``dataset/``
   Created during training. It contains the labeled dataset and train, validation, and
   test splits.

``train/``
   Created during training. It contains model training outputs and the selected checkpoint.

``predict/model_input.tsv``
   Processed input table used for prediction.

``predict/predictions.csv``
   Main prediction output.

   The output contains region metadata, predicted probabilities, predicted labels, and
   optionally true labels if the input contains a ``label`` column.

``model_config.json``
   Model configuration used for the run.

``dataset_config.json``
   Dataset configuration used for the run.

Prediction output
=================

For binary classification, ``predictions.csv`` contains:

* ``prob_<class_0>``
* ``predicted_label``
* ``predicted_name``

For multiclass classification, ``predictions.csv`` contains:

* ``prob_<class_i>`` for each class
* ``predicted_label``
* ``predicted_name``

Predict-only mode
=================

Predict-only mode is used when both of the following are provided:

* ``--ft-ckpt``
* ``--predict-file``

In this mode, ChromBERT-tools does not require training BED files and does not train a
model. It loads the checkpoint and writes predictions directly to:

.. code-block:: text

   <odir>/predict/predictions.csv

Required cache files
====================

The command uses the following ChromBERT cache files:

* ChromBERT reference region file
* ChromBERT HDF5 feature file
* metadata file
* pre-trained ChromBERT checkpoint
* mask matrix

If ``--ignore-regulator`` is used, the ChromBERT regulator list is also required.

Tips
====

1. Pair each ``--function-bed`` with its corresponding ``--function-name`` in the same
   order.
2. Use semicolons to combine multiple BED files into one class.
3. Use ``--function-mode and`` for intersections and ``--function-mode or`` for unions.
4. Use ``--mode fast`` for quick runs and ``--mode full`` to use all labeled regions.
5. Use ``--ft-ckpt`` together with ``--predict-file`` for predict-only mode.
6. To see all options, run:

.. code-block:: bash

   chrombert-tools region_function_classification -h

Tutorials
=========

.. toctree::
   :maxdepth: 1

   Python API example <../examples/api/region_function_classification>