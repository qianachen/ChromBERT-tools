===================================
predict_regulator_context_cofactors
===================================

Identify context-specific cofactors for a regulator of interest.

This command compares how the co-association pattern of a focal regulator changes across
different functional region classes, such as active enhancers versus poised enhancers.

Overview
========

``predict_regulator_context_cofactors`` is useful for regulators that may work with
different cofactors in different genomic contexts.

The command:

1. builds a labeled region dataset from functional BED files
2. trains a region function classifier, or loads an existing checkpoint
3. generates regulator embeddings for each region class
4. compares the focal regulator's cofactor patterns between region classes
5. reports candidate context-specific cofactors

For multiple region classes, all class pairs are compared.

Basic Usage
===========

Compare two region classes
--------------------------

.. code-block:: bash

   chrombert-tools predict_regulator_context_cofactors \
     --function-bed active_enh.bed --function-name active_enh \
     --function-bed poised_enh.bed --function-name poised_enh \
     --dual-regulator "CTCF;BRD4" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Compare multiple region classes
-------------------------------

With three classes, ChromBERT-tools compares all three class pairs.

.. code-block:: bash

   chrombert-tools predict_regulator_context_cofactors \
     --function-bed enhancer.bed --function-name enhancer \
     --function-bed promoter.bed --function-name promoter \
     --function-bed silencer.bed --function-name silencer \
     --dual-regulator "EZH2" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Reuse an existing checkpoint
----------------------------

Use ``--ft-ckpt`` to skip classifier fine-tuning.

.. code-block:: bash

   chrombert-tools predict_regulator_context_cofactors \
     --function-bed active_enh.bed --function-name active_enh \
     --function-bed poised_enh.bed --function-name poised_enh \
     --dual-regulator "CTCF;BRD4" \
     --ft-ckpt path/to/region_function_finetuned.ckpt \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Run with Apptainer
------------------

Use ``--nv`` to enable GPU access.

.. code-block:: bash

   apptainer exec --nv /path/to/chrombert-tools.sif chrombert-tools predict_regulator_context_cofactors \
     --function-bed active_enh.bed \
     --function-name active_enh \
     --function-bed poised_enh.bed \
     --function-name poised_enh \
     --dual-regulator "CTCF;BRD4" \
     --genome hg38 \
     --resolution 1kb \
     --odir output

Parameters
==========

Class definition
----------------

``--function-bed`` *(file path or semicolon-separated paths)*
   BED file(s) defining one functional region class.

   Repeat this option to provide multiple classes. At least two classes are needed for
   context comparison. If only one class is provided, a background class is generated
   automatically.

``--function-name`` *(string, optional)*
   Name of each functional region class.

   Class names are used in output directory names and make results easier to read.

``--function-mode`` *(and | or, default: and)*
   How to combine multiple BED files within one class.

   ``and`` keeps shared regions.

   ``or`` keeps the union of regions.

Focal regulator
---------------

``--dual-regulator`` *(string, required)*
   Regulators of interest, separated by semicolons. For example:

   ``"EZH2;BRD4"``

``--ignore-regulator`` *(string, optional)*
   Regulators to mask during fine-tuning, separated by semicolons.

Cofactor detection options
--------------------------

``--threshold`` *(float, default: 0.1)*
   Minimum difference in regulator similarity between two classes.

   Lower values keep more candidate cofactors. Higher values produce a shorter and more
   stringent candidate list.

``--quantile`` *(float, default: 0.95)*
   Quantile used to build the regulator similarity graph for subnetwork visualization.

   This option affects the PDF subnetwork figure, not the candidate cofactor table.

Training options
----------------

``--ft-ckpt`` *(file path, optional)*
   Existing fine-tuned region function classifier.

   If provided, ChromBERT-tools skips classifier fine-tuning.

``--mode`` *(fast | full, default: fast)*
   Training mode.

   ``fast`` uses a subset of regions for quicker training.

   ``full`` uses all eligible regions.

``--batch-size`` *(int, default: 4)*
   Batch size used for training and embedding generation.

Reference and output options
----------------------------

``--genome`` *(hg38 | mm10, default: hg38)*
   Reference genome.

``--resolution`` *(200bp | 1kb | 2kb | 4kb, default: 1kb)*
   ChromBERT bin resolution. For ``mm10``, only ``1kb`` is currently supported.

``--odir`` *(directory, default: ./output)*
   Output directory. It will be created automatically if needed.

``--chrombert-cache-dir`` *(directory, default: ~/.cache/chrombert/data)*
   Directory for ChromBERT reference files, model files, and cached data.

Required cache files
====================

The command uses the following ChromBERT cache files:

* ChromBERT reference region file
* ChromBERT HDF5 feature file
* metadata file
* ChromBERT regulator list
* pre-trained ChromBERT checkpoint
* mask matrix

Outputs
=======

The following files are written under ``<odir>``.

``dataset/``
   Labeled region dataset and train, validation, and test splits.

``train/``
   Classifier fine-tuning outputs and checkpoints.

   This directory is created only when ``--ft-ckpt`` is not provided.

``emb/region<i>/``
   Mean regulator embeddings for region class ``i``.

``results/<i>_<j>_<name_i>_vs_<name_j>/``
   Results for each pair of region classes.

   Main files include:

   ``dual_regulator_<reg>_candidate_cofactors.csv``
      Candidate context-specific cofactors for the focal regulator.

   ``dual_regulator_<reg>_subnetwork.pdf``
      Subnetwork figure showing the focal regulator and its context-specific partners.

Interpretation
==============

Candidate cofactors are regulators whose similarity to the focal regulator changes between
two region classes.

A larger similarity difference suggests stronger context specificity.

For example, a cofactor may be strongly associated with the focal regulator in active
enhancers but not in poised enhancers.

Tips
====

1. Use biologically meaningful region classes, such as active versus poised enhancers or
   gained versus lost peaks.
2. Use ``--function-name`` to make pairwise result directories easy to understand.
3. Use ``--threshold`` to control how many candidate cofactors are reported.
4. Use ``--quantile`` to control the density of the subnetwork figure.
5. Use ``--ft-ckpt`` to reuse a trained classifier and test different focal regulators.
6. To see all options, run:

.. code-block:: bash

   chrombert-tools predict_regulator_context_cofactors -h

Tutorials
=========

.. toctree::
   :maxdepth: 1

   CLI example <../examples/cli/predict_regulator_context_cofactors>