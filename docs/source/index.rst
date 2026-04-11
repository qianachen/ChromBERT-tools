Welcome to ChromBERT-tools Documentation!
==========================================

**ChromBERT** is a pre-trained deep learning model designed to capture genome-wide co-association patterns of ~1,000 transcription regulators and to learn context-specific transcriptional regulatory networks (TRNs).

**ChromBERT-tools** is a lightweight toolkit designed to generate and interpret regulation-informed embeddings derived from ChromBERT, providing user-friendly command-line interfaces (CLIs) and Python APIs.

Features
--------

* **Easy-to-use CLI**: Simple command-line interface
* **Flexible**: Works with hg38 (human) and mm10 (mouse) genomes, and different resolutions
* **Comprehensive**: Tools for embedding, imputation, inference, and driver factor analysis
* **Cell-specific**: Support for cell-type specific analysis

ChromBERT-tools CLI
---------------------

Run ``chrombert-tools --help`` for the authoritative list of subcommands. The :doc:`usage` page groups RST references for each registered command.

Embeddings
^^^^^^^^^^

* :doc:`commands/embed_region`
* :doc:`commands/embed_regulator`

Supervised models & prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`commands/region_function_classification`
* :doc:`commands/region_activity_regression`
* :doc:`commands/gene_activity_repression`
* :doc:`commands/predict_cell_type_master_regulators`
* :doc:`commands/predict_transition_driver_regulators`
* :doc:`commands/predict_regulator_context_cofactors`
* :doc:`commands/predict_tf_binding_regions`

Interpretation & region–regulator analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`commands/interpret_region_region_interactions`
* :doc:`commands/interpret_regulator_regulator_interactions`
* :doc:`commands/interpret_regulator_effects_between_region_groups`

Legacy / tutorial RST pages (older example names)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`commands/embed_cistrome`
* :doc:`commands/embed_gene`
* :doc:`commands/embed_cell_cistrome`
* :doc:`commands/embed_cell_gene`
* :doc:`commands/embed_cell_region`
* :doc:`commands/embed_cell_regulator`
* :doc:`commands/infer_ep`
* :doc:`commands/infer_regulator_network`
* :doc:`commands/impute_cistrome`
* :doc:`commands/infer_cell_key_regulator`
* :doc:`commands/find_driver_in_transition`
* :doc:`commands/find_context_specific_cofactor`

Quick Start
-----------

Check out the :doc:`installation` section for setup instructions, and the :doc:`usage` section to learn how to use ChromBERT-tools.

.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   usage


Links
-----

* ChromBERT GitHub: https://github.com/TongjiZhanglab/ChromBERT
* ChromBERT-tools GitHub: https://github.com/TongjiZhanglab/ChromBERT-tools
* Documentation: https://chrombert-tools.readthedocs.io/

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
