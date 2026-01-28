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

Generation of regulation-informed embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`commands/embed_cistrome`: Generate cistrome embeddings for specified regions
* :doc:`commands/embed_gene`: Generate gene embeddings
* :doc:`commands/embed_region`: Generate region embeddings for specified regions
* :doc:`commands/embed_regulator`: Generate regulator embeddings for specified regions
* :doc:`commands/embed_cell_cistrome`: Generate cell-type-specific cistrome embeddings for specified regions
* :doc:`commands/embed_cell_gene`: Generate cell-type-specific gene embeddings
* :doc:`commands/embed_cell_region`: Generate cell-type-specific region embeddings for specified regions
* :doc:`commands/embed_cell_regulator`: Generate cell-type-specific regulator embeddings for specified regions


Interpretation of regulation-informed embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* :doc:`commands/infer_ep`: Infer enhancer-promoter loops on specified regions
* :doc:`commands/infer_regulator_network`: Infer regulator-regulator networks on specified regions
* :doc:`commands/impute_cistrome`: Impute cistrome data on specified regions
* :doc:`commands/infer_cell_key_regulator`: Infer cell-type-specific key regulators on specified regions
* :doc:`commands/find_driver_in_transition`: Find driver factors in cell-state transitions
* :doc:`commands/find_context_specific_cofactor`: Find context-specific cofactors in different regions

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
