Welcome to ChromBERT-tools Documentation!
==========================================

**ChromBERT** is a pre-trained foundation model designed to capture genome-wide co-association
patterns of ~1,000 transcription regulators and to learn context-specific transcriptional
regulatory networks (TRNs).
See `ChromBERT on GitHub <https://github.com/TongjiZhanglab/ChromBERT>`_.

**ChromBERT-tools** is a lightweight toolkit built upon ChromBERT that operationalizes
context-specific regulatory representations for user data through modular command-line
interfaces (CLIs) and Python APIs, organized into **three functional layers**:

* **Representation generation** — derive regulation-informed embeddings for genomic regions or regulators.
* **Predictive modeling** — train / load supervised models on top of those embeddings.
* **Regulatory interpretation** — analyze region–region, regulator–regulator, and regulator-effect relationships.

.. image:: _static/figure1_new.png
   :alt: ChromBERT-tools framework
   :align: center

Features
--------

* **Easy-to-use CLI** — a single ``chrombert-tools`` entry point with composable subcommands.
* **Flexible** — works with **hg38** (human) and **mm10** (mouse) genomes at multiple resolutions.
* **Comprehensive** — covers embedding, supervised modeling, interpretation, and end-to-end driver-factor analysis.
* **Cell-specific** — supports cell-type-specific analysis via accessibility tracks or fine-tuned checkpoints.
* **CLI + Python API** — every core capability is callable both as a shell command and as an ``import``-able function.

Quick Start
-----------

1. Follow :doc:`installation` to set up the Apptainer image (recommended) or install from source.
2. Download the ChromBERT pre-trained model and annotation files via ``download-data``.
3. Browse :doc:`usage` for a categorized reference of all subcommands and tutorials.

To list all available subcommands::

   chrombert-tools --help

To get help for a specific subcommand::

   chrombert-tools <command> --help

CLI / API Reference
-------------------

The :doc:`usage` page groups each registered command by functional layer. Direct links
to per-command pages are also provided below.

1) Generation of context-specific regulatory representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`commands/embed_region` — extract embeddings for specified genomic regions or promoter-centered gene regions.
* :doc:`commands/embed_regulator` — extract regulator embeddings for specified regulators across specified genomic regions.

2) Predictive modeling of context-specific regulatory representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`commands/region_function_classification` — classify genomic regions into functional classes.
* :doc:`commands/region_activity_regression` — predict quantitative region activity (e.g. accessibility or activity fold change).
* :doc:`commands/gene_activity_regression` — predict gene expression or expression fold change from TSS-centered regulatory context.

3) Interpretation of context-specific regulatory representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`commands/interpret_region_region_interactions` — identify functionally similar genomic regions.
* :doc:`commands/interpret_regulator_regulator_interactions` — identify potentially cooperative regulators.
* :doc:`commands/interpret_regulator_effects_between_region_groups` — compare regulator effects between region groups.

End-to-end application commands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`commands/predict_cell_type_master_regulators` — infer cell-type-specific key regulators.
* :doc:`commands/predict_transition_driver_regulators` — identify driver regulators in cell-state transitions.
* :doc:`commands/predict_regulator_context_cofactors` — identify context-specific cofactors.
* :doc:`commands/predict_tf_binding_regions` — predict TF-binding regions.

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

Contact
-------

For questions or suggestions, please contact us at `2211083@tongji.edu.cn <mailto:2211083@tongji.edu.cn>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
