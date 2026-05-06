=====
Usage
=====

ChromBERT-tools supports two ways to run:

* **Command-line interface (CLI)** — run from a terminal as ``chrombert-tools <command>``.
* **Python API** — call functions directly from Python code (``import chrombert_tools``).

For runnable examples, see the Jupyter notebooks in
`examples/cli/ <https://github.com/TongjiZhanglab/ChromBERT-tools/tree/main/examples/cli>`_
and
`examples/api/ <https://github.com/TongjiZhanglab/ChromBERT-tools/tree/main/examples/api>`_.

Quick Start
===========

List all available subcommands:

.. code-block:: bash

   chrombert-tools --help

Get help for a specific subcommand:

.. code-block:: bash

   chrombert-tools <command> --help

Run the example notebooks (with GPU support inside the Apptainer image):

.. code-block:: bash

   cd ChromBERT-tools/examples/
   apptainer exec --nv /path/to/chrombert-tools.sif jupyter-notebook

.. _cli-api-reference:

CLI / API Reference
===================

ChromBERT-tools is organized into **three functional layers** plus a set of
**end-to-end application commands**, mirroring the layout in the README. Each subcommand below
is callable both from the CLI (``chrombert-tools <command>``) and from the Python API (``chrombert_tools.<command>``).

1) Generation of context-specific regulatory representations
------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - :doc:`commands/embed_region`
     - Extract embeddings for specified genomic regions or promoter-centered gene regions.
   * - :doc:`commands/embed_regulator`
     - Extract regulator embeddings for specified regulators across specified genomic regions.

.. toctree::
   :maxdepth: 1
   :hidden:

   commands/embed_region
   commands/embed_regulator

2) Predictive modeling of context-specific regulatory representations
---------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - :doc:`commands/region_function_classification`
     - Classify genomic regions into functional classes.
   * - :doc:`commands/region_activity_regression`
     - Predict quantitative region activity, such as accessibility or activity fold change.
   * - :doc:`commands/gene_activity_regression`
     - Predict gene expression or expression fold change from TSS-centered regulatory context.

.. toctree::
   :maxdepth: 1
   :hidden:

   commands/region_function_classification
   commands/region_activity_regression
   commands/gene_activity_regression

3) Interpretation of context-specific regulatory representations
----------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - :doc:`commands/interpret_region_region_interactions`
     - Identify functionally similar genomic regions.
   * - :doc:`commands/interpret_regulator_regulator_interactions`
     - Identify potentially cooperative regulators.
   * - :doc:`commands/interpret_regulator_effects_between_region_groups`
     - Compare regulator effects between region groups.

.. toctree::
   :maxdepth: 1
   :hidden:

   commands/interpret_region_region_interactions
   commands/interpret_regulator_regulator_interactions
   commands/interpret_regulator_effects_between_region_groups

End-to-end application commands
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - :doc:`commands/predict_cell_type_master_regulators`
     - Infer cell-type-specific key regulators.
   * - :doc:`commands/predict_transition_driver_regulators`
     - Identify driver regulators in cell-state transitions.
   * - :doc:`commands/predict_regulator_context_cofactors`
     - Identify context-specific cofactors.
   * - :doc:`commands/predict_tf_binding_regions`
     - Predict TF-binding regions.

.. toctree::
   :maxdepth: 1
   :hidden:

   commands/predict_cell_type_master_regulators
   commands/predict_transition_driver_regulators
   commands/predict_regulator_context_cofactors
   commands/predict_tf_binding_regions

Tutorials (Jupyter notebooks)
=============================

Each command page ships a **Tutorials** section at the bottom that links directly to
the matching CLI and / or Python API notebook from the ``examples/`` directory — open
any command above to see them.

You can also browse the raw notebooks on GitHub:

* CLI examples: https://github.com/TongjiZhanglab/ChromBERT-tools/tree/main/examples/cli
* Python API examples: https://github.com/TongjiZhanglab/ChromBERT-tools/tree/main/examples/api

The following notebooks are not tied to a single command and live here:

.. toctree::
   :maxdepth: 1

   Run with Apptainer / Singularity <examples/cli/singularity_use>

Next Steps
==========

* Pick the per-command page for full options, inputs, outputs, tips, and tutorials.
* Combine subcommands to build custom analysis pipelines (e.g. ``embed_regulator`` →
  ``interpret_regulator_regulator_interactions`` → ``predict_cell_type_master_regulators``).
