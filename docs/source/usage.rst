=====
Usage
=====

Welcome to the ChromBERT-tools usage guide. This section provides documentation for all command-line tools.

Quick Start
===========

To see all available commands:

.. code-block:: bash

   chrombert-tools --help

To get help for a specific command:

.. code-block:: bash

   chrombert-tools <command> --help

CLI Reference
=============

``chrombert-tools`` subcommands (see ``chrombert-tools --help``) are documented below. Additional pages under **Legacy / tutorial command names** keep older notebook titles; where the CLI name differs, the page notes the current subcommand.

Embeddings
----------

.. toctree::
   :maxdepth: 1

   commands/embed_region
   commands/embed_regulator

Supervised models & prediction
------------------------------

.. toctree::
   :maxdepth: 1

   commands/region_function_classification
   commands/region_activity_regression
   commands/gene_activity_repression
   commands/predict_cell_type_master_regulators
   commands/predict_transition_driver_regulators
   commands/predict_regulator_context_cofactors
   commands/predict_tf_binding_regions

Interpretation & region–regulator analysis
------------------------------------------

.. toctree::
   :maxdepth: 1

   commands/interpret_region_region_interactions
   commands/interpret_regulator_regulator_interactions
   commands/interpret_regulator_effects_between_region_groups

Legacy / tutorial command names
-------------------------------

These pages describe workflows that may use **older example names** in notebooks; prefer the subcommand names in the sections above when calling ``chrombert-tools``.

.. toctree::
   :maxdepth: 1

   commands/embed_gene
   commands/embed_cistrome
   commands/embed_cell_gene
   commands/embed_cell_region
   commands/embed_cell_regulator
   commands/embed_cell_cistrome
   commands/infer_ep
   commands/infer_regulator_network
   commands/impute_cistrome
   commands/infer_cell_key_regulator
   commands/find_driver_in_transition
   commands/find_context_specific_cofactor

Notebooks (CLI examples)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   Extract general embeddings <examples/cli/embed>
   Cell-type-specific Embedding <examples/cli/embed_cell_specific>

Interpretation & imputation (example notebooks)
-----------------------------------------------

.. toctree::
   :maxdepth: 1

   Infer enhancer-promoter loops <examples/cli/infer_ep>
   Infer regulator-regulator networks <examples/cli/infer_regulator_network>
   Impute cistromes <examples/cli/impute_cistrome>
   Infer cell-type-specific key regulators <examples/cli/infer_cell_key_regulator>
   Find driver factors in cell-state transitions <examples/cli/find_driver_in_transition>
   Find context-specific cofactors in different regions <examples/cli/find_context_specific_cofactor>

General Notebooks
-----------------

Workflows and examples for running ChromBERT-tools in a Singularity container, including embedding, regulator-regulator network inference etc.

.. toctree::
   :maxdepth: 1

   Run ChromBERT-tools with Singularity <examples/cli/singularity_use>


API Reference
=============

In addition to CLI commands, you can now call ChromBERT-tools directly in Python. It currently supports tasks that do not require fine-tuning:

Generation of regulation-informed embeddings API
-----------------------------------------------

.. toctree::
   :maxdepth: 1

   Extract general embeddings from the pre-trained ChromBERT model <examples/api/embed>


Interpretation of regulation-informed embeddings API
-----------------------------------------

.. toctree::
   :maxdepth: 1

   Infer regulator-regulator networks <examples/api/infer_regulator_network>
   Infer enhancer-promoter loops <examples/api/infer_ep>
   Impute cistromes <examples/api/impute_cistrome>

Next Steps
==========

* Explore specific command documentation for detailed usage.
