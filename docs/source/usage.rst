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

Embedding CLI
-------------

Extract general embeddings from the pre-trained ChromBERT model:

.. toctree::
   :maxdepth: 1

   commands/embed_gene
   commands/embed_region
   commands/embed_regulator
   commands/embed_cistrome

**Example Notebooks:**

* `embed.ipynb <../../examples/cli/embed.ipynb>`_ - Complete embedding workflow examples
* `singularity_use.ipynb <../../examples/cli/singularity_use.ipynb>`_ - Singularity container usage

Cell-type-specific Embedding CLI
--------------------------------

Fine-tune ChromBERT and extract cell-type-specific embeddings:

.. toctree::
   :maxdepth: 1

   commands/embed_cell_gene
   commands/embed_cell_region
   commands/embed_cell_regulator
   commands/embed_cell_cistrome

**Example Notebooks:**

* `embed_cell_specific.ipynb <../../examples/cli/embed_cell_specific.ipynb>`_

TRN Inference CLI
-----------------

Infer transcriptional regulatory networks:

.. toctree::
   :maxdepth: 1

   commands/infer_trn
   commands/infer_cell_trn

**Example Notebooks:**

* `infer_trn.ipynb <../../examples/cli/infer_trn.ipynb>`_
* `infer_cell_trn.ipynb <../../examples/cli/infer_cell_trn.ipynb>`_

Imputation CLI
--------------

Impute missing cistrome data:

.. toctree::
   :maxdepth: 1

   commands/impute_cistrome

**Example Notebooks:**

* `impute_cistrome.ipynb <../../examples/cli/impute_cistrome.ipynb>`_

Driver Factor CLI
-----------------

Identify key regulatory factors:

.. toctree::
   :maxdepth: 1

   commands/find_driver_in_transition
   commands/find_driver_in_dual_region

**Example Notebooks:**

* `find_driver_in_transition.ipynb <../../examples/cli/find_driver_in_transition.ipynb>`_
* `find_driver_in_dual_region.ipynb <../../examples/cli/find_driver_in_dual_region.ipynb>`_

API Reference
=============

In addition to CLI commands, you can now call ChromBERT-tools directly in Python. It currently supports tasks that do not require fine-tuning:

Embedding API
-------------

Extract general embeddings from the pre-trained ChromBERT model:

**Example Notebooks:**

* `embed.ipynb <../../examples/api/embed.ipynb>`_ - Complete embedding workflow examples


TRN Inference API
-----------------

Infer transcriptional regulatory networks:

**Example Notebooks:**

* `infer_trn.ipynb <../../examples/api/infer_trn.ipynb>`_

Imputation API
--------------

Impute missing cistrome data:

**Example Notebooks:**

* `impute_cistrome.ipynb <../../examples/api/impute_cistrome.ipynb>`_

Next Steps
==========

* Explore specific command documentation for detailed usage.
