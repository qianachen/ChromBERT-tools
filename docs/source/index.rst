Welcome to ChromBERT-tools
==========================

**ChromBERT-tools** is a lightweight toolkit built on top of the
`ChromBERT <https://github.com/TongjiZhanglab/ChromBERT>`_ foundation model.
It helps users generate, model, and interpret context-specific regulatory
representations from their own genomic data through command-line interfaces
(CLIs) and Python APIs.

ChromBERT is a pre-trained foundation model designed to capture genome-wide
co-association patterns among transcription regulators and to learn
context-specific transcriptional regulatory networks.

.. image:: ../_static/figure1_new.png
   :alt: ChromBERT-tools framework
   :align: center

Key features
------------

* **Simple interface** — provides a single ``chrombert-tools`` entry point with composable subcommands.
* **Human and mouse support** — supports **hg38** and **mm10** genomes at multiple resolutions.
* **Modular workflows** — supports representation generation, predictive modeling, and regulatory interpretation.
* **Cell-type-specific analysis** — supports context-specific analyses using accessibility tracks or fine-tuned checkpoints.
* **CLI and Python API** — exposes core functions through both shell commands and importable Python APIs.

Core functionality
------------------

ChromBERT-tools is organized into three functional layers:

* **Representation generation** — generate context-specific representations for genomic regions or regulators.
* **Predictive modeling** — train or apply models for region classification, region activity prediction, and gene activity prediction.
* **Regulatory interpretation** — infer functional relationships among regions, regulators, and regulatory programs.

It also provides end-to-end application workflows for common biological tasks,
including cell-type-specific key regulator identification, driver regulator
identification during cell-state transitions, context-specific cofactor analysis,
and TF-binding prediction.

Getting started
---------------

1. Follow :doc:`installation` to install ChromBERT-tools using the recommended
   Apptainer image or from source.
2. Download the required ChromBERT model and annotation files using
   ``download-data``.
3. See :doc:`usage` for the full list of available commands and tutorials.

To list all available commands::

   chrombert-tools --help

To get help for a specific command::

   chrombert-tools <command> --help

User guide
----------

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

For questions or suggestions, please contact us at
`2211083@tongji.edu.cn <mailto:2211083@tongji.edu.cn>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`