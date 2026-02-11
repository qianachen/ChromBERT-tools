============
Installation
============

ChromBERT-tools depends on the ChromBERT runtime environment (packages + pre-trained model files). The **official
Singularity/Apptainer image is the easiest option**, and it **already includes ChromBERT-tools**.

This page describes the recommended setup:

1. Pull the ChromBERT container image (recommended)
2. Download ChromBERT pre-trained models and annotation files
3. (Optional) Update packages inside the container image
4. (Optional) Install ChromBERT-tools from source on the host


-------------------------------
1) ChromBERT environment (recommended)
-------------------------------

Installing Singularity/Apptainer
================================

First, install ``Apptainer`` (or ``Singularity``):

.. code-block:: bash

   conda install -c conda-forge apptainer


Pulling the official image (recommended)
========================================

Pull the image from ORAS:

.. code-block:: bash

   apptainer pull chrombert.sif oras://docker.io/chenqianqian515/chrombert:20260202


Optional: download the image from Google Drive:

- ``chrombert.sif`` (Google Drive): https://drive.google.com/file/d/14I-BQxrBNPwdZn-TKaG0Z8lpNiJlUd1f/view?usp=drive_link


Quick check
===========

.. code-block:: bash

   singularity exec /path/to/chrombert.sif python -c "import chrombert; print('hello chrombert')"  # may take some time on first run
   singularity exec /path/to/chrombert.sif chrombert-tools


-----------------------------------------
2) ChromBERT pre-trained models files
-----------------------------------------

ChromBERT requires pre-trained models and annotation data. These files will be downloaded to ``~/.cache/chrombert/data``.

Supported genomes and resolutions:

* **hg38** (Human): 200bp, 1kb, 2kb, 4kb
* **mm10** (Mouse): 1kb


Download the required files
===========================

.. code-block:: bash

   singularity exec /path/to/chrombert.sif chrombert_prepare_env --genome hg38 --resolution 1kb


Using a Hugging Face mirror
===========================

If Hugging Face is slow or unreachable, add ``--hf-endpoint``:

.. code-block:: bash

   singularity exec /path/to/chrombert.sif chrombert_prepare_env --genome hg38 --resolution 1kb --hf-endpoint <Hugging Face endpoint>


------------------------------------------------------------
3) (Optional) Update the Singularity image (ChromBERT-tools)
------------------------------------------------------------

The official image already includes ChromBERT-tools. If you need to install additional packages or update ChromBERT-tools
to the latest version, edit ``edit_image.def`` and rebuild a new image.

Example: rebuild a new image with updated ChromBERT-tools
=========================================================

.. code-block:: bash

   git clone https://github.com/TongjiZhanglab/ChromBERT-tools.git
   cd ChromBERT-tools
   apptainer build <new_image_name>.sif edit_image.def


------------------------------------------------
4) (Optional, host/dev) Install from source
------------------------------------------------

This option is intended for **development** (e.g., editing the code on the host, running tests, or contributing).
If you only want to *use* ChromBERT-tools, we recommend the container workflow above.

.. note::

   A host installation requires a working **ChromBERT environment** (runtime dependencies).
   Please follow the setup instructions in the `ChromBERT repository <https://github.com/TongjiZhanglab/ChromBERT>`_ first.


Installing ChromBERT-tools on the host
======================================

.. code-block:: bash

   git clone https://github.com/TongjiZhanglab/ChromBERT-tools.git
   cd ChromBERT-tools
   pip install .


Or install in editable (development) mode:

.. code-block:: bash

   pip install -e .


Verifying the host installation
===============================

.. code-block:: bash

   chrombert-tools


Note: downloading models/annotations is still required
=====================================================

Even with a host installation, you still need to download ChromBERT pre-trained models and annotations to
``~/.cache/chrombert/data``:

.. code-block:: bash

   chrombert_prepare_env --genome hg38 --resolution 1kb


If Hugging Face is slow or unreachable:

.. code-block:: bash

   chrombert_prepare_env --genome hg38 --resolution 1kb --hf-endpoint <Hugging Face endpoint>


Next Steps
==========

Once installation is complete, check out the :doc:`usage` section to learn how to use ChromBERT-tools for your analysis.
