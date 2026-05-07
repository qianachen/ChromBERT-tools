============
Installation
============

ChromBERT-tools is implemented in Python and requires **Python 3.9 or above**.
It uses **FlashAttention 2** for efficient model computation.

Two installation options are available:

* **Apptainer image (recommended)** — the official image already includes
  ChromBERT-tools and all runtime dependencies.
* **Source installation** — recommended for development or for running directly
  on the host system.

After installation, you must also :ref:`download the required ChromBERT model and
annotation files <download-resources>` before running any subcommand.


Installation options
====================

Option 1: Apptainer image (recommended)
---------------------------------------

This is the recommended installation method. The Apptainer image provides a
ready-to-use environment with ChromBERT-tools and its dependencies already installed.

Install Apptainer
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   conda install -c conda-forge apptainer

Pull the official image
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   apptainer pull chrombert-tools.sif oras://docker.io/chenqianqian515/chrombert-tools:20260505

Check the installation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   apptainer exec /path/to/chrombert-tools.sif chrombert-tools -h

.. note::

   If ``apptainer pull`` fails, you can download the image from Google Drive instead:
   `chrombert-tools.sif <https://drive.google.com/file/d/14I-BQxrBNPwdZn-TKaG0Z8lpNiJlUd1f/view?usp=drive_link>`_.

Optional: update or rebuild the image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you need to add packages or update existing ones, edit ``edit_image.def`` and
rebuild the image. The example below rebuilds the image with the latest
ChromBERT-tools source code:

.. code-block:: bash

   git clone https://github.com/TongjiZhanglab/ChromBERT-tools.git
   cd ChromBERT-tools
   apptainer build <new_image_name>.sif edit_image.def


Option 2: Install from source
-----------------------------

Source installation is useful for development or for users who want to run
ChromBERT-tools directly on the host system.

Create a conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   conda create -n ChromBERT python=3.9 -y
   conda activate ChromBERT

Install PyTorch
^^^^^^^^^^^^^^^

Install PyTorch with a CUDA version compatible with your system. ChromBERT-tools
requires PyTorch **< 2.4**.

Example for CUDA 12.1:

.. code-block:: bash

   pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
       --index-url https://download.pytorch.org/whl/cu121

Install FlashAttention 2
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install "flash-attn==2.4.*" --no-build-isolation

Install bedtools
^^^^^^^^^^^^^^^^

.. code-block:: bash

   conda install -c conda-forge -c bioconda bedtools

Install ChromBERT-tools
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/TongjiZhanglab/ChromBERT-tools.git
   cd ChromBERT-tools
   pip install .

Check the installation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   chrombert-tools -h

Optional: use a pre-built FlashAttention 2 wheel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If building ``flash-attn`` from source fails, you can install a pre-built wheel
that matches your Python, PyTorch, CUDA, and Linux environment.

Example:

.. code-block:: bash

   wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.3.post1/flash_attn-2.4.3.post1+cu122torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
   pip install /path/to/flash_attn-*.whl


.. _download-resources:

Download required resources
===========================

ChromBERT-tools requires ChromBERT pre-trained model files and annotation data.
These files are downloaded into ``~/.cache/chrombert/data`` using the
``download-data`` command.

Supported genomes and resolutions
---------------------------------

* **hg38** human: 200bp, 1kb, 2kb, 4kb
* **mm10** mouse: 1kb

Download resources with the Apptainer image
-------------------------------------------

.. code-block:: bash

   apptainer exec /path/to/chrombert-tools.sif download-data \
       --genome hg38 --resolution 1kb

If Hugging Face is slow or unreachable, specify a mirror endpoint:

.. code-block:: bash

   apptainer exec /path/to/chrombert-tools.sif download-data \
       --genome hg38 --resolution 1kb \
       --hf-endpoint <Hugging Face endpoint>

Download resources with a source installation
---------------------------------------------

.. code-block:: bash

   conda activate ChromBERT
   download-data --genome hg38 --resolution 1kb

Or with a Hugging Face mirror:

.. code-block:: bash

   download-data --genome hg38 --resolution 1kb \
       --hf-endpoint <Hugging Face endpoint>


Next steps
==========

After installation and resource download, see :doc:`usage` for the full list of
available ``chrombert-tools`` subcommands and tutorials.