============
Installation
============

ChromBERT-tools is implemented in Python and requires **Python 3.9 or above**. It uses
**FlashAttention 2** for efficient model computation. Two installation options are supported:

1. **Install with an Apptainer image (recommended)** — the official image already includes
   ChromBERT-tools and all runtime dependencies.
2. **Install from source** — for development or when running directly on the host.

After installation, you must :ref:`download the ChromBERT pre-trained model and annotation
files <download-resources>` before running any subcommand.


Option 1: Install with an Apptainer image (recommended)
=======================================================

Install Apptainer
-----------------

.. code-block:: bash

   conda install -c conda-forge apptainer

Pull the official image
-----------------------

.. code-block:: bash

   apptainer pull chrombert-tools.sif oras://docker.io/chenqianqian515/chrombert-tools:20260505

Check the installation:

.. code-block:: bash

   apptainer exec /path/to/chrombert-tools.sif chrombert-tools -h

.. note::

   If ``apptainer pull`` fails, you can download the image from Google Drive instead:
   `chrombert-tools.sif <https://drive.google.com/file/d/14I-BQxrBNPwdZn-TKaG0Z8lpNiJlUd1f/view?usp=drive_link>`_.

(Optional) Update the Apptainer image
-------------------------------------

If you need to add new packages or update existing ones, edit ``edit_image.def`` and rebuild
the image. The example below rebuilds the image with the latest ChromBERT-tools:

.. code-block:: bash

   git clone https://github.com/TongjiZhanglab/ChromBERT-tools.git
   cd ChromBERT-tools
   apptainer build <new_image_name>.sif edit_image.def


Option 2: Install from source
=============================

Create a conda environment and install PyTorch (< 2.4) with a CUDA build that matches your
system, then install FlashAttention 2, bedtools, and ChromBERT-tools.

.. code-block:: bash

   # Create and activate a conda environment.
   conda create -n ChromBERT python=3.9 -y
   conda activate ChromBERT

   # Install PyTorch (< 2.4) with a CUDA version compatible with your system.
   # Example for CUDA 12.1:
   pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
       --index-url https://download.pytorch.org/whl/cu121

   # Install FlashAttention 2.
   pip install "flash-attn==2.4.*" --no-build-isolation

   # Install bedtools.
   conda install -c conda-forge -c bioconda bedtools

   git clone https://github.com/TongjiZhanglab/ChromBERT-tools.git
   cd ChromBERT-tools
   pip install .

   # Check the installation.
   chrombert-tools -h

(Optional) Use a pre-built FlashAttention 2 wheel
-------------------------------------------------

If building ``flash-attn`` from source fails, download a pre-built wheel that matches your
Python, PyTorch, CUDA, and Linux environment, then install it directly:

.. code-block:: bash

   wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.3.post1/flash_attn-2.4.3.post1+cu122torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
   pip install /path/to/flash_attn-*.whl  # Replace with your downloaded wheel.


.. _download-resources:

Download required resources
===========================

ChromBERT requires pre-trained model files and annotation data. These are downloaded into
``~/.cache/chrombert/data`` by the ``download-data`` command.

Supported genomes and resolutions:

* **hg38** (Human): 200bp, 1kb, 2kb, 4kb
* **mm10** (Mouse): 1kb

With the Apptainer image
------------------------

.. code-block:: bash

   apptainer exec /path/to/chrombert-tools.sif download-data \
       --genome hg38 --resolution 1kb

If Hugging Face is slow or unreachable, specify a mirror endpoint:

.. code-block:: bash

   apptainer exec /path/to/chrombert-tools.sif download-data \
       --genome hg38 --resolution 1kb --hf-endpoint <Hugging Face endpoint>

With a source install
---------------------

.. code-block:: bash

   conda activate ChromBERT
   download-data --genome hg38 --resolution 1kb

Or with a Hugging Face mirror:

.. code-block:: bash

   download-data --genome hg38 --resolution 1kb --hf-endpoint <Hugging Face endpoint>


Next Steps
==========

Once installation and resource download are complete, head to :doc:`usage` to learn how to
run each ``chrombert-tools`` subcommand from the CLI or call the Python API.
