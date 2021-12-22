.. _installation.index:

=============
Installation
=============

Prerequisites
=============

* Python 3.7 or above.

Creating a virtual environment
==============================

It's always good practise to create a virtual environment for installing Iguanas. Once created, you need to activate the virtual environment before installing Iguanas.

This process is outlined below, for both *venv* and *conda*:

Using venv
----------

.. code-block:: bash

   # Create VE
   cd <path-for-iguanas-virtual-environment>
   python3 -m venv iguanas_venv
   # Activate VE
   source <path-to-iguanas-virtual-environment>/bin/activate


Using a conda environment
-------------------------

.. code-block:: bash

   # Create VE
   conda create --name iguanas_venv python=3
   # Activate VE
   source activate iguanas_venv


Installing from PyPi or conda-forge repositories
================================================

With the virtual environment activated, run:

.. code-block:: bash
   
   pip install iguanas

Optional - install Spark requirements
-------------------------------------

To install the packages required to run Iguanas in Spark, you will also need to run the following command:

`bash` shell:

.. code-block:: bash

   pip install iguanas[spark]

`zsh` shell:

.. code-block:: bash

   pip install iguanas\[spark\]

Installing from Github
======================

Follow the instructions below to install *Iguanas* from Github:

1. Download the iguanas repository
------------------------------------

You can download the `iguanas` repository using one of the following methods:

* Clone the repository using git
* Download the repository via github

1a. Clone the repository using git
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To clone the `iguanas`, use the following command:

.. code-block:: bash

   cd <path-for-iguanas-repo>
   git clone https://github.com/paypal/Iguanas.git


1b. Download the repository via github
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To download the repo, go to <https://github.com/paypal/Iguanas>, click the green `code` button in the top right hand corner and then click `Download ZIP`. This will download the repo as a ZIP folder, which can then be extracted.

2. Install Iguanas
------------------

With the `iguanas_venv` activated, install *Iguanas* using the following command:

.. code-block:: bash

   pip install <path-to-iguanas-repo>/.


Optional - install Spark requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the packages required to run Iguanas in Spark, you will also need to run the following command:

`bash` shell:

.. code-block:: bash

   pip install <path-to-iguanas-repo>/.[spark]

`zsh` shell:

.. code-block:: bash

   pip install <path-to-iguanas-repo>/.\[spark\]


Creating a Jupyter kernel
=========================

To use Iguanas in Jupyter, you will first need to create a kernel using `iguanas_venv`. With the virtual environment activated, run the following command:

.. code-block:: bash

   python -m ipykernel install --user --name iguanas


This will create the Jupyter kernel `iguanas` - use this kernel when running Iguanas in a Jupyter notebook.