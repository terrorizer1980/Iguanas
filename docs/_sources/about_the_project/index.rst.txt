About the project
=================

Iguanas was started as a project by James Laidler when working at Simility (which is now part of PayPal). It was developed to reduce the time the team spent on building and optimising rules for Simility's clients. 

It has now grown into a fully-fledged Python package for developing a Rules-Based System. However, it is still being actively developed, and contributors are encouraged to get involved!

Our vision
----------

**Our vision is to make Iguanas the go-to Python package developing a deployment-ready Rules-Based System (RBS) for all use cases.**

Authors
-------

* `James Laidler <https://githubmemory.com/index.php/@lamesjaidler>`_

Contributing
------------

If you'd like to contribute to Iguanas, you can do so by raising a pull-request and assigning the reviewer as James Laidler. 

Please ensure that, when contributing:

* Code is profiled thouroughly first to ensure runtime is optimised.
* Unit tests are added/extended.
* Unit test coverage is >95%.
* Docstrings are added, in the same style as the existing docstrings.
* An example notebook is added, in the same style as the existing notebooks.

Also make sure that before contrbuting, you install the `dev` extras:

.. code-block:: bash
   
   pip install iguanas.\[dev\]

Roadmap
-------

By the end of H1 2022, we aim to have the following completed:

* Build on the existing Iguanas pipeline module to cover more complex workflows (i.e. parallel pipelines).
* Make all modules in Iguanas Spark-ready.
* Runtime and memory improvements.
