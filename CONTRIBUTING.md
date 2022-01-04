# Contributing to Iguanas

First of all, thank you for taking the time to contribute to the project! We love having you here :grinning:

## Table of Contents

- [Getting started](#getting-started)
- [Testing](#testing)
- [Submitting a change](#submitting-a-change)
- [Styleguide](#styleguide)
  - [Code](#code)
  - [Docstrings](#docstrings)
  - [Notebooks](#notebooks)
- [Raising a bug](#raising-a-bug)
- [Requesting a feature](#requesting-a-feature)
- [Code of Conduct](#code-of-conduct)
- [I need more help!](#i-need-more-help)

## Getting started

To begin contributing, you'll first need to create a virtual environment, which Iguanas will be installed into:

```bash
cd <path-for-iguanas-virtual-environment>
python3 -m venv iguanas_dev
```

Then activate the virtual environment:

```bash
source <path-to-iguanas-virtual-environment>/bin/activate
```

Now `cd` to the location where you want to store the Iguanas repo and clone it using the command:

```bash
git clone https://github.com/paypal/Iguanas.git
```

`cd` to the `Iguanas` folder and install Iguanas in editable mode using the command:

```bash
pip install -e .
```

Then install the `dev` extras:

```bash
pip install .[dev]
```

If contributing to a module that utilises Spark, you need to install the `spark` extras:

```bash
pip install .[spark]
```

Once Iguanas has been installed in the virtual environment, create a jupyter kernel using the command:

```bash
python -m ipykernel install --user --name iguanas_dev
```

## Testing

You can run all unit and notebook tests for the Spark and non-Spark modules using the commands:

```bash
pytest --nbmake --nbmake-kernel=iguanas_dev <path-to-Iguanas-repo>/iguanas
pytest --nbmake --nbmake-kernel=iguanas_dev <path-to-Iguanas-repo>/examples
```

To run only the unit tests of the non-Spark modules, use the command:

```bash
sh <path-to-Iguanas-repo>/run_unit_tests.sh
```

To run only the unit tests of the Spark modules, use the command:

```bash
sh <path-to-Iguanas-repo>/run_unit_tests_spark.sh
```

To run only the notebook tests of the non-Spark modules, use the command:

```bash
sh <path-to-Iguanas-repo>/run_nb_makes.sh iguanas_dev
```

To run only the notebook tests of the Spark modules, use the command:

```bash
sh <path-to-Iguanas-repo>/run_nb_makes_spark.sh
```

## Submitting a change

You can submit a change by raising a [pull-request](https://github.com/paypal/Iguanas/pulls) and assigning the reviewer as James Laidler.

Please ensure that, before raising a pull-request:

* Your code has been profiled thouroughly to ensure runtime is optimised.
* Unit tests are added/extended.
* Unit test coverage is >95%.
* Docstrings are added in the [numpy](https://numpydoc.readthedocs.io/en/latest/format.html) format, using the same style as the existing docstrings.
* An example notebook is added, in the same style as the existing notebooks.

## Styleguide

### Code 

Python code should follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) convention. It's recommended that you install [autopep8](https://pypi.org/project/autopep8/) before contributing, which will ensure that your code follows the PEP 8 convention.

### Docstrings

Docstrings should follow the [numpy](https://numpydoc.readthedocs.io/en/latest/format.html) format.

### Notebooks

Notebooks should follow the general structure:

* A brief introduction of the module, what it does and why it is used.
* Apply the module.
* A brief description of the outputs.

For example notebooks, see the [Examples](https://paypal.github.io/Iguanas/examples/index.html) section of the documentation.

## Raising a bug

You can raise a bug in the [Issues](https://github.com/paypal/Iguanas/issues) area of the repo.

## Requesting a feature

You can request a new feature in the [Issues](https://github.com/paypal/Iguanas/issues) area of the repo.

## Code of Conduct

You can find the Code of Conduct [here](https://github.com/paypal/Iguanas/blob/main/CODE_OF_CONDUCT.md).

## I need more help!

If you have any other queries or questions, feel free to contact James Laidler:

* [Email](james.a.laidler@gmail.com)
* [Linkedin](https://www.linkedin.com/in/james-laidler-430571a7)