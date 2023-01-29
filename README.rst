========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/odds-ratio-criterion/badge/?style=flat
    :target: https://odds-ratio-criterion.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/breezerider/odds-ratio-criterion/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/breezerider/odds-ratio-criterion/actions

.. |codecov| image:: https://codecov.io/gh/breezerider/odds-ratio-criterion/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://codecov.io/github/breezerider/odds-ratio-criterion

.. |version| image:: https://img.shields.io/badge/test.pypi-v0.1.0-green?style=flat
    :alt: PyPI Package latest release
    :target: https://test.pypi.org/project/odds-ratio-criterion

.. |wheel| image:: https://img.shields.io/badge/wheel-no-red?style=flat
    :alt: PyPI Wheel
    :target: https://test.pypi.org/project/odds-ratio-criterion

.. |supported-versions| image:: https://img.shields.io/badge/python-3.10-blue?style=flat
    :alt: Supported versions
    :target: https://test.pypi.org/project/odds-ratio-criterion

.. |supported-implementations| image:: https://img.shields.io/badge/implementation-cpython-blue?style=flat
    :alt: Supported implementations
    :target: https://test.pypi.org/project/odds-ratio-criterion

.. |commits-since| image:: https://img.shields.io/github/commits-since/breezerider/odds-ratio-criterion/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/breezerider/odds-ratio-criterion/compare/v0.1.0...main



.. end-badges

Goodness-of-fit criterion based on probability ratios for scikit-learn decision trees.

* Free software: BSD license

Installation
============

::

    pip install odds-ratio-criterion

You can also install the in-development version with::

    pip install https://github.com/breezerider/odds-ratio-criterion/archive/main.zip


Documentation
=============


https://odds-ratio-criterion.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
