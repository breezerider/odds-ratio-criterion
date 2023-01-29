#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import io
import os
import re
from os.path import dirname
from os.path import join

from setuptools import Extension
from setuptools import setup
from setuptools.dist import Distribution

try:
    import Cython
except ImportError:
    Cython = None

try:
    import numpy
except ImportError:
    numpy = None

# Enable code coverage for C code: we can't use CFLAGS=-coverage in tox.ini, since that may mess with compiling
# dependencies (e.g. numpy). Therefore we set SETUPPY_CFLAGS=-coverage in tox.ini and copy it to CFLAGS here (after
# deps have been safely installed).
if 'TOX_ENV_NAME' in os.environ and os.environ.get('SETUPPY_EXT_COVERAGE') == 'yes':
    CFLAGS = os.environ['CFLAGS'] = '-DCYTHON_TRACE=1'
    LFLAGS = os.environ['LFLAGS'] = ''
else:
    CFLAGS = ''
    LFLAGS = ''


class BinaryDistribution(Distribution):
    """Distribution which almost always forces a binary package with platform name"""
    def has_ext_modules(self):
        return super().has_ext_modules() or not os.environ.get('SETUPPY_ALLOW_PURE')


def read(*names, **kwargs):
    with io.open(join(dirname(__file__), *names), encoding=kwargs.get('encoding', 'utf8')) as fh:
        return fh.read()


setup(
    name='odds-ratio-criterion',
    version='0.1.0',
    license='new BSD',
    description='Goodness-of-fit criterion based on probability ratios for scikit-learn decision trees.',
    long_description='{}\n{}'.format(
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst')),
    ),
    author='Oleksandr Ostrenko',
    author_email='oleksandr.ostrenko@gmail.com',
    url='https://github.com/breezerider/odds-ratio-criterion',
    packages=['odds_ratio_criterion'],
    package_dir={'odds_ratio_criterion': 'src/odds_ratio_criterion'},
    py_modules=['odds_ratio_criterion.xnboost'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Development Status :: 4 - Beta",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    project_urls={
        'Documentation': 'https://odds-ratio-criterion.readthedocs.io/',
        'Changelog': 'https://odds-ratio-criterion.readthedocs.io/en/latest/changelog.html',
        'Issue Tracker': 'https://github.com/breezerider/odds-ratio-criterion/issues',
    },
    python_requires='>=3.8',
    install_requires=[
        'scikit-learn>=1.2.0',
        'numpy>=1.17.3'
    ],
    setup_requires=[
        'cython',
        'numpy'
    ],
    ext_modules=[
        Extension(
            'odds_ratio_criterion.odds_ratio_criterion',
            sources=['src/odds_ratio_criterion/_odds_ratio_criterion.pyx'],
            extra_compile_args=CFLAGS.split(),
            extra_link_args=LFLAGS.split(),
            include_dirs=[numpy.get_include()] if numpy else [],
        )
    ],
    distclass=BinaryDistribution,
)
