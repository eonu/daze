#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import print_function
from setuptools import setup, find_packages

python_requires = '>=3.5,<3.10'
setup_requires = [
    'Cython', # 'Cython>=0.28.5',
    'numpy', # 'numpy>=1.17,<2',
    'scipy' # 'scipy>=1.3,<2'
]
install_requires = [
    'numpy', # 'numpy>=1.17,<2',
    # 'scipy', # 'scipy>=1.3,<2',
    'scikit-learn', # 'scikit-learn>=0.22,<1',
    'matplotlib'
]

VERSION = '0.1.1'

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name = 'daze',
    version = VERSION,
    author = 'Edwin Onuonga',
    author_email = 'ed@eonu.net',
    description = 'Better multi-class confusion matrix plots for Scikit-Learn, incorporating per-class and overall evaluation measures.',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/eonu/daze',
    project_urls = {
        'Documentation': 'https://daze.readthedocs.io/en/latest',
        'Bug Tracker': 'https://github.com/eonu/daze/issues',
        'Source Code': 'https://github.com/eonu/daze'
    },
    license = 'MIT',
    package_dir = {'': 'lib'},
    packages = find_packages(where='lib'),
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Natural Language :: English'
    ],
    python_requires = python_requires,
    setup_requires = setup_requires,
    install_requires = install_requires
)