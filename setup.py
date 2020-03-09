#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script"""

import os
from setuptools import setup, find_packages

if os.environ.get('READTHEDOCS') == 'True':
    requirements = []
else:
    requirements = ["numpy", "pandas", "scikit-learn",
                    "flask", "flask-restplus", "flask-cors",
                    "coloredlogs", "pyyaml",
                    "werkzeug<1.0"  # Needed for flask-restplus
                    ]

setup(author="Dih5",
      author_email='dihedralfive@gmail.com',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
      description='Package to implement Case-Based Reasoning systems',
      extras_require={
          "docs": ["nbsphinx", "sphinx-rtd-theme", "IPython"],
          "test": ["pytest"],
      },
      keywords=[],
      name='pycbr',
      packages=find_packages(include=['pycbr'], exclude=["demos", "tests", "docs"]),
      install_requires=requirements,
      url='https://github.com/dih5/pycbr',
      version='0.1.1',

      )
