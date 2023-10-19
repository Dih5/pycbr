#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script"""

import os
from setuptools import setup, find_packages

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()
    
if os.environ.get('READTHEDOCS') == 'True':
    requirements = []
else:
    requirements = ["numpy", "pandas", "scikit-learn",
                    "flask", "flask-restx", "flask-cors",
                    "coloredlogs", "pyyaml",
                    "werkzeug<1.0"  # Needed for flask-restx
                    ]

setup(author="Dih5",
      author_email='dihedralfive@gmail.com',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
      ],
      description='Package to implement Case-Based Reasoning systems',
      extras_require={
          "docs": ["nbsphinx", "sphinx-rtd-theme", "IPython"],
          "test": ["pytest"],
          "text": ["nltk"],
      },
      keywords=[],
      long_description=long_description,
      long_description_content_type='text/markdown',
      name='pycbr',
      packages=find_packages(include=['pycbr'], exclude=["demos", "tests", "docs"]),
      install_requires=requirements,
      url='https://github.com/dih5/pycbr',
      version='0.2.3',

      )
