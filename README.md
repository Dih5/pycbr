# pycbr

Package to implement Case-Based Reasoning systems


## Installation
Assuming you have a [Python3](https://www.python.org/) distribution with [pip](https://pip.pypa.io/en/stable/installing/), to install a development version, cd to the directory with this file and:

```
pip3 install -e .
```
As an alternative, a virtualenv might be used to install the package:
```
# Prepare a clean virtualenv and activate it
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
# Install the package
pip3 install -e .
```

To install also the dependencies to run the tests or to generate the documentation install some of the extras like
```
pip3 install -e '.[docs,test]'
```
Mind the quotes.

## Documentation
To generate the documentation, the *docs* extra dependencies must be installed. Furthermore, **pandoc** must be
available in your system.

To generate an html documentation with sphinx run:
```
make docs
```

To generate a PDF documentation using LaTeX:
```
make pdf
```



## Test
To run the unitary tests:
```
make test
```
