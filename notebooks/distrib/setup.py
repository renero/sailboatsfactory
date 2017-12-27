#!/usr/bin/env python

from setuptools import setup, find_packages
setup(
    name="lst_ibex_konk",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    scripts=['lstm_ibex_konk.py', 'params.yaml'],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.yaml', '*.rst'],
    },

    # metadata for upload to PyPI
    author="Me",
    author_email="me@example.com",
    description="This is an Example Package",
    license="PSF",
    keywords="hello world example examples",
    url="http://example.com/HelloWorld/",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)

