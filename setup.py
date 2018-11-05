from setuptools import setup, find_packages
import sys, os

version = '0.1'

setup(
    name='ALICE',
    version=version,
    description="",
    long_description="""""",
    classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    keywords='',
    author='',
    author_email='',
    url='',
    license='',
    packages=find_packages(exclude=['ez_setup', 'tests']),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        # -*- Extra requirements: -*-
    ],
    entry_points=\
    """
    """,
)

git+https://github.com/pathak22/pyflow.git