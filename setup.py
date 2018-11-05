from setuptools import find_packages, setup

version = '0.1'

setup(
    name='ALICE',
    version=version,
    description='ALICE: Angled Label Image Capture and Extraction',
    long_description='Attempts to locate and extract images of labels attached to '
                     'pinned insects. Given views of the specimen from multiple angles, '
                     'it can isolate the labels.',
    classifiers=[],
    keywords='',
    author='Alice Butcher; James Durrant; Pieter Holtzhausen; Ben Scott',
    author_email='',
    url='https://github.com/NaturalHistoryMuseum/ALICE',
    license='',
    packages=find_packages(exclude=['ez_setup', 'tests']),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'pyflow',
        'pygco',
        'cached-property',
        'matplotlib',
        'numba',
        'scikit-image',
        'scikit-learn',
        'pandas',
        'jupyter',
        'numpy',
        'cython'
        ],
    dependency_links=[
        "git+https://github.com/pathak22/pyflow#egg=pyflow",
        ]
    )
