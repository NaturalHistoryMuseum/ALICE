
from setuptools import find_packages, setup

setup(
    name="alice",
    version="2.0",
    description='ALICE: Angled Label Somthing Capture Equipment',
    url='http://github.com/naturalhistorymuseum/ALICE',
    author='',
    author_email='',    
    packages=["alice", "alice/tasks"],
    # package_data={'alice': ['alice/**/*']},
    # packages=find_packages(),
    # packages=find_packages(include=('alice',)),
    include_package_data=True,
    install_requires=[
        'torch==2.0.1'  # torch is required before installing detectron
    ],
)

