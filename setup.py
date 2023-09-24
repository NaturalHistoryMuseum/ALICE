from setuptools import find_packages, setup



setup(
    name="alice",
    version="2.0",
    description='ALICE: Angled Label Somthing Capture Equipment',
    url='http://github.com/naturalhistorymuseum/ALICE',
    author='Arianna Salili-James; Ben Scott',
    author_email='',    
    license="MIT",
    packages=["alice", "alice/tasks"],
    # package_data={'alice': ['alice/**/*']},
    # packages=find_packages(),
    # packages=find_packages(include=('alice',)),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[],
    entry_points={
        'console_scripts': [
            'process = scripts.process:main',
        ],
    },    
    keywords="machine-learning, deep-learning, ml, pytorch, text, text-detection",
)