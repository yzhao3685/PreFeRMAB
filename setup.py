from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "This repo requires Python 3.6 or greater." \
    + "Please install it before proceeding."

setup(
    name='prefermab',
    py_modules=['prefermab'],
    version='0.1.0',
    install_requires=[
        'numpy',
        'pandas',
        'ipython',
        'joblib',
        'matplotlib',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'torch',
        'gym',
        'nashpy',
        'tqdm'
    ],
    description="PreFeRMAB code. Adapted from OpenAI's SpinningUp repository and https://github.com/killian-34/RobustRMAB",
)
