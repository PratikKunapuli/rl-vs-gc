from setuptools import setup, find_packages

setup(
    name='aerialmanipulation',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'torch',
        'ruamel.yaml',
        'seaborn',
        'matplotlib',
        'warp-lang',
        'optuna',
    ],
)