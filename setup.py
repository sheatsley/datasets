"""
Build script for Machine Learning Datasets
Author: Ryan Sheatsley
Date: Thu Jun 2 2022
"""
import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    author="Ryan Sheatsley",
    author_email="ryan@sheatsley.me",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    description="Scripts & an API for working with machine learning datasets",
    entry_points={"console_scripts": "mlds=mlds:main"},
    install_requires=[
        "dill",
        "numpy",
        "matplotlib",
        "pandas",
        "requests",
        "scikit-learn",
        "tensorflow-datasets",
        "torchvision",
    ],
    license="BSD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="machine-learning numpy datasets",
    name="mlds",
    packages=setuptools.find_namespace_packages(),
    package_data={"mlds.out": ["*.pkl"]},
    python_requires=">=3.8",
    url="https://github.com/sheatsley/datasets",
    version="3.3",
)
