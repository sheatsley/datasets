"""
Build script for machine learning datasets.
Author: Ryan Sheatsley
Mon Apr 4 2022
"""
import setuptools  # Easily download, build, install, upgrade, and uninstall Python packages

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    author="Ryan Sheatsley",
    author_email="ryan@sheatsley.me",
    description="Machine Learning Datasets",
    license="BSD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "dill>=0.3.4",
        "matplotlib>=3.5.1",
        "numpy>=1.22.3",
        "pandas>=1.4.1",
        "requests>=1.0.2",
        "scikit-learn>=1.0.2",
        "tensorflow-datasets>=4.5.2",
        "torchvision>=0.12.0",
    ],
    keywords="machine-learning numpy datasets",
    name="mlds",
    py_modules=["mlds"],
    python_requires=">=3.8",
    url="https://github.com/sheatsley/datasets",
    version="3.0",
)
