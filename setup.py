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
    keywords="machine-learning numpy datasets",
    name="mlds",
    py_modules=["mlds"],
    python_requires=">=3",
    url="https://github.com/sheatsley/datasets",
    version="3.0",
)
