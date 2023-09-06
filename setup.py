"""
Build script for Machine Learning Datasets
"""
import subprocess

import setuptools

# compute git hash and save to file for non-editable installs
# overriding install for package data is bugged: https://github.com/pypa/setuptools/issues/1064
version = subprocess.check_output(
    ("git", "rev-parse", "--short", "HEAD"), text=True
).strip()
with open("mlds/VERSION", "w") as f:
    f.write(f"{version}\n")

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    author="Ryan Sheatsley",
    author_email="ryan@sheatsley.me",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    description="Scripts & an API for working with machine learning datasets",
    entry_points={"console_scripts": "mlds=mlds.datasets:command_line"},
    install_requires=[
        "numpy",
        "pandas",
        "requests",
        "scikit-learn",
        "tensorflow-datasets",
    ],
    license="BSD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="machine-learning numpy datasets",
    name="mlds",
    packages=setuptools.find_namespace_packages(),
    package_data={"mlds": ["VERSION"], "mlds.datasets": ["*.pkl"]},
    python_requires=">=3.10",
    url="https://github.com/sheatsley/datasets",
    version="4.1",
)
