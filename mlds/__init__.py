""""
This module initializes the datasets repo.
"""
import importlib
import pathlib
import pickle
import subprocess

__available__ = {p.stem for p in pathlib.Path(__file__).parent.glob("datasets/*.pkl")}
try:
    cmd = ("git", "-C", *__path__, "rev-parse", "--short", "HEAD")
    __version__ = subprocess.check_output(
        cmd, stderr=subprocess.DEVNULL, text=True
    ).strip()
except subprocess.CalledProcessError:
    with open(pathlib.Path(__file__).parent / "VERSION", "r") as f:
        __version__ = f.read().strip()


def __getattr__(dataset):
    """
    This function leverages PEP 562 (https://www.python.org/dev/peps/pep-0562)
    to enable quickly reading processed datasets into memory by avoiding
    loading heavyweight dependencies used when processing new datasets.
    Specifically, this function checks if the argument is an available dataset
    (and loads it if so), otherwise it assumes a standard import.

    :param dataset: the dataset to load (or module to import)
    :type dataset: str
    :return: the dataset (or module)
    :rtype: Dataset or module object
    """
    if dataset in __available__:
        with open(pathlib.Path(__file__).parent / f"datasets/{dataset}.pkl", "rb") as f:
            return pickle.load(f)
    return importlib.import_module(f".{dataset}", __name__)
