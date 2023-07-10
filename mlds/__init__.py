""""
This module initializes the datasets repo.
"""
import importlib
import pathlib
import pickle

__available__ = {p.stem for p in pathlib.Path(__file__).parent.glob("datasets/*.pkl")}


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


# def load(dataset, out="out/"):
#     """
#     This function is the main entry point from datasets that been proccessed
#     and saved by MLDS. It depickles saved data, and returns the named tuple
#     structure defined in the utilities module (i.e., the assemble function).
#     Importantly, loading datasets executes the following based on the dataset
#     arugment (parsed as {dataset}-{partition}): (1) if the partition is
#     specified, then that partition is returned, else, (2) if the specified
#     partition is "all", then *all* partitions are returned, else, if a
#     partition is *not* specified and (3) there is only one partition, it is
#     returned, else (4) if "train" & "test" are partitions, they are returned,
#     else (4) an error is raised.

#     :param dataset: the dataset to load
#     :type dataset: str
#     :param out: directory where the dataset is saved
#     :type out: str
#     :return: loaded dataset
#     :rtype: namedtuple object
#     """
#     import pathlib  # Object-oriented filesystem paths

#     import dill  # serialize all of python

#     # check if a partition is specified
#     out = pathlib.Path(__file__).parent / "out/" if out == "out/" else pathlib.Path(out)
#     print(f"Loading {dataset}.pkl from {out}/...")
#     data, part = split if len(split := dataset.rsplit("-", 1)) == 2 else (dataset, None)
#     partitions = set(out.glob(f"{dataset}*.pkl"))
#     part_stems = [p.stem.rsplit("-")[1] for p in partitions]

#     # case 1: the partition is specified
#     if part and part != "all":
#         with open(out / f"{dataset}.pkl", "rb") as f:
#             return Dataset(data, [(part, dill.load(f))])

#     # case 2: the partition is "all"
#     elif part and part == "all":
#         datasets = []
#         for part in partitions:
#             print(f"Loading {part.stem} partition...")
#             with open(part, "rb") as f:
#                 datasets.append((part, dill.load(f)))
#         return Dataset(data, datasets)

#     # case 3: there is only one partition
#     elif len(partitions) == 1:
#         with open(*partitions, "rb") as f:
#             return Dataset(data, [("dataset", dill.load(f))])

#     # case 4: train and test are available
#     elif all(p in part_stems for p in ("train", "test")):
#         datasets = []
#         for part in ("train", "test"):
#             print(f"Loading {part} partition...")
#             with open(out / (f"{'-'.join([dataset, part])}.pkl"), "rb") as f:
#                 datasets.append((part, dill.load(f)))
#         return Dataset(data, datasets)

#     # case 5: the pickle was not found
#     raise FileNotFoundError(f"{dataset} not found in '{out}'! (Is it downloaded?)")""
