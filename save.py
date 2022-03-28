"""
The save module writes (and reads) transformed machine learning datasets.
Author: Ryan Sheatsley
Wed Mar 2 2022
"""
import collections  # Container datatypes
import dill  # serialize all of python
import numpy as np  # The fundamental package for scientific computing with Python
import pathlib  # Object-oriented filesystem paths
import utilities  # Miscellaneous helper functions
from utilities import print  # Timestamped printing


def read(dataset, outdir=pathlib.Path("out/")):
    """
    This function is the main entry point from datasets that been proccessed
    and saved by MLDS. It depickles saved data, and returns the namedtuple
    structure defined in the utilities module (i.e., the assemble function).
    Importantly, loading datasets executes the following based on the dataset
    arugment (parsed as {dataset}-{partition}): (1) if the partition is
    specified, then that partition is returned, else, (2) if the specified
    partition is "all", then *all* partitions are returned, else, if a
    partition is *not* specified and (3) there is only one partition, it is
    returned, else (4) if "train" & "test" are partitions, they are returned,
    else (4) an error is raised.

    :param dataset: the dataset to load
    :type dataset: str
    :param outdir: directory where the dataset is saved
    :type outdir: pathlib path
    :return: loaded dataset
    :rtype: namedtuple
    """

    # check if a partition is specified
    print(f"Loading {dataset} from {outdir}...")
    data, part = split if len(split := dataset.rsplit("-", 1)) == 2 else (dataset, None)
    partitions = set(outdir.glob(dataset + "*.pkl"))
    part_stems = [p.stem.rsplit("-")[1] for p in partitions]

    # case 1: the partition is specified
    if part and part != "all":
        with open(outdir / (dataset + ".pkl"), "rb") as f:
            return dill.load(f)

    # case 2: the partition is "all"
    elif part and part == "all":
        datasets = []
        for part in partitions:
            print(f"Loading {part.stem} partition...")
            with open(part, "rb") as f:
                datasets.append(dill.load(f))
        return collections.namedtuple("Dataset", part_stems)(*datasets)

    # case 3: there is only one partition
    elif len(partitions) == 1:
        with open(*partitions, "rb") as f:
            return dill.load(f)

    # case 4: train and test are available
    elif all(p in part_stems for p in ("train", "test")):
        datasets = []
        for part in ("train", "test"):
            print(f"Loading {part} partition...")
            with open(outdir / ("-".join([dataset, part]) + ".pkl"), "rb") as f:
                datasets.append(dill.load(f))
        return collections.namedtuple("Dataset", ("train", "test"))(*datasets)

    # case 5: dataset does not exist
    raise FileNotFoundError(outdir / ("-".join((data, "*.pkl"))))


def write(
    dataframe,
    labelframe,
    name,
    metadata={},
    precision=np.float32,
    analytics=False,
    outdir=pathlib.Path("out/"),
):
    """
    This function is the main exit point from MLDS. It consumes a dataset and
    labels, casts them into numpy arrays (with precision no greater than
    specified), instantiates & populates a dataset object, and writes a
    namedtuple object containing data & metadata to disk with the specified
    filename and output directory. Optionally, analytics are computed and saved
    in the same directory.

    :param dataframe: dataset
    :type dataframe: pandas dataframe
    :param labelframe: labels
    :type labelframe: pandas series
    :param name: filename of dataset (and partition)
    :type name: str
    :param metadata: metadata to be saved alongside the dataset
    :type metadata: dict of various datatypes
    :param precision: maximum dataset precision
    :type precision: numpy type
    :param analytics: whether to compute and save dataset analytics
    :type analytics: bool
    :param outdir: output directory
    :type outdir: pathlib path
    :return: None
    :rtype: NoneType
    """

    # convert to numpy arrays & check if numerical casting is possible (ie not strings)
    print(f"Assembling {name} and casting to (max) {precision.__name__} arrays...")
    data = dataframe.to_numpy()
    labels = labelframe.to_numpy()
    data_precision = (
        precision
        if issubclass(data.dtype.type, np.floating)
        and np.finfo(precision).precision < np.finfo(data.dtype).precision
        else data.dtype
    )
    label_precision = next(
        precision
        for precision in (np.int8, np.int16, np.int32, np.int64, labels.dtype)
        if all(np.can_cast(label, precision) for label in np.unique(labels))
    )

    # set maximum precision (label precision is set based on the number of classes)
    print(
        f"Casting data to {data_precision.__name__}",
        f"and labels to {label_precision.__name__}...",
    )
    data = data.astype(data_precision, copy=False)
    labels = labels.astype(label_precision, copy=False)

    # populate a dataset object
    print("Populating dataset object...")
    dataset = utilities.assemble(data, labels, metadata)

    # save the results to disk
    print(f"Pickling dataset & writing {name} to {outdir}...")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / (name + ".pkl"), "wb") as f:
        dill.dump(dataset, f)
    print(f"{name} saved to {outdir}!")

    # compute analyitcs if desired
    utilities.analyze(
        dataframe,
        labelframe.replace(metadata.get("class_map")),
        name,
        outdir,
    ) if analytics else None
    return None


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
