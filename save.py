"""
The save module writes (and reads) transformed machine learning datasets.
Author: Ryan Sheatsley
Wed Mar 2 2022
"""
import dill as pickle  # serialize all of python
import numpy as np  # The fundamental package for scientific computing with Python
import pathlib  # Object-oriented filesystem paths
import utilities  # Miscellaneous helper functions
from utilities import print  # Timestamped printing


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
    This function is the main exit point from the datasets repo. It consumes a
    dataset and labels, casts them into numpy arrays (with precision no greater
    than specified), instantiates & populates a dataset object, and writes a
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
    data.astype(data_precision, copy=False)
    labels.astype(label_precision, copy=False)

    # populate a dataset object
    print("Populating dataset object...")
    dataset = utilities.assemble(data, labels, metadata)

    # save the results to disk
    print(f"Pickling dataset & writing {name} to {outdir}...")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / name, "wb") as f:
        pickle.dump(dataset, f)
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
