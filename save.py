"""
The save module writes (and reads) transformed machine learning datasets.
Author: Ryan Sheatsley
Wed Mar 2 2022
"""
import collections  # Container datatypes
import dill as pickle  # serialize all of python
import numpy as np  # The fundamental package for scientific computing with Python
import pandas  # Python Data Analysis Library
import pathlib  # Object-oriented filesystem paths
from utilities import print  # Timestamped printing

# TODO
# save n_samples & features, n_classes, class breakdown, means & medians & stds
# compute figures of feature distributions, class-color-coded
# compute pearson correlation matricies


def assemble(x, y, metadata={}):
    """
    This function populates namedtuples with data & labels, as well as any
    desired metadata.

    :param x: data samples
    :type x: numpy array
    :param y: labels
    :type y: numpy array
    :param metadata: metadata to be stored
    :type metadata: dictionary
    :return: complete dataset with metadata
    :rtype: namedtuple object
    """
    return collections.namedtuple("Dataset", ["data", "labels", *metadata])(
        x, y, **metadata
    )


def analyze(dataframe, labels, name, outdir):
    """
    :param dataframe: dataset
    :type dataframe: pandas dataframe
    :param labels: labels
    :type labels: pandas series
    :param name: filename of dataset (and partition)
    :type name: str
    :param outdir: output directory
    :type outdir: pathlib path
    :return: None
    :rtype: NoneType
    """
    return None


def write(
    dataframe,
    labels,
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
    :param labels: labels
    :type labels: pandas series
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
    print(f"Assembling {name} and converting to (max) {precision} numpy arrays...")
    data = dataframe.to_numpy()
    labels = labels.to_numpy()
    data_precision = (
        precision
        if issubclass(data.dtype, np.floating)
        and np.finfo(precision).precision < np.finfo(data.dtype).precision
        else data.dtype
    )
    label_precision = next(
        precision
        for precision in (np.int8, np.int16, np.int32, np.int64, labels.dtype)
        if all(np.can_cast(label, precision) for label in np.unique(labels))
    )

    # set maximum precision (label precision is set based on the number of classes)
    print(f"Casting data to {data_precision} and labels to {label_precision}...")
    data.astype(data_precision, copy=False)
    labels.astype(label_precision, copy=False)

    # populate a dataset object
    print("Populating dataset object...")
    dataset = assemble(data, labels, metadata)

    # save the results to disk
    print(f"Pickling dataset & writing {name} to {outdir}...")
    with open(outdir / name, "wb") as f:
        pickle.dump(dataset, f)

    # compute analyitcs if desired
    analyze(dataframe, labels, name, outdir) if analytics else None
    return None


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
