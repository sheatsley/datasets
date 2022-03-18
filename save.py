"""
The save module writes (and reads) transformed machine learning datasets.
Author: Ryan Sheatsley
Wed Mar 2 2022
"""
import collections  # Container datatypes
import numpy as np  # The fundamental package for scientific computing with Python
import pandas  # Python Data Analysis Library
import pathlib  # Object-oriented filesystem paths
import pickle  # Python object serilization
from utilities import print  # Timestamped printing

# TODO
# save n_samples & features, n_classes, class breakdown, means & medians & stds
# compute figures of feature distributions, class-color-coded
# compute pearson correlation matricies


def tupalize(x, y, **metadata):
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
    return None


def analyze(training, testing=None):
    """ """
    return None


def write(
    dataframe,
    labels,
    part,
    name,
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
    :param part: dataset partition type
    :type part: str
    :param name: filename of dataset (dataset partition type will be appended)
    :type name: str
    :param precision: maximum dataset precision
    :type precision: numpy type
    :param analytics: whether to compute and save dataset analytics
    :type analytics: bool
    :param outdir: output directory
    :type outdir: pathlib path
    :return: None
    :rtype: NoneType
    """

    # convert to numpy arrays
    print(f"Assembling {name} and converting to (max) {precision} numpy arrays...")
    data = dataframe.to_numpy()
    labels = labels.to_numpy()
    precision = (
        precision
        if np.finfo(precision).precision < np.finfo(data.dtype).precision
        else data.dtype
    )

    # set maximum precision (label precision is set based on the number of classes)
    print(f"Setting precision to {precision}...")
    data = np.astype(data, precision)

    # populate a dataset object
    print("Populating dataset object...")
    dataset = Dataset(data, labels)

    # save the results to disk
    print(f"Writing {name + '-' + part} to {outdir}...")
    np.save(outdir / (name + "_training" if test_data else ""), training)

    # compute analyitcs if desired
    analyze(training, testing) if analytics else None
    return None


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
