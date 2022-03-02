"""
The save module writes (and reads) transformed machine learning datasets.
Author: Ryan Sheatsley
Wed Mar 2 2022
"""
import numpy as np  # The fundamental package for scientific computing with Python
import pandas  # Python Data Analysis Library
import pathlib  # Object-oriented filesystem paths
from utilities import print  # Timestamped printing

# TODO
# save n_samples & features, n_classes, class breakdown, means & medians & stds
# compute figures of feature distributions, class-color-coded
# compute pearson correlation matricies


def write(
    train_data,
    train_labels,
    test_data,
    test_labels,
    name,
    precision=np.float32,
    analytics=False,
    outdir=pathlib.Path("out/"),
):
    """
    This function is the main exit point from the datasets repo. It consumes
    training and testing (if applicable) data and labels, casts them into numpy
    arrays (with precision no greater than specified), concatenates labels as
    the last column, and writes to disk with the specified filename and output
    directory. Optionally, analytics are computed and saved in the same
    directory.

    :param train_data: training data
    :type train_data: pandas dataframe
    :param train_labels: training labels
    :type train_labels: pandas series
    :param test_data: testing data
    :type test_data: pandas dataframe
    :param test_labels: testing labels
    :type test_labels: pandas series
    :param name: filename of dataset ("train" & "test" are appended, if applicable)
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

    # concatenate labels & convert to numpy arrays
    print(f"Assembling dataset and converting to (max) {precision} numpy arrays...")
    training = pandas.concat((train_data, train_labels), axis=1)
    testing = pandas.concat((test_data, test_labels), axis=1) if test_data else None
    training = training.to_numpy()
    testing = testing.to_numpy() if testing else None
    precision = (
        precision
        if np.finfo(precision).precision < np.finfo(training.dtype).precision
        else training.dtype
    )
    print(
        f"Lowering precision to {precision}..."
    ) if precision != training.dtype else None
    training = np.astype(training, precision)
    testing = np.astype(testing, precision) if testing else None
    return None


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
