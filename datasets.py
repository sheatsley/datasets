"""
This module defines the main entry point for the datasets repo. It consists of
(1) parsing arguments, (2) retrieving datasets, (3) feature scaling
applications, (4) one-hot, label, & integer encoding, and (5) writing the
resultant arrays to disk.
Author: Ryan Sheatsley
Mon Feb 28 2022
"""

import retrieve  # Download machine learning datasets
import transform  # Apply transformations to machine learning datasets
from utilities import print  # Timestamped printing


def main(
    analytics,
    dataset,
    destupefy,
    features,
    labels,
    names,
    outdir,
    precision,
    schemes,
    verbose,
):
    """
    This function represents the heart of MachineLearningDatasets (MLDS). Given
    a dataset, a list of features, a list of label transformations, a list of
    output names, an output directory, numerical precision, a list of
    transformation schemes, and whether to produce resultant analyitcs and
    apply (experimental) destupefication subroutines, MLDS will perform the
    following steps:

        (1) Retrieve the dataset from either torchvision, tensorflow_datasets,
            or a custom dataset (found in the adapaters directory).
        (2) Apply transformations to specific features and reassmble the
            dataset is in the original order.
        (3) Optionally clean the data and produce basic statistics.
        (4) Save the dataset in the specified output directory with
            the specified name in the specified precision.

    Practically speaking, this function mediates interactions between
    Downloader and Transformer objects, while saving data so that it can be
    readily retrieved by the load function in this module.

    :param analytics: whether analytics are computed and saved
    :type analytics: bool
    :param dataset: dataset to download
    :type dataset: string
    :param destupefy: whether data cleaning is performed (experimental)
    :type destupefy: bool
    :param features: features to manipulate
    :type features: tuple of tuples containing indicies
    :param labels: transfomrations to apply to the labels
    :type labels: tuple of tuples of Transformer callables
    :param names: filenames of the saved datasets
    :type names: tuple of strings
    :param outdir: ouput directory of saved datasets
    :type outdir: pathlib.Path object
    :param precision: dataset precision
    :type precision: numpy data type
    :param schemes: transformations to apply to the data
    :type schemes: tuple of tuples of Transformer callables
    :param verbose: whether to use verbose output
    :type verbose: bool
    :return: None
    :rtype: NoneType
    """

    # instantiate Downloader and Transformer objects
    print("Instantiating Downloader and Transformer objects...")
    downloader = retrieve.Downloader(dataset)
    transformer = transform.Transformer(features, labels, schemes)
    return None


if __name__ == "__main__":
    """
    This is the entry point when running MachineLearningDataSets via the
    command-line. It first parses arguments from the command line, sets the
    appropriate parameters, and calls main to execute the parsed instructions.

    Example usage:

        $ mlds nslkdd -f duration count -f service --outdir datasets
            -n nslkdd_ss nslkdd_mms -s standardscaler minmaxscaler
            -s onehotencoder -l labelencoder -a --destupefy

    This (1) downloads the NSL-KDD, (2) selects "duration" and "count" as one
    group and "service" as the second group, (3) specifies an alternative
    output directory (instead of "out/"), (4) changes the base dataset name (to
    "nslkdd_mod") when saved, (5) creates two copies of the dataset: one where
    "duration" & "count" (subgroup one) are standaridized and another where
    they are rescaled, and, in both copies, "service" (group two) is one-hot
    encoded, (6) encodes labels as integers for both dataset copies, and (7)
    computes basic analytics, and (8) applies destupefication (to both dataset
    copies).
    """
    import arguments  # Command-line Argument parsing

    # parse command-line args and enter main
    main(**arguments.validate_args())
    raise SystemExit(0)
