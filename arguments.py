"""
The args module serves as a wrapper for argparse.
Author: Ryan Sheatsley
Tue May 25 2021
"""
import argparse  # Parser for command-line options, arguments and sub-commands
import itertools  # Functions creating iterators for efficient looping¶
import numpy as np  # The fundamental package for scientific computing with Python
import pathlib  # Object-oriented filesystem paths
import transform  # Order-preserving transformations for machine learning data
from utilities import print  # Timestamped printing


def parse_args():
    """
    This function instantiates, populates, and parses command-line arguments
    for the datasets module.

    :return: command-line arguments
    :rtype: argparse.Namesapce
    """
    p = argparse.ArgumentParser(
        description="Retrieves and processes popular machine learning datasets.",
        prog="mlds",
    )

    # mandatory arguments
    p.add_argument(
        "dataset",
        help="dataset to retrieve and process",
    )

    # optional arguments
    p.add_argument(
        "-f",
        "--features",
        action="append",
        default=[],
        help="transformable features (or indicies) ('all' uses all)",
        metavar="FEATURE",
        nargs="*",
    )
    p.add_argument(
        "-l",
        "--labels",
        # action="append",
        default=[],
        help="label manipulation scheme(s) to apply",
        nargs="*",
        type=transformer_args,
    )
    p.add_argument(
        "-n",
        "--names",
        help="output file name",
        metavar="DATASET_NAME",
        nargs="*",
    )
    p.add_argument(
        "--outdir",
        default="out",
        help="output directory",
        type=pathlib.Path,
    )
    p.add_argument(
        "-p",
        "--precision",
        help="maximum dataset precision",
        default=np.float32,
        type=np.dtype,
    )
    p.add_argument(
        "-s",
        "--schemes",
        action="append",
        default=[],
        help="feature manipulation scheme(s) to apply",
        nargs="*",
        type=transformer_args,
    )

    # flags
    p.add_argument(
        "-a",
        "--analytics",
        action="store_true",
        help="compute basic data analytics",
    )
    p.add_argument(
        "--destupefy",
        action="store_true",
        help="cleanup datasets automagically (experimental)",
    )
    p.add_argument(
        "--version",
        action="version",
        help="displays module version",
        version="3.0",
    )
    return p.parse_args()


def transformer_args(arg):
    """
    This function casts schemes and labels as transformer callables.

    :param args: scheme or label arguments
    :type args: list of strings
    :return: transformer callables
    :rtype: list of Transformer methods
    """
    return getattr(transform.Transformer, arg)


def validate_args():
    """
    This function executes a series of assertions and prints debug information
    to confirm proper behavior of the MachineLearningDataSets API.

    :return: command-line arguments
    :rtype: dictionary; keys are arguments and values are entries
    """

    # extract command-line args, correct dataset names, & run assertions
    args = parse_args()
    datasets = list(itertools.product(*args.schemes, args.labels))
    args.names = (
        args.names
        if args.names
        else [
            "_".join([args.dataset] + [s.__name__ for s in scheme])
            for scheme in datasets
        ]
    )

    # schemes and features must be the same length
    assert len(args.schemes) == len(args.features), (
        "Schemes and features are not equal length!"
        + f"{[[s.__name__ for s in scheme] for scheme in args.schemes], args.features}"
    )

    # names must be equal to the product of lengths of schemes & labels
    assert len(args.names) == len(datasets), (
        "Names must be equal to the number of produced datasets! "
        + f"{args.names} != "
        + str(
            "×".join(
                [str([s.__name__ for s in sc]) for sc in args.schemes + [args.labels]]
            )
        )
    )

    # print parsed arguments as a sanity check
    print("Arguments:", *(f"{a}={v}" for a, v in vars(args).items()))
    return vars(args)


if __name__ == "__main__":
    """
    Example usage of MachineLearningDataSets via the command-line as:

        $ mlds nslkdd -f duration count -f service --outdir datasets
            -n nslkdd_ss nslkdd_mms -s standardscaler minmaxscaler
            -s onehotencoder -l labelencoder -a --destupefy

    This (1) downloads the NSL-KDD, (2) selects "duration" and "count" as one
    group and "service" as the second group, (3) specifies an alternative
    output directory (instead of "out/"), (4) changes the base dataset name (to
    "nslkdd_ss" and "nslkdd_mms") when saved, (5) creates two copies of the
    dataset: one where "duration" & "count" (subgroup one) are standaridized
    and another where they are rescaled, and, in both copies, "service" (group
    two) is one-hot encoded, (6) encodes labels as integers for both dataset
    copies, and (7) computes basic analytics, and (8) applies destupefication
    (to both dataset copies).
    """
    import sys  # System-specific parameters and functions

    sys.argv = "args.py nslkdd -f duration count -f service --outdir datasets\
                -n nslkdd_ss nslkdd_mms -s standardscaler minmaxscaler\
                -s onehotencoder -l labelencoder -a --destupefy".split()
    validate_args()
    raise SystemExit(0)
