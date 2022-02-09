"""
The args module serves as a wrapper for argparse.
Author: Ryan Sheatsley
Tue May 25 2021
"""
import argparse  # Parser for command-line options, arguments and sub-commands
import pathlib  # Object-oriented filesystem paths


def parse_args():
    """
    This function instantiates, populates, and parses command-line arguments
    for the datasets module.

    :return: command-line arguments
    :rtype: ArgumentParser
    """
    p = argparse.ArgumentParser(
        description="Retrieves and processes popular machine learning datasets.",
        prog="mlds",
    )

    # mandatory arguments
    p.add_argument("dataset", help="dataset to retrieve and process")

    # optional arguments
    p.add_argument(
        "-f",
        "--feature",
        action="append",
        default=[],
        help="column (or indicies) to transform",
        metavar="FEATURE",
        nargs="*",
    )
    p.add_argument(
        "-l",
        "--label",
        action="append",
        default=[],
        help="label manipulation scheme(s) to apply",
        nargs="*",
    )
    p.add_argument(
        "-n",
        "--name",
        help="output file name",
        metavar="DATASET_NAME",
    )
    p.add_argument(
        "--outdir", default="out", help="output directory", type=pathlib.Path
    )
    p.add_argument(
        "-p",
        "--precision",
        help="maximum dataset precision",
        default="float32",
    )
    p.add_argument(
        "-s",
        "--scheme",
        action="append",
        default=[],
        help="feature manipulation scheme(s) to apply",
        nargs="*",
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
        "-v",
        "--verbose",
        action="store_true",
        help="increase verbosity",
    )
    p.add_argument(
        "--version", action="version", help="displays module version", version="3.0"
    )
    return p.parse_args()


if __name__ == "__main__":
    """
    Example usage of MachineLearningDataSets via the command-line as:

        $ mlds nslkdd -f protocol flag -f service --outdir datasets
            -n nslkdd_mod -s standardscaler minmaxscaler -s onehotencoder
            -l labelencoder -a --destupefy

    This (1) downloads the NSL-KDD, (2) selects "protocol" & "flag" features as
    one group and "service" as the second group, (3) specifies an alternative
    output directory (instead of "out/"), (4) changes the base dataset name
    when saved, (5) creates three copies of dataset: two where "protocol" &
    "flag" (group one) are standaridized and the other where they are rescaled,
    and a third copy where "service" (group two) is one-hot encoded, (6)
    encodes labels as integers for all three dataset copies, and (7) computes
    basic analytics and applies destupification (to both dataset copies).
    """
    import sys  # System-specific parameters and functions

    sys.argv = "args.py nslkdd -f protocol flag -f service --outdir datasets\
            -n nslkdd_mod -s standardscaler minmaxscaler -s onehotencoder\
            -l labelencoder -a --destupefy".split()
    print(parse_args())
    raise SystemExit(0)
