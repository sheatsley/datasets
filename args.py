"""
The args module serves as a wrapper for argparse.
Author: Ryan Sheatsley
Tue May 25 2021
"""
import argparse  # Parser for command-line options, arguments and sub-commands
import pathlib  # Object-oriented filesystem paths
import sys  # System-specific parameters and functions

# TODO
# - create symlinks to target dir
# - destupify should cleanse for unknown values 
# - drop multiple datasets (zip -i and -s for better label-specific schema)

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
    p.add_argument("dataset", help="dataset to retrieve and process", nargs="+")

    # optional arguments
    p.add_argument(
        "-i",
        "--include",
        action="append",
        help="column (or indicies) to keep",
        nargs="*",
        metavar="FEATURE",
        default=[],
    )
    p.add_argument(
        "-n",
        "--name",
        help="output file names",
        nargs="+",
        metavar="DATASET_NAME",
    )

    p.add_argument(
        "--outdir", default="out", help="output directory", type=pathlib.Path
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

        $ mlds mnist nslkdd -i 50-700 -i protocol flag --outdir datasets
            -n mnist_mod nslkdd_mod -s normalization -s standardization minmax
            -a --destupefy

    This (1) downloads MNIST and NSL-KDD, (2) selects all features for MNIST
    (necessary to correctly associate the following "-i" with NSL-KDD) and
    "protocol" & "flag" for NSL-KDD, (3) specifies an alternative output
    directory (instead of "out/"), (4) changes the base dataset name when
    saved, (5) applies minmax scaling to MNIST and creates two copies of the
    NSL-KDD that are standardization & normalized, respectively, and (6)
    computes basic analytics and applies destupification (to both datasets).
    """
    sys.argv = "args.py mnist nslkdd -i -i protocol flag --outdir datasets\
            -n mnist_mod nslkdd_mod -s minmax -s standardize normalize\
            -a --destupefy".split()
    print(parse_args())
    raise SystemExit(0)
