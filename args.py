"""
The args module serves as a wrapper for argparse.
Author: Ryan Sheatsley
Tue May 25 2021
"""
import argparse  # Parser for command-line options, arguments and sub-commands

# TODO
# ensure number of ouputs match number of inputs


def parse_args():
    """
    This function instantiates, populates, and parses command-line arguments
    for the datasets module.

    :return: command-line arguments
    :rtype: ArgumentParser
    """
    p = argparse.ArgumentParser(
        description="Retrieves and processes popular machine learning datasets."
    )

    # mandatory arguments
    p.add_argument("dataset", help="dataset to retrieve and process")

    # optional arguments
    p.add_argument(
        "-i",
        "--include",
        help="column (or indicies) to keep",
        nargs="+",
        metavar="FEATURE",
    )
    p.add_argument(
        "--outdir", help="output directory", type=pathlib.PAth, default="out"
    )
    p.add_argument(
        "-o", "--output", help="output file names", nargs="+", action="append", metavar="DATASET_NAME"
    )
    p.add_argument(
        "-s",
        "--scheme",
        help="feature manipulation scheme(s) to apply",
        nargs="+",
    )

    # flags
    p.add_argument(
        "-a", "--analytics", help="compute basic data analytics", action="store_true"
    )
    p.add_argument(
        "--destupefy",
        help="cleanup datasets automagically (experimental)",
        action="store_true",
    )
    p.add_argument("-v", "--verbose", help="increase verbosity", action="store_true")
    p.add_argument(
        "--version", help="displays module version", action="version", version="3.0"
    )
    return p


if __name__ == "__main__":
    raise SystemExit(0)
