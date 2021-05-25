"""
The args module serves as a wrapper for argparse.
Author: Ryan Sheatsley
Tue May 25 2021
"""
import argparse  # Parser for command-line options, arguments and sub-commands


def parse_args():
    """
    This function instantiates, populates, and parses command-line arguments
    for the datasets module.

    :return: command-line arguments
    :rtype: ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Retrieves and processes popular machine learning datasets."
    )

    # mandatory arguments
    parser.add_arugment("-v", "--verbose", help="increase verbosity", action="store_true")

    # optional arguments
    parser.add_arugment("-v", "--verbose", help="increase verbosity", action="store_true")
    parser.add_arugment("--version", help="displays module version", action="version", version='3.0')
    return parser


if __name__ == "__main__":
    raise SystemExit(0)
