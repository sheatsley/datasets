"""
The retrieve module downloads machine learning datasets from popular
repositories.
Author: Ryan Sheatsley
Fri Jun 18 2021
"""
from utils import print # use timestamped print

class Downloader:
    """
    This downloader class serves as a wrapper for popular
    machine learning libraries to retrieve datasets. Moreover,
    it is designed to be easily extendable to support downloading
    datasets from ostensibly any location via the Requests module.

    """
    
    def __init__(self):
        """
        """
        return None


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
    raise SystemExit(0)
