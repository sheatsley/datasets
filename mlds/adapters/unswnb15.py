"""
This module defines a custom adapter for downloading the UNSW-NB15.
Author: Ryan Sheatsley
Wed Mar 30 2022
"""
from mlds.adapters import baseadapter  # Base Adapter class for custom datasets
import io  # Core tools for working with streams
import pandas  # Python Data Analysis Library
from mlds.utilities import print  # Timestamped printing
import zipfile  # Work with ZIP archives


class UNSWNB15(baseadapter.BaseAdapter):
    """
    This class adds support for downloading, preprocessing, and saving the
    UNSW-NB15(https://research.unsw.edu.au/projects/unsw-nb15-dataset). It
    inherits the following interfaces from the BaseAdapter parent class:

    :func:`download`: retrieves datasets via HTTP through the requests module

    Moreover, it redefines the following interfaces:

    :func:`__init__`: instantiates UNSWNB15 objects
    :func:`preprocess`: resolves any dataset particulars
    :func:`read`: ensures the dataset conforms to the required standard

    Finally, since the UNSW-NB15 dataset is avaiable via CSV, feature names are
    extracted and made available for applying feature-specific transformations
    by name (as opposed to exclusively by index).
    """

    def __init__(self, directory="/tmp/", force_download=False):
        """
        The UNSW-NB15 can be readily retrieved as-is; no special preprocessing
        is necessary.

        :param directory: dataset download directory
        :type directory: string
        :return: UNSW-NB15-ready adapter
        :rtype: UNSWNB15 object
        """
        super().__init__(directory, force_download)

        # note that the training and test sets are mistakeningly labeled in reverse
        self.url = (
            "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path"
            "=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and"
            "%20testing%20set&files="
        )
        self.files = (
            f"a part of training and testing set/{file}"
            for file in ("UNSW_NB15_testing-set.csv", "UNSW_NB15_training-set.csv")
        )
        return None

    def preprocess(self, data):
        """
        Conforming to the BaseAdapter guidelines, we: (1) read the dataset
        as-is (no unpacking necessary), (2) we apply a series of desired
        transformations (i.e., dropping the "id" column and the last column,
        "label", since the UNSW-NB15 is commonly used for signature detection,
        via the "attack_cat" feature), and (3) we apply a series of necessary
        transformations to categorical variables, specifically, we remove: (a)
        state "CLO" and "ACC" are only present in the test set (which causes
        one-hot-encoding to fail), (b) state "PAR", "URN", and "no" are only
        found in one sample each in the training set (and not present in the
        test set), (c) protocol "rtp" is found in one sample in the training
        set (and not present in the test set).

        :param data: the current dataset file
        :type data: bytes
        :return: santized data file
        :rtype: pandas dataframe
        """
        with zipfile.ZipFile(io.BytesIO(data)) as zipped:

            # extract training and testing sets into pandas dataframe
            for file in self.files:
                print(f"Extracting {file}...")
                with io.TextIOWrapper(zipped.open(file)) as datafile:

                    # drop the first and last column
                    print("Loading & applying transformations...")
                    df = pandas.read_csv(datafile)
                    df.drop(columns=["id", "label"], inplace=True)

                    # apply remaining transformations
                    invalids = {"CLO", "ACC", "PAR", "URN", "no", "rtp"}
                    df.replace(invalids, None, inplace=True)
                    df.dropna(inplace=True)
                    df.reset_index(drop=True, inplace=True)
                yield df

    def read(self):
        """
        This method defines the exclusive interface expected by Dataset
        objects. Thus, this method should download (if necessary), prepare, and
        return the dataset as a pandas dataframe. Importantly, the read data
        must conform to the following standard:

        (1) If the dataset is for supervised learning, labels must be pointed
        to via the 'labels' key (as done with TensorFlow datasets), in their
        respective data category (data must be pointed to by a 'data' key).
        (2a) Training, testing, and validation data categories must be pointed
        to via "train", "test", and "validation" keys, respectively.
        (2b) If all dataset categories are disjoint in nature or if there is
        only a single source of data, then the key names can be arbitrary.
        (3) All data should be returned as a pandas dataframe.

        :return: the downloaded datasets
        :rtype: dictionary; keys are the dataset types & values are dataframes
        """
        return {
            partition: {
                "data": df.drop(columns="attack_cat"),
                "labels": df.attack_cat.copy(),
            }
            for partition, df in zip(
                ("train", "test"),
                self.preprocess(self.download(self.url)),
            )
        }


if __name__ == "__main__":
    """
    Example usage of the UNSW-NB15 class. It downloads the UNSW-NB15 (if it
    hasn't already) to /tmp, preprocesses the data, and returns the data as a
    pandas dataframe. Importantly, to debug adapters, they must be run form the
    root directory of the machine learning datasets repo as a module, such as:

                            python3 -m adapters.unswnb15
    """
    dataset = UNSWNB15().read()
    print(
        f'UNSW-NB15 has {len(dataset["train"]["data"])} training samples,',
        f'{len(dataset["test"]["data"])} test samples, and',
        f'{len(dataset["test"]["data"].columns)} features with',
        f'{len(dataset["test"]["labels"].unique())} classes.',
    )
    raise SystemExit(0)
