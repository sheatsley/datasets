"""
This module defines a custom adapter for downloading the Phishing dataset.
Author: Ryan Sheatsley
Tue Mar 29 2022
"""
from adapters import baseadapter  # Base Adapter class for custom datasets
import io  # Core tools for working with streams
import pandas  # Python Data Analysis Library
from utilities import print  # Timestamped printing
import zipfile  # Work with ZIP archives


class Phishing(baseadapter.BaseAdapter):
    """
    This class adds support for downloading, preprocessing, and saving the
    Phishing dataset 2019 (https://www.fcsit.unimas.my/phishing-dataset). It
    inherits the following interfaces from the BaseAdapter parent class:

    :func:`download`: retrieves datasets via HTTP through the requests module

    Moreover, it redefines the following interfaces:

    :func:`__init__`: instantiates Phishing objects
    :func:`preprocess`: resolves any dataset particulars
    :func:`read`: ensures the dataset conforms to the required standard

    Finally, since the Phishing dataset is avaiable as ARFF, feature names are
    extracted and made available for applying feature-specific transformations
    by name (as opposed to exclusively by index).
    """

    def __init__(self, directory="/tmp/", force_download=False):
        """
        The Phishing dataset can be readily retrieved as-is; no special
        preprocessing is necessary.

        :param directory: dataset download directory
        :type directory: string
        :return: Phishing-dataset-ready adapter
        :rtype: Phishing object
        """
        super().__init__(directory, force_download)

        # metadata to retrieve the dataset
        self.url = (
            "https://md-datasets-public-files-prod.s3.eu-west-1.amazonaws.com/"
            "2c3b3b4e-cec8-4094-8f0d-b69e6f1234d5"
        )
        return None

    def preprocess(self, data):
        """
        Conforming to the BaseAdapter guidelines, we: (1) unzip the desired
        files for the NSL-KDD,  and (2) apply the desired transformations
        (i.e., dropping the last "difficulty" column and applying the label
        transformation defined above to the second-to-last "attack" column).

        :param data: the current dataset file
        :type data: bytes
        :return: santized data file
        :rtype: pandas dataframe
        """

        # extract feature names from ARFF header & instantiate dataframe
        print("Extracting feature names and loading dataframe...")
        datafile = data.decode().splitlines()
        features = [line.split()[1] for line in datafile[2:51]]
        df = pandas.DataFrame([x.split(",") for x in datafile[53:]], columns=features)
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
            "dataset": {
                "data": df.drop(columns="CLASS_LABEL"),
                "labels": df["CLASS_LABEL"].copy(),
            }
            for df in self.preprocess(self.download(self.url))
        }


if __name__ == "__main__":
    """
    Example usage of the Phishing class. It downloads the Phishing dataset (if
    it hasn't already) to /tmp, preprocesses the data, and returns the data as
    a pandas dataframe. Importantly, to debug adapters, they must be run form
    the root directory of the machine learning datasets repo as a module, such
    as:

                            python3 -m adapters.nslkdd
    """
    dataset = Phishing().read()
    print(
        f'Phishing has {len(dataset["dataset"]["data"])} training samples,',
        f'{len(dataset["dataset"]["data"].columns)} features with',
        f'{len(dataset["dataset"]["labels"].unique())} classes.',
    )
    raise SystemExit(0)
