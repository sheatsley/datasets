"""
This module defines a custom adapter for downloading the CIC-MalMem2022.
Author: Ryan Sheatsley
Fri Apr 1 2022
"""
from adapters import baseadapter  # Base Adapter class for custom datasets
import io  # Core tools for working with streams
import pandas  # Python Data Analysis Library
from utilities import print  # Timestamped printing


class CICMalMem2022(baseadapter.BaseAdapter):
    """
    This class adds support for downloading, preprocessing, and saving the
    CIC-MalMem-2022 (https://www.unb.ca/cic/datasets/malmem-2022.html). It
    inherits the following interfaces from the BaseAdapter parent class:

    :func:`download`: retrieves datasets via HTTP through the requests module

    Moreover, it redefines the following interfaces:

    :func:`__init__`: instantiates Phishing objects
    :func:`preprocess`: resolves any dataset particulars
    :func:`read`: ensures the dataset conforms to the required standard

    Finally, since the CIC-MalMem-2022 is avaiable via CSV, feature names are
    extracted and made available for applying feature-specific transformations
    by name (as opposed to exclusively by index).
    """

    def __init__(self, directory="/tmp/", force_download=False):
        """
        The CIC-MalMem-2022 can be readily retrieved as-is; no special
        preprocessing is necessary.

        :param directory: dataset download directory
        :type directory: string
        :return: Phishing-dataset-ready adapter
        :rtype: Phishing object
        """
        super().__init__(directory, force_download)

        # metadata to retrieve the dataset
        self.url = (
            "http://205.174.165.80/CICDataset/CICMalMem2022/Dataset/"
            "Obfuscated-MalMem2022.csv"
        )
        return None

    def preprocess(self, data):
        """
        Conforming to the BaseAdapter guidelines, we: (1) read the dataset
        as-is (no unpacking necessary), and (2) use the data for signature
        detection by dropping the last column, "Class", and using the first
        column, "Category", as the label, as well as parsing the malware
        category from the "Category" column (as to support signature
        detection).

        :param data: the current dataset file
        :type data: bytes
        :return: santized data file
        :rtype: pandas dataframe
        """

        # read the CSV directly into a dataframe and drop last column
        print("Read into dataframe and applying transformations...")
        df = pandas.read_csv(io.BytesIO(data))
        df.drop(columns="Class", inplace=True)
        df.Category = df.Category.str.split("-").str[0]
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
                "data": df.drop(columns="Category"),
                "labels": df.Category.copy(),
            }
            for df in self.preprocess(self.download(self.url))
        }


if __name__ == "__main__":
    """
    Example usage of the CICMalMem2022 class. It downloads the CIC-MalMem-2022
    (if it hasn't already) to /tmp, preprocesses the data, and returns the data
    as a pandas dataframe. Importantly, to debug adapters, they must be run
    form the root directory of the machine learning datasets repo as a module,
    such as:

                            python3 -m adapters.cicmalmem2022
    """
    dataset = CICMalMem2022().read()
    print(
        f'CIC-MalMem-2022 has {len(dataset["dataset"]["data"])} samples,',
        f'{len(dataset["dataset"]["data"].columns)} features with',
        f'{len(dataset["dataset"]["labels"].unique())} classes.',
    )
    raise SystemExit(0)
