"""
This module defines the adapter class inherited by custom dataset retrievers.
Author: Ryan Sheatsley
Fri Feb 11 2022
"""
import pandas  # Python Data Analysis Library
import pathlib  # Object-oriented filesystem paths
import requests  # HTTP for Humans
from mlds.utilities import print  # Timestamped printing


class BaseAdapter:
    """
    This BaseAdapter class defines an interface to retrieve, open, and process
    arbitrary datasets from web resources. It is designed to work with the
    Downloader class below. Downloader objects expect a single interface: a
    read function which returns the dataset (as a pandas dataframe). If
    available, Downloader objects will use column headers to allow accesing
    features by name (as well as index). Thus, this BaseAdapter class defines
    the essential preprocesing operations to be readily consumable by
    Downloader objects.

    :func:`__init__`: instantiates BaseAdapter objects
    :func:`download`: retrieves datasets via HTTP through the requests module
    :func:`preprocess`: resolves any dataset particulars
    :func:`read`: reads the dataset into memory
    """

    def __init__(self, directory="/tmp/", force_download=False):
        """
        All relevant dataset information should be defined here (e.g., the URL
        to retrieve the dataset and the directory to save it to).

        :param directory: directory to download the datasets to
        :type directory: string
        :param force_download: redownload the data, even if it exists
        :type force_download: boolean
        :return: example adapter
        :rtype: BaseAdapter object
        """
        self.url = "https://httpbin.org/get"
        self.directory = directory
        self.force_download = force_download
        return None

    def download(self, url):
        """
        This method uses the requests module to retrieve datasets from web
        resources. Designed to facilitate a simple and robust interface,
        subclasses need only specify the relevant URL to download the dataset.

        :param url: location of a dataset file
        :type url: string
        :param directory: directory to download the datasets to
        :type directory: string
        :return: the current dataset file
        :rtype: bytes
        """

        # create destination folder & download dataset (if necessary)
        path = pathlib.Path(self.directory, type(self).__name__.lower())
        path.mkdir(parents=True, exist_ok=True)
        data = path / url.split("/")[-1]
        if not data.is_file() or self.force_download:
            print(f"Downloading {url} to {self.directory}...")
            req = requests.get(url)
            req.raise_for_status()
            data.write_bytes(req.content)
        return data.read_bytes()

    def preprocess(self, data):
        """
        This method applies any dataset-specific nuances. Specifically, it
        should perform two functions: (1) data unpacking (be it, tarballs, JSON
        objects, ARFF files, etc.), and (2) any particular data transformations
        (such as manipulating labels, dropping features, etc.) Machine learning
        data is rarely "model-ready"; this function should make it so.

        :param data: the data to process
        :type data: bytes
        :return: santized data
        :rtype: pandas dataframe
        """
        return pandas.read_json(data)

    def read(self):
        """
        This method defines the exclusive interface expected by Downloader
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
        return {"httpbin": self.preprocess(self.download(self.url))}


if __name__ == "__main__":
    """
    Example usage of the BaseAdapter class. It tests HTTP GET on httpbin.org to
    retrieve the query parmaeters.
    """
    dataset = BaseAdapter().read()
    print(f"BaseAdapter httpbin GET reponse: {dataset}")
    raise SystemExit(0)
