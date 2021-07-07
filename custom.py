"""
The custom module defines templates for retrieving arbitrary datasets from
online web resources.
Author: Ryan Sheatsley
Tue Jul 6 2021
"""
import numpy as np  # The fundamental package for scientific computing with Python
import pathlib  # Object-oriented filesystem paths
import requests  # HTTP for Humans
from utils import print  # Timestamped printing


class DatasetTemplate(object):
    """
    This dataset template class defines an interface to retrieve, open, and
    process arbitrary datasets from web resources. It is designed to work with
    the Downloader class in the retrieve.py module. The Downloader class
    expects three function definitions from all templates: (1) downloading the
    dataset, (2) reading the dataset into memory, and (3) any preprocessing
    that must occur before it can be prepared into a numpy array. This
    DatasetTemplate class thus defines the essential interfaces for the
    Downloader class in the retrieve.py module and also provides simple example
    usage of how custom dataset subclasses should be designed.

    :func:`__init__`: instantiates DatasetTemplate objects
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
        :return: dataset template
        :rtype: DatasetTemplate object
        """
        self.urls = ["https://httpbin.org/get"]
        self.directory = directory
        return None

    def download(self, url, directory="/tmp/"):
        """
        This function uses the requests module to retrieve datasets from web
        resources.  Designed to facilitate a simple and robust interface,
        subclasses need only specify the relevant URL to download the dataset
        and this function will attempt to be as informative as possible with
        respect to any errors. A skeleton example is shown below.

        :param url: location of dataset
        :type url: string
        :param directory: directory to download the datasets to
        :type directory: string
        :return: none
        :rtype: NoneType
        """

        # create destination folder & download dataset (if necessary)
        path = pathlib.Path(directory, type(self).__name__.lower())
        path.mkdir(parents=True, exist_ok=True)
        for url in self.urls:
            data = path / url.split("/")[-1]
            if not data.is_file() or self.force_download:
                print("Downloading", url, "...")
                req = requests.get(url)
                req.raise_for_status()
                data.write_bytes(self.preprocess(req.content))
        return None

    def preprocess(self, data):
        """
        This function is designed to apply any dataset-specific nuances. For
        example, some datasets require that the labels be merged according to a
        certain scheme, some features should be dropped (such as index), among
        many other reasons. Concisely, "machine learning" data on the web is
        almost never ready for models as input; this function aims to
        essentially santize data.

        :param data: the data to process
        :type data: dataset-specific
        :return: santized data
        :rtype: dataset-specific
        """
        return data

    def read(self, directory="/tmp/"):
        """
        Datasets can be packaged in a myriad of different formats, such as
        tarballs, zip files, JSON objects, ARFF files, among many more. Thus,
        this function should be written as to conform to the particulars of any
        dataset. To maintain simplicity, only the directory for which the
        dataset was downloaded to should be specified -- the folder containing
        the dataset should be dervived from the name of the template (which is
        necessary to dynamically inform Downloader objects from retrieve.py
        what custom datasets are available, without hardcoding their
        availability inside Downloader objects), which should be the name of
        the dataset. Finally, to be compatible with the rest of the machine
        learning datasets pipepline, read data must conform to the following
        standard:

        (1) If the dataset is for supervised learning, labels must be
        pointed to via the 'labels' key (as done with TensorFlow datasets).
        (2) Training, testing, and validation data categories must be pointed
        to via "train", "test", and "validation" keys, respectively.
        (3) If all dataset categories are disjoint in nature or if there is
        only a single source of data, then the key names can be arbitrary (when
        saved, the dataset names will be defined by the key names).
        (4) All data should be returned as a numpy array.

        :param directory: directory where the dataset is downloaded
        :type directory: string
        :return: the downloaded datasets
        :rtype: dictionary; keys are the dataset types & values are numpy arrays
        """
        return {
            data.stem: np.array(data.read_bytes())
            for url in self.urls
            if (
                data := pathlib.Path(directory, type(self).__name__, url.split("/")[-1])
            )
        }


class NSL_KDD(DatasetTemplate):
    """
    This class adds support for downloading, preprocessing, and saving the
    NSL-KDD (https://www.unb.ca/cic/datasets/nsl.html). It inherits the following
    interfaces from the DatasetTemplate parent class:

    :func:`download`: retrieves datasets via HTTP through the requests module

    Moreover, it redefines the following interfaces:

    :func:`__init__`: instantiates NSL_KDD objects
    :func:`preprocess`: resolves any dataset particulars
    :func:`read`: reads the dataset into memory
    """

    def __init__(self, directory="/tmp/"):
        """
        All relevant dataset information should be defined here (e.g., the URL
        to retrieve the dataset and the directory to save it to).

        :param directory: directory to download the datasets to
        :type directory: string
        :return: dataset template
        :rtype: DatasetTemplate object
        """
        self.urls = ["http://205.174.165.80/CICDataset/NSL-KDD/Dataset/NSL-KDD.zip"]
        self.directory = directory
        return None


if __name__ == "__main__":
    """
    Example usage of instaniating the base DatasetTemplate class. It retrieves
    a webpage from https://httpbin.org/get, saves it as a binary file "get" in
    /tmp/, and reads the binary file into memory.
    """
    dataset = DatasetTemplate()
    dataset.download(dataset.urls)
    print(dataset.read())
    raise SystemExit(0)
