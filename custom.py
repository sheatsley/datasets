"""
The custom module defines templates for retrieving arbitrary datasets from
online web resources.
Author: Ryan Sheatsley
Tue Jul 6 2021
"""
import io  # Core tools for working with streams
import numpy as np  # The fundamental package for scientific computing with Python
import pandas  # Python Data Analysis Library
import pathlib  # Object-oriented filesystem paths
import requests  # HTTP for Humans
from utils import print  # Timestamped printing
import zipfile  # Work with ZIP archives


class DatasetTemplate:
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
        self.urls = ("https://httpbin.org/get",)
        self.directory = directory
        self.force_download = force_download
        return None

    def download(self, urls, directory="/tmp/"):
        """
        This function uses the requests module to retrieve datasets from web
        resources. Designed to facilitate a simple and robust interface,
        subclasses need only specify the relevant URL to download the dataset
        and this function will attempt to be as informative as possible with
        respect to any errors. A skeleton example is shown below.

        :param url: location of dataset
        :type url: list of strings
        :param directory: directory to download the datasets to
        :type directory: string
        :return: none
        :rtype: NoneType
        """

        # create destination folder & download dataset (if necessary)
        path = pathlib.Path(directory, type(self).__name__.lower())
        path.mkdir(parents=True, exist_ok=True)
        for url in urls:
            data = path / url.split("/")[-1]
            if not data.is_file() or self.force_download:
                print("Downloading", url, "to", directory, "...")
                req = requests.get(url)
                req.raise_for_status()
                data.write_bytes(self.preprocess(req.content))
        return None

    def preprocess(self, data):
        """
        This function is designed to apply any dataset-specific nuances.
        Specifically, this function should be written to resolve at least two
        dataset-specific nuances: (1) datasets can be packaged in a myrida of
        formats, such as tarballs, zip files, JSON objects, ARFF files, among
        others; unpacking the data appropriately should be done here, and (2)
        some datasets require that the labels be merged according to a certain
        scheme, some features should be dropped (such as index), among many
        other reasons; such transformations should be done here. In many ways,
        "machine learning" data on the web is almost never ready for models as
        input; this function should aim to extract and santize data.

        :param data: the data to process
        :type data: dataset-specific
        :return: santized data
        :rtype: dataset-specific
        """
        return data

    def read(self, directory="/tmp/"):
        """
        Many machine learning frameworks that have dataset-support built-in
        often have interfaces where data is nearly ready to use; this function
        should aim to simply read data that is nearly ready to use (i.e., via
        preprocess) into memory. This function should be inherently simple
        (since it is likely to be called multiple times, versus preprocess,
        which will, on average, be called once per dataset throughout the
        lifetime of a user) To enforce this simplicity objective, only the
        directory for which the dataset was downloaded to should be specified
        -- the folder containing the dataset should be dervived from the name
        of the template (which is necessary to dynamically inform Downloader
        objects from retrieve.py what custom datasets are available, without
        hardcoding their availability inside Downloader objects), which should
        be the name of the dataset. Finally, to be compatible with the rest of
        the machine learning datasets pipepline, read data must conform to the
        following standard:

        (1) If the dataset is for supervised learning, labels must be pointed
        to via the 'labels' key (as done with TensorFlow datasets), in their
        respective data category.
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


class NSLKDD(DatasetTemplate):
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

    def __init__(self, directory="/tmp/", force_download=False):
        """
        For the NSL-KDD, the relevant dataset information is (1) the URL to
        retrieve the dataset, (2) where to save it, (3) the desired
        files to use from the zip archive, and (4) the label
        transformation applied in the preprocess function,

        :param directory: directory to download the datasets to
        :type directory: string
        :return: dataset template
        :rtype: DatasetTemplate object
        """
        super().__init__(directory, force_download)
        self.urls = ("http://205.174.165.80/CICDataset/NSL-KDD/Dataset/NSL-KDD.zip",)
        self.files = ("KDDTrain+.txt", "KDDTest+.txt")
        self.categories = {
            file: category for file, category in zip(self.files, ["train", "test"])
        }
        self.transform = {
            **dict.fromkeys(
                [
                    "apache",
                    "back",
                    "land",
                    "neptune",
                    "pod",
                    "processtable",
                    "smurf",
                    "teardrop",
                    "udpstorm",
                    "worm",
                ],
                "dos",
            ),
            **dict.fromkeys(
                ["ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"], "probe"
            ),
            **dict.fromkeys(
                [
                    "ftp_write",
                    "guess_password",
                    "httptunnel",
                    "imap",
                    "named",
                    "multihop",
                    "phf",
                    "sendmail",
                    "snmpgetattack",
                    "snmpguess",
                    "spy",
                    "warezclient",
                    "warezmaster",
                    "xlock",
                    "xsnoop",
                ],
                "r2l",
            ),
            **dict.fromkeys(
                [
                    "buffer_overflow",
                    "loadmodule",
                    "perl",
                    "ps",
                    "rootkit",
                    "xterm",
                    "sqlattack",
                ],
                "u2r",
            ),
        }
        return None

    def preprocess(self, data, directory="/tmp/"):
        """
        As described in the preprocess comments for the DatasetTemplate class,
        this function should aim to (1) extract the dataset, and (2) apply any
        dataset-specific transformations. To this end, the NSL-KDD is (1)
        packaged as a zip into multiple files in both plaintext and ARFF
        formats (we will use the plaintext to support label transformation),
        and (2) the last column (i.e. "difficulty") needs to be removed and the
        second to last column (i.e., the attack) needs to be transformed into
        an attack category (as is reported in nearly all papers that use the
        NSL-KDD). The transformation is defined as:

        Attack - Attack Category
        DoS    - {apache, back, land, neptune, pod, processtable, smurf,
                    teardrop, udpstorm, worm}
        Probe  - {ipsweep, mscan, nmap, portsweep, saint, satan}
        R2L    - {ftp_write, guess_password, httptunnel, imap, named, multihop,
                    phf, sendmail, snmpgetattack, snmpguess, spy, warezclient,
                    warezmaster, xlock, xsnoop}
        U2R    - {buffer_overflow, loadmodule, perl, ps, rootkit, xterm,
                    sqlattack}

        :param data: the data to process
        :type data: dataset-specific
        :param directory: directory to save the datasets to
        :type directory: string
        :return: santized data
        :rtype: dataset-specific
        """
        with zipfile.ZipFile(io.BytesIO(data)) as zipped:
            for file in self.files:
                with zipped.open(file) as dataset:

                    # unzip and read into dataframe
                    df = pandas.read_csv(dataset, header=None)

                    # drop the last column, apply label mapping and save
                    df.drop(columns=df.columns[-1], inplace=True)
                    df.replace({df.columns[-1]: self.transform}, inplace=True)
                    df.to_csv(
                        pathlib.Path(directory, type(self).__name__, file), index=False
                    )

        # return the zip so that it is saved to disk
        return data

    def read(self, directory="/tmp/"):
        """
        As described in the read comments for the DatasetTemplate class, this
        function adheres to the provided standard, namely: (1) labels are
        encoded as 'labels', (2) the training and test sets are encoded as
        'train' and 'test', respectively, and (3) the data is returned as a
        numpy array.

        :param directory: directory where the dataset is downloaded
        :type directory: string
        :return: the downloaded datasets
        :rtype: dictionary; keys are the dataset types & values are numpy arrays
        """
        dataset = {}
        for file in self.files:

            # read in the data, split labels, and return as a dictionary
            df = pandas.read_csv(pathlib.Path(directory, type(self).__name__, file))
            dataset[self.categories[file]] = {
                "data": df.drop(columns=df.columns[-1]).to_numpy(),
                "labels": df.filter([df.columns[-1]]).to_numpy(),
            }
        return dataset


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
