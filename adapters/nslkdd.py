"""
This module defines a custom adapter for downloading the NSL-KDD.
Author: Ryan Sheatsley
Wed Feb 9 2022
"""
import collections  # Container datatypes
import io  # Core tools for working with streams
import pandas  # Python Data Analysis Library
import pathlib  # Object-oriented filesystem paths
import retrieve  # Wrappers for popular machine learning datasets
from utilities import print  # Timestamped printing
import zipfile  # Work with ZIP archives


class NSLKDD(retrieve.BaseAdapter):
    """
    This class adds support for downloading, preprocessing, and saving the
    NSL-KDD (https://www.unb.ca/cic/datasets/nsl.html). It inherits the following
    interfaces from the BaseAdapter parent class:

    :func:`download`: retrieves datasets via HTTP through the requests module

    Moreover, it redefines the following interfaces:

    :func:`__init__`: instantiates NSL_KDD objects
    :func:`preprocess`: resolves any dataset particulars
    :func:`read`: reads the dataset into memory

    Finally, since the NSL-KDD is avaiable as ARFF, feature names are extracted
    and made available for applying feature-specific transformations by name
    (as opposed to exclusively by index).
    """

    def __init__(self, directory="/tmp/", force_download=False):
        """
        Aside from the standard pattern of retrieving files that define the
        dataset, processing the NSL-KDD is unique in that: (1) the last column
        (i.e., "difficulty") is necessarily dropped, and (2) a common label
        transformation is applied which bundles specific attacks into families,
        defined as:

        Attack - Attack Category
        DoS    - {apache, back, land, neptune, pod, processtable, smurf,
                    teardrop, udpstorm, worm}
        Probe  - {ipsweep, mscan, nmap, portsweep, saint, satan}
        R2L    - {ftp_write, guess_password, httptunnel, imap, named, multihop,
                    phf, sendmail, snmpgetattack, snmpguess, spy, warezclient,
                    warezmaster, xlock, xsnoop}
        U2R    - {buffer_overflow, loadmodule, perl, ps, rootkit, xterm,
                    sqlattack}

        :param directory: dataset download directory
        :type directory: string
        :return: NSL-KDD-ready adapter
        :rtype: NSLKDD object
        """
        super().__init__(directory, force_download)

        # metadata to retrieve the dataset
        self.urls = ("http://205.174.165.80/CICDataset/NSL-KDD/Dataset/NSL-KDD.zip",)
        self.files = ("KDDTrain+.arff", "KDDTest+.arff")
        self.categories = dict(zip(self.files, ("train", "test")))

        # label transformation scheme
        self.transform = {
            old_label: new_label
            for family, new_label in (
                (
                    (
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
                    ),
                    "dos",
                ),
                (
                    ("ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"),
                    "probe",
                ),
                (
                    (
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
                    ),
                    "r2l",
                ),
                (
                    (
                        "buffer_overflow",
                        "loadmodule",
                        "perl",
                        "ps",
                        "rootkit",
                        "xterm",
                        "sqlattack",
                    ),
                    "u2r",
                ),
            )
            for old_label in family
        }
        return None

    def preprocess(self, data):
        """
        Conforming to the BaseAdapter guidelines, we: (1) unzip the desired
        files for the NSL-KDD (ARFF), and (2) apply the desired transformations
        (i.e., dropping the last "difficulty" column and applying the label
        transformation defined above to the second-to-last "attack" column).

        :param data: the dataset (as partitions)
        :type data: list of bytes
        :return: santized data
        :rtype: dataset-specific
        """

        # we 0-index because data is expected to be a list of partitions
        with zipfile.ZipFile(io.BytesIO(data[0])) as zipped:
            for file in self.files:
                print(f"Processing {file}...")
                with zipped.open(file) as dataset:

                    # unzip and read into dataframe
                    df = pandas.read_csv(dataset, header=None)

                    # drop the last column, apply label mapping and save
                    df.drop(columns=df.columns[-1], inplace=True)
                    df.replace({df.columns[-1]: self.transform}, inplace=True)
                    df.to_csv(
                        pathlib.Path(self.directory, type(self).__name__, file),
                        index=False,
                    )

        # return the zip so that it is saved to disk
        return data

    def read(self, directory="/tmp/"):
        """
        The BaseAdapter specification requires that: (1) labels and data are
        encoded as "labels" and "data", respectively, (2) the training and test
        sets are encoded as "train", and "test", respectively, (3) if
        available, feature names are exposed through a "name_map" key, and (4)
        the data is returned as a pandas dataframe.

        :return: the downloaded datasets as a pandas dataframe
        :rtype: dictionary; keys are the dataset types & values are dataframes
        """
        dataset = {}
        for file in self.files:

            # read in the data, split labels, and return as a dictionary
            df = pandas.read_csv(pathlib.Path(directory, type(self).__name__, file))
            dataset[self.categories[file]] = {
                "data": df.drop(columns=df.columns[-1]),
                "labels": df.filter([df.columns[-1]]),
            }
        return dataset


if __name__ == "__main__":
    """
    Example usage of the NSLKDD class. It downloads the NSL-KDD (if it hasn't
    already) to /tmp, preprocesses the data, and returns the data as a pandas
    dataframe. Importantly, this must be run from the root directory of the machine
    learning datasets repo as a module (as the BaseAdapter definition is found
    there). That is, if this file is ran directly, it must done so as:

                            python3 -m adapters.nslkdd
    """
    NSLKDD().read()
    raise SystemExit(0)
