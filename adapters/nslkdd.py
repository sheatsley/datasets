"""
This module defines a custom adapter for downloading the NSL-KDD.
Author: Ryan Sheatsley
Wed Feb 9 2022
"""
from adapters import baseadapter  # Base Adapter class for custom datasets
import io  # Core tools for working with streams
import pandas  # Python Data Analysis Library
import zipfile  # Work with ZIP archives


class NSLKDD(baseadapter.BaseAdapter):
    """
    This class adds support for downloading, preprocessing, and saving the
    NSL-KDD (https://www.unb.ca/cic/datasets/nsl.html). It inherits the following
    interfaces from the BaseAdapter parent class:

    :func:`download`: retrieves datasets via HTTP through the requests module

    Moreover, it redefines the following interfaces:

    :func:`__init__`: instantiates NSLKDD objects
    :func:`preprocess`: resolves any dataset particulars
    :func:`read`: ensures the dataset conforms to the required standard

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
        self.url = "http://205.174.165.80/CICDataset/NSL-KDD/Dataset/NSL-KDD.zip"
        self.files = ("KDDTrain+.txt", "KDDTest+.txt")
        self.fnames = "KDDTest-21.arff"

        # label transformation scheme
        self.transform = {
            old_label: new_label
            for family, new_label in (
                (
                    (
                        "apache",
                        "apache2",
                        "back",
                        "land",
                        "mailbomb",
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
                        "guess_passwd",
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
        files for the NSL-KDD,  and (2) apply the desired transformations
        (i.e., dropping the last "difficulty" column and applying the label
        transformation defined above to the second-to-last "attack" column).

        :param data: the current dataset file
        :type data: bytes
        :return: santized data file
        :rtype: pandas dataframe
        """
        with zipfile.ZipFile(io.BytesIO(data)) as zipped:

            # extract feature names from any ARFF file first
            print(f"Extracting feature names from {self.fnames}...")
            with io.TextIOWrapper(zipped.open(self.fnames)) as datafile:
                features = []
                for line in datafile.readlines()[1:43]:
                    features.append(line.split("'")[1])

            for file in self.files:
                print(f"Processing {file}...")
                with io.TextIOWrapper(zipped.open(file)) as datafile:

                    # drop the last column and apply label mapping
                    print("Loading & applying transformations...")
                    df = pandas.read_csv(datafile, header=None)
                    df.drop(columns=df.columns[-1], inplace=True)
                    df.replace({df.columns[-1]: self.transform}, inplace=True)

                    # add feature names as the column header
                    print("Setting feature names as column header...")
                    df.columns = features
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
            partition: {"data": df.drop(columns="class"), "labels": df["class"].copy()}
            for partition, df in zip(
                ("train", "test"), self.preprocess(self.download(self.url))
            )
        }


if __name__ == "__main__":
    """
    Example usage of the NSLKDD class. It downloads the NSL-KDD (if it hasn't
    already) to /tmp, preprocesses the data, and returns the data as a pandas
    dataframe. Importantly, to debug adapters, they must be run form the root
    directory of the machine learning datasets repo as a module, such as:

                            python3 -m adapters.nslkdd
    """
    dataset = NSLKDD().read()
    print(
        f'NSL-KDD has {len(dataset["train"]["data"])} training samples,',
        f'{len(dataset["test"]["data"])} test samples, and',
        f'{len(dataset["test"]["data"].columns)} features with',
        f'{len(dataset["test"]["labels"].unique())} classes.',
    )
    raise SystemExit(0)
