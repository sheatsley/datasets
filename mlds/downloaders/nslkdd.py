"""
This module downloads the NSL-KDD.
"""
import io
import pathlib
import zipfile

import mlds.downloaders
import pandas


def retrieve(binary=False, directory=pathlib.Path("/tmp/nslkdd"), force=False):
    """
    This function downloads, preprocesses, and saves the NSL-KDD dataset
    (https://www.unb.ca/cic/datasets/nsl.html). Specifically, this: (1)
    downloads the dataset, (2) drops the last column (i.e., "difficulty"), (3)
    extracts feature names, and (4) applies a common label transformation (if
    using multiclass labels) which bundles specific attacks into families,
    defined as:

        DoS   - {apache, back, land, neptune, pod, processtable, smurf,
                teardrop, udpstorm, worm}
        Probe - {ipsweep, mscan, nmap, portsweep, saint, satan}
        R2L   - {ftp_write, guess_password, httptunnel, imap, named, multihop,
                phf, sendmail, snmpgetattack, snmpguess, spy, warezclient,
                warezmaster, xlock, xsnoop}
        U2R   - {buffer_overflow, loadmodule, perl, ps, rootkit, xterm,
                sqlattack}

    :param binary: whether to use the binary or multiclass labels
    :type binary: bool
    :param directory: directory to download the datasets to
    :type directory: str
    :param force: redownload the data, even if it exists
    :type force: bool
    :return: the NSL-KDD dataset
    :rtype: dict
    """

    # define where to download the dataset and what needs extracted
    urls = (
        "https://github.com/Jehuty4949/NSL_KDD/raw/master/"
        "Original%20NSL%20KDD%20Zip.zip",
    )
    files = (("train", "KDDTrain+.txt"), ("test", "KDDTest+.txt"))
    feature_file = "KDDTest-21.arff.txt"

    # define the label transformation
    label_transform = (
        {}.fromkeys(
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
                "ipsweep",
                "mscan",
                "nmap",
                "portsweep",
                "saint",
                "satan",
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
                "buffer_overflow",
                "loadmodule",
                "perl",
                "ps",
                "rootkit",
                "xterm",
                "sqlattack",
            ),
            "attack",
        )
        if binary
        else (
            {}.fromkeys(
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
            )
            | {}.fromkeys(
                (
                    "ipsweep",
                    "mscan",
                    "nmap",
                    "portsweep",
                    "saint",
                    "satan",
                ),
                "probe",
            )
            | {}.fromkeys(
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
            )
            | {}.fromkeys(
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
            )
        )
    )

    # retrieve dataset, get feature names, drop last column, and fix labels
    dataset = {}
    download = mlds.downloaders.download(directory=directory, force=force, urls=urls)
    _, download = download.popitem()
    print("Processing the NSL-KDD...")
    with zipfile.ZipFile(io.BytesIO(download)) as zipped:
        with io.TextIOWrapper(zipped.open(feature_file)) as datafile:
            features = [line.split("'")[1] for line in datafile.readlines()[1:43]]

        for partition, file in files:
            with io.TextIOWrapper(zipped.open(file)) as datafile:
                df = pandas.read_csv(datafile, header=None)
            df.drop(columns=df.columns[-1], inplace=True)
            df.replace(to_replace={df.columns[-1]: label_transform}, inplace=True)
            df.columns = features
            data = df.drop(columns="class")
            labels = df["class"].copy()
            dataset[partition] = {"data": data, "labels": labels}
    return dataset
