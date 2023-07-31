"""
This module downloads the CIC-MalMem2022.
"""
import pathlib

import pandas


def retrieve(directory=pathlib.Path("/tmp/cicmalmem2022"), force=False):
    """
    This function downloads, preprocesses, and saves the CIC-MalMem-2022
    (https://www.unb.ca/cic/datasets/malmem-2022.html). Specifically, this: (1)
    downloads the dataset, (2) drops the last column (which is used for anomaly
    detection; the first column is used for signature detection, which is
    subsequently used as the label instead), and (3) parses the malware family
    from the labels (as samples are encoded as {category}-{family}-{sha256}).

    :param directory: directory to download the datasets to
    :type directory: str
    :param force: redownload the data, even if it exists
    :type force: bool
    :return: the CIC-MalMem-2022
    :rtype: dict
    """

    # define where to download the dataset
    url = (
        "http://205.174.165.80/CICDataset/CICMalMem2022/Dataset/"
        "Obfuscated-MalMem2022.csv"
    )

    # retrieve the dataset, drop last column, and fix labels
    dataset = {}
    directory.mkdir(parents=True, exist_ok=True)
    try:
        if force:
            raise FileNotFoundError
        df = pandas.read_csv(directory / "Obfuscated-MalMem2022.csv")
    except FileNotFoundError:
        df = pandas.read_csv(url)
        df.to_csv(directory / "Obfuscated-MalMem2022.csv", index=False)
    print("Processing the CIC-MalMem-2022...")
    df.drop(columns="Class", inplace=True)
    df.Category = df.Category.str.split("-").str[0]
    data = df.drop(columns="Category")
    labels = df["Category"].copy()
    dataset["dataset"] = {"data": data, "labels": labels}
    return dataset
