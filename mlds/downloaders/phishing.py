"""
This module downloads the Phishing dataset.
"""
import pathlib

import mlds.downloaders
import pandas


def retrieve(directory=pathlib.Path("/tmp/phishing"), force=False):
    """
    This function downloads, preprocesses, and saves the Phishing dataset.
    (https://www.fcsit.unimas.my/phishing-dataset). Specifically, this: (1)
    downloads the dataset, and (2) extracts feature names.

    :param directory: directory to download the datasets to
    :type directory: str
    :param force: redownload the data, even if it exists
    :type force: bool
    :return: the Phishing dataset
    :rtype: dict
    """

    # define where to download the dataset
    urls = (
        "https://data.mendeley.com/public-files/datasets/h3cgnj8hft/files/"
        "84a399ef-c57e-4ee6-9207-7bffb5ace261/file_downloaded",
    )

    # retrieve the dataset and get feature names
    dataset = {}
    download = mlds.downloaders.download(directory=directory, force=force, urls=urls)
    _, download = download.popitem()
    print("Processing the Phishing dataset...")
    datafile = download.decode().splitlines()
    features = [line.split()[1] for line in datafile[2:51]]
    df = pandas.DataFrame([x.split(",") for x in datafile[53:]], columns=features)
    data = df.drop(columns="CLASS_LABEL")
    labels = df["CLASS_LABEL"].copy()
    dataset["dataset"] = {"data": data, "labels": labels}
    return dataset
