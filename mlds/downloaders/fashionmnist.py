"""
This module downloads the Fashion-MNIST dataset.
"""
import pathlib

import pandas
import tensorflow_datasets


def retrieve(directory=pathlib.Path("/tmp/fashionmnist"), force=False):
    """
    This function downloads, preprocesses, and saves the MNIST database.
    (https://github.com/zalandoresearch/fashion-mnist). Specifically, this: (1)
    downloads the dataset, (2) converts columns to strings, and (3) flattens
    the images.

    :param directory: directory to download the datasets to
    :type directory: str
    :param force: redownload the data, even if it exists
    :type force: bool
    :return: the Fashion-MNIST dataset
    :rtype: dict
    """

    # define partitions
    partitions = ("train", "test")

    # retrieve the dataset and get feature names
    dataset = {}
    download_config = tensorflow_datasets.download.DownloadConfig(
        download_mode=tensorflow_datasets.GenerateMode.REUSE_CACHE_IF_EXISTS
        if force
        else tensorflow_datasets.GenerateMode.REUSE_DATASET_IF_EXISTS
    )
    download = tensorflow_datasets.load(
        name="fashion_mnist",
        batch_size=-1,
        data_dir=directory,
        download_and_prepare_kwargs=dict(download_config=download_config),
    )
    print("Processing the Fashion-MNIST dataset...")
    for partition in partitions:
        data = pandas.DataFrame(
            download[partition]["image"].numpy().reshape(-1, 784),
            columns=map(str, range(784)),
        )
        labels = pandas.Series(download[partition]["label"].numpy())
        dataset[partition] = {"data": data, "labels": labels}
    return dataset
