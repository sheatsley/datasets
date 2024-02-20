"""
This module lists available datasets and defines the downloader function.
"""

import requests
from mlds.downloaders import (
    cicmalmem2022,
    cifar10,
    fashionmnist,
    mnist,
    nslkdd,
    phishing,
    unswnb15,
)

__all__ = [
    "cicmalmem2022",
    "cifar10",
    "fashionmnist",
    "nslkdd",
    "mnist",
    "phishing",
    "unswnb15",
]


def download(directory, force, urls):
    """
    This function serves as a helper function for all dataset modules by
    downloading the dataset from a set of URLs. Specifically, it: (1) creates
    the specified directory (if it doesn't exist), (2) downloads and caches the
    dataset (if it hasn't already been downloaded or if force is True), and (3)
    returns the downloaded file.

    :param directory: directory to download the datasets to
    :type directory: pathlib Path object
    :param force: redownload the data, even if it exists
    :type force: bool
    :param urls: URLs to download the dataset from
    :type urls: tuple of str
    :return: dowloaded dataset paired with URLs
    :rtype: dict
    """
    dataset = {}
    directory.mkdir(parents=True, exist_ok=True)
    for url in urls:
        data = directory / url.split("/").pop()
        if not data.is_file() or force:
            print(f"Downloading {url} to {directory}...")
            with requests.Session() as session:
                retries = requests.adapters.Retry(total=3, status_forcelist=[54])
                session.mount(url, requests.adapters.HTTPAdapter(max_retries=retries))
                response = session.get(url)
                response.raise_for_status()
                data.write_bytes(response.content)
        dataset[url] = data.read_bytes()
    return dataset
