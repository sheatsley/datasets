"""
This module downloads the UNSW-NB15.
"""

import pathlib

import pandas


def retrieve(binary=False, directory=pathlib.Path("/tmp/unswnb15"), force=False):
    """
    This function downloads, preprocesses, and saves the UNSW-NB15
    (https://research.unsw.edu.au/projects/unsw-nb15-dataset). Specifically,
    this: (1) downloads the dataset, (2) drops the first (i.e., 'id') and last
    column (i.e., 'label' which is used for anomaly detection; the second to
    last column, i.e., 'attack_cat', is used for signature detection, which is
    subsequently used as the label instead), (3) drops null values, and (4)
    applies a series of transformations to categorical variables, specifically:

        states   - {ACC, CLO, PAR, URN, no}
        protocol - {rtp}

    Connection states 'ACC' and 'CLO' are only present in the test set (which
    causes one-hot-encoding to fail), while states 'PAR', 'URN', and 'no' are
    only found in one sample each in the training set (and not present in the
    test set). Network protocol 'rtp' is found in one sample in the training
    set (and not present in the test set). Notably, the UNSW-NB15 training and
    test sets are mistakenly labeled in reverse.

    :param binary: whether to use the binary or multiclass labels
    :type binary: bool
    :param directory: directory to download the datasets to
    :type directory: str
    :param force: redownload the data, even if it exists
    :type force: bool
    :return: the CIC-MalMem-2022
    :rtype: dict
    """

    # define where to download the dataset and designate partitions
    urls = tuple(
        "https://github.com/ColdAsYou165/dataset_UNSWNB15/raw/main/"
        f"%E5%8E%9F%E5%A7%8B%E6%95%B0%E6%8D%AE%E9%9B%86/UNSW_NB15_{partition}-set.csv"
        for partition in ("testing", "training")
    )

    # note that the training and test sets are mistakenly labeled in reverse
    files = (
        ("train", "UNSW_NB15_testing-set.csv"),
        ("test", "UNSW_NB15_training-set.csv"),
    )

    # define invalid feature values
    invalid_features = {"CLO", "ACC", "PAR", "URN", "no", "rtp"}

    # retrieve the dataset, drop columns, nulls, and invalid features
    dataset = {}
    dataframes = []
    label = ("attack_cat", "label")
    directory.mkdir(parents=True, exist_ok=True)
    for (partition, file), url in zip(files, urls):
        try:
            if force:
                raise FileNotFoundError
            df = pandas.read_csv(directory / file)
        except FileNotFoundError:
            df = pandas.read_csv(url)
            df.to_csv(directory / file, index=False)
        dataframes.append(df)
    print("Processing the UNSW-NB15...")
    for (partition, file), df in zip(files, dataframes):
        df.drop(columns=["id", label[~binary]], inplace=True)
        df.dropna(inplace=True)
        df = df[~df.isin(invalid_features).any(axis=1)]
        df.reset_index(drop=True, inplace=True)
        data = df.drop(columns=label[binary])
        labels = df[label[binary]].copy()
        dataset[partition] = {"data": data, "labels": labels}
    return dataset
