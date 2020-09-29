""" 
This module contains different attribute preprocessing and utility functions.
The first few functions are simply scikit-learn wrappers, while the others are
one-off pieces of code to handle any dataset particulars.
"""


def encode_attributes(x, features):
    """
    Encodes categorical features as a one-hot array
    """
    from numpy import concatenate
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    encoder = ColumnTransformer(
        [("", OneHotEncoder(sparse=False), features)], n_jobs=-1
    )
    return concatenate(
        (x[:, : features[0]], encoder.fit_transform(x), x[:, features[-1] + 1 :]),
        axis=1,
    )


def encode_labels(x, features):
    """
    Maps strings to integers
    """
    from numpy import concatenate
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    encoder = ColumnTransformer([("", OrdinalEncoder(), features)], n_jobs=-1)
    x[:, features] = encoder.fit_transform(x)
    return x


def can_cast(x):
    """
    Checks if x can be cast as a float
    """
    import numpy as np

    try:
        x.astype(np.float)
        return True
    except ValueError:
        return False


def nslkdd():
    """
    Transforms the default labels of the NSLKDD from specific attacks to
    attack categories
    """
    from numpy import argwhere, isin, load, save
    from os import listdir

    # mappings are benign, dos, probe, r2l, and u2r
    mappings = [
        [16],
        [1, 8, 14, 19, 27, 32, 0, 33, 21, 10],
        [25, 7, 15, 20, 11, 24],
        [4, 3, 6, 18, 12, 35, 34, 30, 37, 38, 29, 28, 26, 13, 36],
        [2, 9, 23, 17, 31, 39, 22, 5],
    ]
    datasets = listdir("../datasets/nslkdd/numpy/")
    for dataset in datasets:
        data = load("../datasets/nslkdd/numpy/" + dataset)
        indicies = []
        for new_label, old_label in enumerate(mappings):
            indicies.append((argwhere(isin(data[:, -1], old_label)), new_label))
        for index, new_label in indicies:
            data[index, -1] = new_label
        save("../datasets/nslkdd/numpy/" + dataset, data)
    datasets = listdir("../datasets/slimkdd/numpy/")
    for dataset in datasets:
        data = load("../datasets/slimkdd/numpy/" + dataset)
        indicies = []
        for new_label, old_label in enumerate(mappings):
            indicies.append((argwhere(isin(data[:, -1], old_label)), new_label))
        for index, new_label in indicies:
            data[index, -1] = new_label
        save("../datasets/slimkdd/numpy/" + dataset, data)
    return
