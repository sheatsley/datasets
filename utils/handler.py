"""
The Handler class supports flexibly preprocessing unique dataset requirements.
Author: Ryan Sheatsley
Thu May 14 2020
"""

import arff
import copy
import numpy as np
import pathlib
from utilities import encode_attributes, encode_labels, can_cast
import scale
import pickle


class Handler:
    def load(self, path, header, include, pickled):
        """
        Loads original dataset into memory.
        """
        try:
            with open(path, "rb" if pickled else "r") as f:
                if pickled:
                    dict = pickle.load(f, encoding="bytes")
                    return np.concatenate(
                        (
                            np.array(dict[b"data"])[:, include],
                            np.expand_dims(np.array(dict[b"labels"]), axis=1),
                        ),
                        axis=1,
                    )
                elif "arff" == path.split(".")[-1]:
                    return np.array(arff.load(f)["data"], dtype=np.unicode_)[:, include]
                else:
                    return np.genfromtxt(
                        f,
                        delimiter=",",
                        dtype=np.unicode_,
                        skip_header=header,
                        usecols=include,
                    )
        except OSError as e:
            print(path, e.strerror)
            return -1

    def preprocess(self, x, special=True, onehot=(), preserve=(), **kwargs):
        """
        Manipulates attributes based on arguments
        """

        # encode labels as ints (if not numerical) & convert relative indicies to absolute
        if not can_cast(
            x[:, [label if label >= 0 else label + x.shape[1] for label in preserve]]
        ):
            x = encode_labels(
                x, [label if label >= 0 else label + x.shape[1] for label in preserve]
            )

        # encode categorical attributes as one-hot vectors (must be sequential)
        if onehot:

            # save a special representation of the data with categoricals encoded as ints
            if special:
                self.save(
                    [[encode_labels(copy.copy(x), onehot).astype(np.float32)]],
                    kwargs["path"],
                    "special",
                    kwargs["test"],
                    **{"size": kwargs["size"]} if kwargs["test"] else {},
                )
            x = encode_attributes(x, onehot)
        return x.astype(np.float32)

    def normalize(self, x, scheme, preserve, **kwargs):
        """
        Scales a dataset based on the scheme
        """

        # compute features that will be normalized and convert any relative indicies to absolute
        features = tuple(
            (
                feature
                for feature in range(x.shape[1])
                if feature not in set(p if p >= 0 else p + x.shape[1] for p in preserve)
            )
        )
        try:
            return (
                [
                    getattr(scale, s)(copy.copy(x), features, **kwargs)
                    for s in dir(scale)
                    if not s.startswith("_")
                ]
                if scheme == "all"
                else [getattr(scale, scheme)(x, features, **kwargs)]
            )
        except AttributeError as e:
            print(scheme, e.strerror)
            return -1

    def save(self, x, path, scheme, test, concat=False, **kwargs):
        """
        Saves dataset arrays as binary files with .npy format
        """

        # check what scaling functions are currently in the module
        schema = (
            [s for s in dir(scale) if not s.startswith("_")]
            if scheme == "all"
            else [scheme]
        )

        # save the datasets (name parsing is reliant on the existence of an extension)
        base = path[0].split("/")[0] + "/numpy/"
        datasets = [p.split("/")[-1].split(".")[0] for p in path]
        pathlib.Path(base).mkdir(parents=True, exist_ok=True)
        if test:
            [
                (
                    np.save(
                        (base + datasets[0] + "_" + s).lower(),
                        x[0][idy][: kwargs["size"]],
                    ),
                    np.save(
                        (base + datasets[-1] + "_" + s).lower(),
                        x[0][idy][kwargs["size"] :],
                    ),
                )
                for idy, s in enumerate(schema)
            ]
        else:
            [
                np.save((base + d + "_" + s).lower(), x[idx][idy])
                for idx, d in enumerate(datasets)
                for idy, s in enumerate(schema)
            ]
        return 0

    def prep(self, opts, save=True):
        """
        Calls handler functions to prepare a dataset
        """
        if opts["test"]:
            x = [
                self.normalize(
                    self.preprocess(
                        np.concatenate(
                            [
                                self.load(
                                    path,
                                    opts["header"],
                                    opts["include"],
                                    opts["pickled"],
                                )
                                for path in opts["path"]
                            ]
                        ),
                        **opts,
                    ),
                    **opts,
                )
            ]
        else:
            x = [
                self.normalize(
                    self.preprocess(
                        self.load(
                            path, opts["header"], opts["include"], opts["pickled"]
                        ),
                        **opts,
                    ),
                    **opts,
                )
                for path in opts["path"]
            ]
        if save:
            self.save(x, **opts)
        return x


if __name__ == "__main__":
    """
    Convert and save all available datasets
    Parameters:
    - header: whether or not a header row exists (which will be removed)
    - include: only load the following columns (affects indicies for other parameters)
    - label: index of the label
    - norm: range of features to be norm'd together for unit_norm scaling scheme
    - onehot: list of categorical features to convert to onehot vectors
    - path: path to the dataset to load
    - preserve: do not modify these indicies for any scaling schemes (ie labels)
    - scheme: feature scaling schema (definitions in scale module)
    - size: if there is a test set, size is the number of training samples
    - test: whether or not a dedicated test set exists
    """

    handler = Handler()
    """
        "slimkdd": {
            "header": False,
            "include": (4, 30, 5, 25, 26, 39, 38, 6, 29, 12, 3, 41),
            "onehot": (0,),
            "path": ("slimkdd/original/KDDTrain+.txt", "slimkdd/original/KDDTest+.txt"),
            "preserve": (-1,),
            "scheme": "all",
            "size": 125973,
            "test": True,
        },
    """
    opts = {
        "cifar10": {
            "header": False,
            "include": tuple(x for x in range(3072)),
            "path": (
                "cifar-10-batches-py/data_batch_1",
                "cifar-10-batches-py/data_batch_2",
                "cifar-10-batches-py/data_batch_3",
                "cifar-10-batches-py/data_batch_4",
                "cifar-10-batches-py/data_batch_5",
                "cifar-10-batches-py/test_batch",
            ),
            "pickled": True,
            "concat": True,
            "preserve": (-1,),
            "size": 50000,
            "scheme": "all",
            "test": True,
        },
        "phishing": {
            "header": False,
            "include": (4, 13, 24, 26, 33, 34, 38, 44, 46, 47, 48),
            "onehot": (7, 8, 9),
            "path": ("phishing/original/Phishing_Legitimate_full.arff",),
            "pickled": False,
            "preserve": (-1,),
            "scheme": "all",
            "test": False,
        },
        "dgd-4": {
            "header": True,
            "include": tuple(x for x in range(6)),
            "norm": range(4),
            "path": (
                "dgd/original/Exp_NoObst_124.csv",
                "dgd/original/Exp_Obst_124.csv",
            ),
            "pickled": False,
            "preserve": (-1, -2),
            "scheme": "all",
            "test": False,
        },
        "dgd-8": {
            "header": True,
            "include": tuple(x for x in range(10)),
            "norm": range(8),
            "path": (
                "dgd/original/FixedObstruction_e6.csv",
                "dgd/original/FixedObstruction_e7.csv",
                "dgd/original/OriginalDataset_e6.csv",
                "dgd/original/RandomObstruction_e6.csv",
                "dgd/original/RandomObstruction_e7.csv",
            ),
            "pickled": False,
            "preserve": (-1, -2),
            "scheme": "all",
            "test": False,
        },
        "drebin": {
            "header": False,
            "include": tuple(x for x in range(9)),
            "path": ("drebin/original/drebin.csv",),
            "pickled": False,
            "preserve": (-1,),
            "scheme": "all",
            "test": False,
        },
        "nslkdd": {
            "header": False,
            "include": tuple(
                x
                for x in range(43)
                if x
                not in set(
                    (
                        19,
                        42,
                    )
                )
            ),
            "onehot": (1, 2, 3),
            "path": ("nslkdd/original/KDDTrain+.txt", "nslkdd/original/KDDTest+.txt"),
            "pickled": False,
            "preserve": (-1,),
            "scheme": "all",
            "size": 125973,
            "test": True,
        },
        "unswnb15": {
            "header": True,
            "include": tuple(x for x in range(45) if x not in set((0, 44))),
            "onehot": (1, 2, 3),
            "path": (
                "unswnb15/original/UNSW_NB15_training-set.csv",
                "unswnb15/original/UNSW_NB15_testing-set.csv",
            ),
            "pickled": False,
            "preserve": (-1,),
            "scheme": "all",
            "size": 175341,
            "test": True,
        },
    }
    for dataset in opts:
        handler.prep(opts[dataset])
        print(dataset, "converted")
    raise SystemExit(0)
