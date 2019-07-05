"""

Most datasets are uniquely preprocessed. This module defines a generic handler
class which handles unique preprocessing requirements flexibly.

"""
import csv
import numpy as np
import pathlib
from preprocess import encode
import scale


class Handler:
    def load(self, path, header=False, onehot=False, categorical=[], **kwargs):
        """
        Loads original dataset into memory
        """
        try:
            with open(path) as f:

                # infer the type
                ext = path.split(".")[-1]
                if ext in ("txt", "csv"):
                    x = list(csv.reader(f, delimiter=","))

                    # strip the header if it exists
                    if header:
                        del x[0]

                    # convert categorical attributes to one-hot vectors
                    if onehot:
                        x = encode(x, categorical)
                    return np.array((x))
                else:
                    print(ext, "file format not recognized/supported")
        except OSError as e:
            print(e.strerror)
            return -1

    def scale(self, x, scheme, exclude, options=None, **kwargs):
        """
        Scales a dataset based on the scheme (with options)
        Available options:
        """

        # ignore excluded attributes any scaling

        if scheme == "all":
            x_scale = [
                getattr(scale, s)(x[:, :labels])
                for s in dir(scale)
                if not s.startswith("_")
            ]
            return [np.concatenate((xs, x[:, labels:])) for xs in x_scale]
        else:
            try:
                x_scale = [getattr(scale, scheme)(x[:, :labels], **options)]
                return [np.concatenate((*x_scale, x[:, labels:]))]
            except AttributeError:
                print(scheme, "not found")
                return -1

    def save(self, x, org_path, scheme, test):
        """
        Saves dataset arrays as binary files with .npy format
        """

        # check what scaling functions are currently in the module
        if scheme == "all":
            schema = [s for s in dir(scale) if not s.startswith("_")]
        else:
            schema = [scheme]

        # every original dataset should have an extension (and no other periods)
        if test:
            save_path = org_path[0].split("/")[0] + "/numpy/"
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            train = org_path[0].split("/")[-1].split(".")[0]
            train = org_path[1].split("/")[-1].split(".")[0]
            for idx, s in enumerate(schema):
                np.save(save_path.lower() + train + "_" + s, x[idx][0])
                np.save(save_path.lower() + test + "_" + s, x[idx][1])
        else:
            save_path = org_path.split("/")[0] + "/numpy/"
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            dataset = org_path.split("/")[-1].split(".")[0]
            for s in schema:
                np.save(save_path.lower() + dataset + "_" + s, x)
        return 0

    def prep(self, opts, save=True):
        """
        Calls handler functions to prepare a dataset
        """

        # operate pair-wise if there's a dedicated test set
        if opts["test"]:
            x = []
            paths = opts["path"]
            for p in opts["path"]:
                opts["path"] = p
                x.append(self.load(**opts))
            opts["path"] = paths
            x = self.scale(x, **opts)
            if save:
                self.save(x, opts["path"], opts["scheme"], True)

        # otherwise, go dataset-by-dataset
        else:
            paths = opts["path"]
            for p in paths:
                opts["path"] = p
                x = self.load(**opts)
                x = self.scale(x, **opts)
                if save:
                    self.save(x, p, opts["scheme"], False)
        return x


if __name__ == "__main__":
    """
    Convert and save all available datasets
    Parameters:
    - path: path to the dataset to load
    - test: whether or not a dedicated test set exists
    - header: whether or not a header row exists (which will be removed)
    - onehot: if categorical features should be converted to onehot vectors
    - categorical: list of categorical indicies to onehot encode
    - scheme: feature scaling schema (schema listed above)
    - exclude: exclude features from any feature scaling (post-onehot indicies)
    """
    import os

    handler = Handler()
    os.chdir("..")
    opts = {
        "dgd": {
            "path": (
                "dgd/original/FixedObstruction_e6.csv",
                "dgd/original/FixedObstruction_e7.csv",
                "dgd/original/OriginalDataset_e6.csv",
                "dgd/original/RandomObstruction_e6.csv",
                "dgd/original/RandomObstruction_e7.csv",
            ),
            "test": False,
            "header": True,
            "scheme": "all",
            "exlude": [[-4], [-4], [-2], [-4], [-4]]
        },
        "nslkdd": {
            "path": ("nslkdd/original/KDDTrain+.txt", "nslkdd/original/KDDTest+.txt"),
            "test": True,
            "onehot": True,
            "categorical": [1, 2, 3],
            "scheme": "all",
            "exclude": [-2]
        },
        "unswnb15": {
            "path": (
                "unswnb15/original/UNSW_NB15_training-set.csv",
                "unswnb15/original/UNSW_NB15_testing-set.csv",
            ),
            "test": True,
            "onehot": True,
            "categorical": [2, 3, 4],
            "scheme": "all",
            "exclude": [1]
        },
    }
    for dataset in opts:
        handler.prep(opts[dataset])
        print(dataset, "converted")
    raise SystemExit(0)
