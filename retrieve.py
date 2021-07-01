"""
The retrieve module downloads machine learning datasets from popular
repositories.
Author: Ryan Sheatsley
Fri Jun 18 2021
"""
import itertools  # Functions creating iterations for efficient looping
import torchvision  # Datasets, transforms and Models specific to Computer Vision
from utils import print  # use timestamped print

# TODO
# - implement force re-download
# - multithread datasets
# - resolve datasets that require labels to be downloaded separetely
# - add print statements


class Downloader:
    """
    This downloader class serves as a wrapper for popular
    machine learning libraries to retrieve datasets. Moreover,
    it is designed to be easily extendable to support downloading
    datasets from ostensibly any location via the Requests module.

    :func:`__init__`: instantiates Downloader objects
    :func:`custom`: defines an interface for custom dataset downloaders
    :func:`pytorch`: retrieve datasets from torchvision
    :func:`tensorflow`: retreive datasets from tensorflow
    :func:`uci`: retreive datasets from UCI machine learning repository
    """

    def __init__(self, datasets):
        """
        This function initializes the supported datasets from PyTorch and
        TensorFlow (since their interfaces are not standardized). The supported
        datasets are encoded as dictionaries of dictionaries, with the dataset
        names as keys and, as values, a dictionary containing two keys: "name",
        which maps the dataset name to the case-sensitive module name; and,
        "split", which maps the the dataset "category" (e.g., training,
        testing, validation, landmarks, outlines, etc.) parameter and possible
        values as a tuple.

        At this time, the following datasets are not supported:
        - Pytorch
            - Cityscapes (does not support downloading)
            - MS Coco Captions (does not support downloading)
            - MS Coco (does not support downloading)
            - EMNIST (incompatible with this library)
            - Flickr8k (does not support downloading)
            - Flickr30k (does not support downloading)
            - HMDB51 (does not support downloading)
            - Kinetics-400 (incompatible with this library)
            - UCF101 (does not support downloading)

        :param datasets: datasets to download
        :type datasets: list of strings
        :return: downloader
        :rtype: Downloader object
        """
        self.datasets = datasets

        # define supported datasets
        self.pytorch_datasets = {
            "caltech101": {
                "name": "Caltech101",
                "split": ("target_type", ["category", "annotation"]),
            },
            "caltech256": {
                "name": "Caltech256",
                "split": None,
            },
            "celeba": {
                "name": "CelebA",
                "split": ("split", ["all"]),
            },
            "cifar10": {
                "name": "CIFAR10",
                "split": ("train", [True, False]),
            },
            "cifar100": {
                "name": "CIFAR100",
                "split": ("train", [True, False]),
            },
            "fakedata": {
                "name": "FakeData",
                "split": None,
            },
            "fashionmnist": {
                "name": "FashionMNIST",
                "split": ("train", [True, False]),
            },
            "imagenet": {
                "name": "ImageNet",
                "split": ("split", ["train", "val"]),
            },
            "kitti": {
                "name": "Kitti",
                "split": ("train", [True, False]),
            },
            "kmnist": {
                "name": "KMNIST",
                "split": ("train", [True, False]),
            },
            "lsun": {
                "name": "LSUN",
                "split": ("classes", ["train", "val", "test"]),
            },
            "mnist": {
                "name": "MNIST",
                "split": ("train", [True, False]),
            },
            "omniglot": {
                "name": "Omniglot",
                "split": ("background", [True, False]),
            },
            "phototour": {
                "name": "PhotoTour",
                "split": ("name", ["notredame", "yosemite", "liberty"]),
            },
            "places365": {
                "name": "Places365",
                "split": ("split", ["train-standard", "train-challenge", "val"]),
            },
            "qmnist": {
                "name": "QMNIST",
                "split": ("train", [True, False]),
            },
            "sbdtaset": {
                "name": "SBDataset",
                "split": ("image_set", ["train", "val"]),
            },
            "sbu": {
                "name": "SBU",
                "split": None,
            },
            "semeion": {
                "name": "SEMEION",
                "split": None,
            },
            "stl10": {
                "name": "STL10",
                "split": ("split", ["train", "test"]),
            },
            "svhn": {
                "name": "SVHN",
                "split": ("split", ["train", "test"]),
            },
            "usps": {
                "name": "USPS",
                "split": ("train", [True, False]),
            },
            "vocsegmentation": {
                "name": "VOCSegmentation",
                "split": ("image_set", ["train", "val"]),
            },
            "vocdetection": {
                "name": "VOCDetection",
                "split": ("image_set", ["train", "val"]),
            },
            "widerface": {
                "name": "WIDERFace",
                "split": ("split", ["train", "val", "test"]),
            },
        }

        # inititate downloads
        return None

    def custom(self, dataset):
        """"""
        return

    def pytorch(self, dataset, splits, directory="/tmp/"):
        """
        This function serves as a wrapper for torchvision.datasets
        (https://pytorch.org/vision/stable/datasets.html). While this API is
        designed to be as standardized as possible, many of the datasets
        implement their own custom API (since the parameters and the values
        they can take are defined by the dataset authors). Specifically, this
        function: (1) downloads the entire dataset, (2) saves it in /tmp/, (3)
        returns the dataset as a numpy array.

        :param dataset: a dataset from torchvision.datasets
        :type dataset: string
        :param splits: list of dataset "categories" to download
        :type splits: list or NoneType
        :param directory: directory to download the datasets to
        :type directory: string
        :return: numpy versions of the dataset
        :rtype: dictionary; keys are dataset types & values are numpy arrays
        """

        # use keyword arguments since interfaces can differ slightly
        args = {
            "root": directory,
            "download": True,
            "transform": torchvision.transforms.ToTensor(),
        }
        import pdb; pdb.set_trace()
        return {
            category: getattr(torchvision.datasets, dataset)(
                **(args | {split: category})
            )
            for split, category in itertools.product(*splits)
        }

    def tensorflow(self, dataset):
        """"""
        return

    def uci(self, dataset):
        """"""
        return


if __name__ == "__main__":
    """
    Example usage of MachineLearningDataSets via the command-line as:

        $ mlds mnist nslkdd -i 50-700 -i protocol flag --outdir datasets
            -n mnist_mod nslkdd_mod -s normalization -s standardization minmax
            -a --destupefy

    This (1) downloads MNIST and NSL-KDD, (2) selects all features for MNIST
    (necessary to correctly associate the following "-i" with NSL-KDD) and
    "protocol" & "flag" for NSL-KDD, (3) specifies an alternative output
    directory (instead of "out/"), (4) changes the base dataset name when
    saved, (5) applies minmax scaling to MNIST and creates two copies of the
    NSL-KDD that are standardization & normalized, respectively, and (6)
    computes basic analytics and applies destupification (to both datasets).
    """
    raise SystemExit(0)
