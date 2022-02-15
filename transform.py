"""
The transform module applies transformations to machine learning datasets.
Author: Ryan Sheatsley
Tue Feb 15 2022
"""
from utilities import print  # Timestamped printing

# TODO
# add print statements
# destupify should cleanse for unknown values
# create FunctionTransformer to remove NaNs, NULLs, etc.


class Transformer:
    """
    This transformer class servs as an intelligent wrapper for scikit-learn's
    data transformation functions. Notably, the transformers (and
    ColumnTransformer compositions) do not implicitly preserve the order of
    features post-transformation. This class enables arbitrary data
    transformation, while conforming to the standard layout the data was
    originally presented in.

    :func:`__init__`: instantiates Transformer objects
    :func:`transform`: applies transformation schemes to the data
    """

    def __init__(self, schemes, features):
        """
        This function initializes Transformer objects with the necessary
        information to apply arbitrary transformations to data.

        :param datasets: datasets to apply transformations to
        :type datasets: dictionary pointing to pandas dataframes
        :param schemes: transformations to apply to the data
        :type schemes: list of strings
        :param features: features to apply the transformations to
        :type features: list of integers (indicies) or strings (names)
        """
        return None

    def download(self, datasets):
        """
        This function dispatches dataset downloads to the respective handlers.

        :param datasets: datasets to download
        :type datasets: list of strings
        :return: the downloaded datasets
        :rtype: dictionary of datasets containing dictionaries of categories
        """
        downloads = {}
        for dataset in datasets:
            if dataset in self.pytorch_datasets:
                downloads[dataset] = self.pytorch(
                    self.pytorch_datasets[dataset]["name"],
                    *self.pytorch_datasets[dataset]["split"]
                )
            elif dataset in self.tensorflow_datasets:
                downloads[dataset] = self.tensorflow(dataset)
            elif dataset in self.custom_datasets:
                pass
            else:
                raise KeyError(dataset, "not supported")
        return downloads

    def pytorch(self, dataset, arg, splits, directory="/tmp/"):
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
        :param arg: the name of the argument governing the dataset splits
        :type arg: string
        :param splits: list of dataset "categories" to download
        :type splits: list or NoneType
        :param directory: directory to download the datasets to
        :type directory: string
        :return: numpy versions of the dataset
        :rtype: dictionary; keys are dataset types & values are numpy arrays
        """
        return {
            # map splits to strings so that they are human-readable
            self.pytorch_map.get(split, split): getattr(torchvision.datasets, dataset)(
                # use keyword arguments since interfaces can differ slightly
                **{
                    "root": directory,
                    "download": True,
                    "transform": torchvision.transforms.ToTensor(),
                    arg: split,
                }
            )
            for split in splits
        }

    def tensorflow(self, dataset, directory="/tmp/"):
        """
        This function serves as a wrapper for tensorflow_datasets.
        (https://www.tensorflow.org/datasets). The interfaces for the vast
        majority of datasets are identical, and thus, this wrapper largely
        prepares the data such that it conforms to the standard used throughout
        the rest of this repository.

        :param dataset: a dataset from tensorflow_datasets
        :type dataset: string
        :param directory: directory to download the datasets to
        :type directory: string
        :return: numpy versions of the dataset
        :rtype: dictionary; keys are dataset types & values are numpy arrays
        """
        return tensorflow_datasets.as_numpy(
            tensorflow_datasets.load(dataset, data_dir=directory, batch_size=-1)
        )


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
