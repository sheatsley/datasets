# Machine Learning Datasets

_Machine Learning Datasets (mlds)_ is a repo for downloading, preprocessing,
and numpy-ifying popular machine learning datasets. Across projects, I commonly
found myself rewriting the same lines of code to standardize, normalize, or
other-ize data, encode categorical variables, parse out subsets of features,
among other transformations. This repo provides a simple interface to download,
parse, transform, and clean datasets via flexible command-line arguments. All
of the information you need to start using this repo is contained within this
one ReadMe, ordered by complexity (No need to parse through any ReadTheDocs
documentation).

## Table of Contents

* [Datasets](#datasets)
* [Quick Start](#quick-start)
* [Advanced Usage](#advanced-usage)
* [Repo Overview](#repo-overview)

## Datasets

Integrating a data source into `mlds` requires writing a simple module within
the `mlds.downloaders` package to download the dataset (by implementing a
`retrieve` function). Afterwards, add the module to the [package
initializer](https://github.com/sheatsley/datasets/blob/master/mlds/downloaders/__init__.py)
and the dataset is ready to be processed by `mlds`. Notably, the only
requirement is that `retrieve` should return a `dict` containing dataset
partitions as keys and [pandas DataFrames](https://pandas.pydata.org) as
values. At this time, the following datasets are supported:

* [CIC-MalMem-2022](https://www.unb.ca/cic/datasets/malmem-2022.html)
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (via [TensorFlow Datasets](https://www.tensorflow.org/datasets))
* [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (via [TensorFlow Datasets](https://www.tensorflow.org/datasets))
* [MNIST](http://yann.lecun.com/exdb/mnist/) (via [TensorFlow Datasets](https://www.tensorflow.org/datasets))
* [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
* [Phishing](https://www.fcsit.unimas.my/phishing-dataset)
* [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

## Quick Start

This repo was designed to be interoperable with the following
[models](https://github.com/sheatsley/models#repo-overview) and
[attacks](https://github.com/sheatsley/attacks) repos. I recommend installing
an editable version of this repo via `pip install -e`. Afterwards, you can
download, rescale, and save [MNIST](http://yann.lecun.com/exdb/mnist/) so that
features are in [0, 1] (via a custom `UniformScaler` transformer, which scales
all features based on the maximum and minimum feature values observed across
the training set) with the dataset referenced as `mnist` via:

    mlds mnist -f all UniformScaler --filename mnist

You can download, rescale, and save the
[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) so that categorical
features are one-hot encoded (via
[scikit-learn](https://scikit-learn.org/stable/index.html)
[OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html))
while the remaining features (`all` can be used as an alias to select all
features except those that are to be one-hot encoded) are individually rescaled to
be in [0, 1] (via [scikit-learn](https://scikit-learn.org/stable/index.html)
[MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)),
with numeric label encoding (via
[scikit-learn](https://scikit-learn.org/stable/index.html)
[LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)),
with the dataset referenced as
`nslkdd_OneHotEncoder_MinMaxScaler_Label_Encoder` (when `filename` is not
specified, the file name defaults to the name of the dataset concatenated with
the transformations) via:

    mlds nslkdd -f protocol_type,service,flag OneHotEncoder -f all MinMaxScaler -l LabelEncoder 

Afterwards, import the dataset filename to load it: 

    >>> import mlds
    >>> from mlds import mnist
    >>> mnist
    mnist(samples=(60000, 10000), features=(784, 784), classes=(10, 10), partitions=(train, test), transformations=(UniformScaler), version=c88b3d6)
    >>> mnist.train
    train(samples=60000, features=784, classes=10)
    >>> mnist.train.data
    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)
    >>> mnist.train.labels
    array([4., 1., 0., ..., 6., 1., 5.], dtype=float32)

Other uses can be found in the
[examples](https://github.com/sheatsley/datasets/tree/master/examples)
directory.

## Advanced Usage

Below are descriptions of some of the more subtle controls within this repo and
complex use cases.

* API: Should you use to interface with `mlds` outside of the command line, the
    main entry point for the repo is the
    [process](https://github.com/sheatsley/datasets/blob/e11f59fa498fb0606a7231f6588607932b8c7d9b/mlds/datasets.py#L179)
    function within the `datasets` module. The docstring contains all of the
    necessary details surrounding arguments and their required types. An
    example of interfacing with `mlds` in this way can be found in the
    [dnn_prepare](https://github.com/sheatsley/datasets/blob/master/examples/dnn_prepare.py)
    example script.

* Cleaning: When a dataset is to be processed from the command line, it can be
    optionally cleaned by passing `--destupefy` as an argument. This applies a
    transformation, after all other transformations, via the
    [Destupefier](https://github.com/sheatsley/datasets/blob/e11f59fa498fb0606a7231f6588607932b8c7d9b/mlds/transformations.py#L16).
    The `Destupifier` is an experimental transformer that removes duplicate
    columns, duplicate rows, and single-valued columns. This is particularly
    useful when exploring data reduction techniques that may produce a large
    amount of irrelevant features or duplicate samples.

* Downloading: If you wish to add a dataset that is not easily retrievable from
    another library (i.e, [TensorFlow
    Datasets](https://www.tensorflow.org/datasets), [Torchvision
    Datasets](https://pytorch.org/vision/main/datasets.html),
    [pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html),
    etc.), then you may find the
    [download](https://github.com/sheatsley/datasets/blob/e11f59fa498fb0606a7231f6588607932b8c7d9b/mlds/downloaders/__init__.py#L26)
    function within `mlds.downloaders` helpful. Specifically, given a tuple of
    URLs, `mlds.downloaders.download` leverages the
    [Requests](https://requests.readthedocs.io/en/latest/) library to return a
    dictionary containing the requested resource as a bytes object.

* Partitions: Following best practices, transformations are fit to the training
    set and transformed to all other partitions. A partition is considered to
    be a training set if its key is "train" in the data dictionary returned by
    dataset downloaders (This is commonly the case when downloading datasets
    through other frameworks such as [Torchvision
    Datasets](https://pytorch.org/vision/main/datasets.html) and [TensorFlow
    Datasets](https://www.tensorflow.org/datasets)). If the data dictionary
    does not contain this key, then all partitions are fitted and transformed
    separately.

* Transformations: Currently, the following data transformations are supported: 
    * 0-1 scaling (per feature): [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
    * One-hot encoding: [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
    * Outlier-aware scaling: [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
    * Remove mean and scale to unit variance: [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
    * 0-1 scaling (across all features): [UniformScaler](https://github.com/sheatsley/datasets/blob/e11f59fa498fb0606a7231f6588607932b8c7d9b/mlds/transformations.py#L110)

    Adding custom transformations requires inheriting
    `sklearn.base.TransformerMixin` and implementing `fit` and `transform`
    methods. Alternatively, transformations from other libraries can be simply
    aliased into the `transformations` module.

## Repo Overview

This repo was designed to allow easily download and manipulate data from a
variety of different sources (which may have many deficiencies) with a suite of
transformations. The vast majority of the legwork is done within the
`mlds.datasets.process` method; once datasets are downloaded via `downloader`,
then data transformations are composed and applied via
[ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html).
As described above, if "train" is a key in the dictionary representing the
dataset, then the transformers are fit only to this partition.

Notably, the order of features is preserved, post-transformation (including
expanding categorical features with one-hot vectors), and a set of metadata is
saved with the transformed dataset, including: the names of the partitions, the
transformations that were applied, the current abbreviated commit hash of the
repo, the number of samples, the number of features, the number of classes,
one-hot mappings (if applicable), and class mappings (if a label encoding
scheme was applied). Metadata that pertains to the entire dataset (such as the
transformations that were applied) is saved as a dictionary attribute for
`Dataset` objects, while partition-specific information (such as the number of
samples) is saved as an attribute for `Partition` objects (which are set as
attributes in `Dataset` objects).
