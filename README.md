# Machine Learning Datasets

This repo contains scripts for downloading, preprocessing, and numpy-ifying
popular machine learning datasets. Across projects, I commonly found myself
rewriting the same lines of code to standardize, normalize, or other-ize data,
encode categorical variables, parse out subsets of features, among other
miscellanea. To alleviate reinventing the wheel, this repo retrieves,
parses, and transforms datasets as desired via flexible command-line arguments.

## Datasets

This repo can retrieve datasets from a variety of online sources, as well as
through popular machine learning frameworks, such as PyTorch and Tensorflow.
While most datasets from
[torchvision](https://pytorch.org/vision/stable/datasets.html) or
[tensorflow_datasets](https://www.tensorflow.org/datasets) should be readily
retrievable, not every dataset has been tested. The following datasets are
currently available (and tested):

* [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
* [UNSW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
* [Phishing Dataset](https://www.fcsit.unimas.my/phishing-dataset)
* [CIC-MalMem-2022](https://www.unb.ca/cic/datasets/malmem-2022.html)
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

## Arguments

The following arguments are supported:

Argument   | Description
-----------|-----------
`dataset`  | dataset to retrieve (`adapters/` contains other datasets)
`features` | features to manipulate (either index or name, if supported)
`schemes`  | feature manipulation scheme(s) to apply (check `transform.py`)
`labels`   | label manipulation scheme(s) to apply (check `transform.py`)
`names`    | filenames for transformed datasets
`outdir`   | specified output directory (default is `out/`)
`precision`| maximum dataset precision
`analytics`| compute basic dataset statistics (check `utilities.py`)
`destupefy`| apply heuristics to clean the dataset (check `transform.py`)
`template` | apply commonly-accepted transformations (check `templates.py`)
`version`  | show the current version number

## Example usage

Below are some commands you may find useful for running this repo:

    python3 mlds.py mnist -t

This downloads MNIST and applies the transformations found in `templates.py`;
features are normalized between 0-1 and the dataset is saved as `mnist` to
`out/`.

    python3 mlds.py fashionmnist -f all -s minmaxscaler standardscaler
    -n fmnist_mms fmnist_ssc --outdir transformations

This downloads fashion-MNIST and creates two copies of the dataset: one where
all features are normalized between 0-1 and another where features are
standardized. The transformed datasets are named `fmnist_mms` & `fmnist_ssc`,
respectively, and both are saved in a folder called `transformations`.

    python3 mlds.py nslkdd -f all -f protocol_type service flag -s minmaxscaler
    -s onehotencoder -l labelencoder

This downloads the NSL-KDD and creates one copy of the dataset:
`protocol_type`, `service`, and `flag` features are one-hot encoded, while the
remaining features (`all` applies transformations to all features, except those
that are one-hot encoded) are normalized between 0-1. The dataset is saved as
`nslkdd_minmaxscaler_onehotencoder_labelencoder` and written to `out/`.

Once datasets have been transformed, they can be readily loaded via (for
example):

    import mlds.nslkdd_minmaxscaler_onehotencoder_labelencoder as nslkdd

From there, training data can be retrieved as `nslkdd.train.data` and the
corresponding labels can be retrieved as `nslkdd.train.labels`.

## Dependencies

This repo was built with the following modules and versions:

* python 3.9.12
* [dill](https://github.com/uqfoundation/dill) 0.3.4
* [matplotlib](https://matplotlib.org) 3.5.1
* [numpy](https://numpy.org) 1.22.3
* [pandas](https://pandas.pydata.org) 1.4.1
* [requests](https://docs.python-requests.org/en/latest/) 2.27.1
* [scikit-learn](https://scikit-learn.org/stable/) 1.0.2
* [tensorflow-datasets](https://www.tensorflow.org/datasets) 4.5.2
* [torchvision](https://pytorch.org/vision/stable/index.html) 0.12.0
