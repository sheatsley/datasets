# Datasets

This repo contains scripts for downloading, preprocessing, and numpy-ifying
popular machine learning datasets. Across projects, I commonly found myself
rewriting the same lines of code to standardize, normalize, or other-ize data,
encode categorical variables, parse out subsets of features, among other
miscellanea. To alleviate reinventing the wheel, this repo consumes a
template-style definition for how a dataset should be parsed and the library
takes care of the rest. 

For loading data, it supports anything `numpy.genfromtxt` can consume and
`arff`. For manipulating data, there are wrappers for many popular
`scikit-learn` `preprocessing` transformers in `utils/scale.py` and
`utils/preprocess.py`.

The main magic is found in `utils/handler.py`; at the bottom, example templates
haven been provided. The arguments are:

Argument  |Description
----------|-----------
`header`  | whether or not a header row exists (which will be removed)
`include` | only load the specified columns
`label`   | indicies for labels
`norm`    | range of features to be norm'd together for `unit_norm` scaler
`onehot`  | list of categorical features to convert to one-hot vectors
`path`    | path to the dataset to load
`preserve`| do not modify these indicies for any scaling scheme (ie labels)
`scheme`  | feature scaling schema (definitions in scale module)
`size`    | the number of training samples (only needed if test is true)
`test`    | whether or not a test set exists (inferred to be 2nd path)


## Installation

To convert the available datasets (ie,
[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html),
[UNSW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/),
[Phishing Websites
Dataset](http://www.fcsit.unimas.my/research/legit-phish-set),
[DREBIN](https://www.sec.cs.tu-bs.de/~danarp/drebin/)), 
[DGD](https://ieeexplore.ieee.org/document/9343331), clone the repo, and run
`python3 utils/handler.py`. While this code has been successful at handling a
handful of unique datasets I have used, this is obviously research-level code;
I am sure some instances will break. 

For CiFar10, download the python version [here](https://www.cs.toronto.edu/~kriz/cifar.html), unzip the file, and move the folder to this directory. 
