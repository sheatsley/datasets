"""
This module defines the core for the machine learning datasets repo. It defines
interfaces for (1) parsing command-line arguments, (2) retrieving datasets, (3)
applying feature and label transformations, (4) and writing data to disk.
"""
import argparse
import pathlib
import subprocess

import numpy as np

import mlds
import mlds.adapters as adapters
import mlds.transform as transformations


class Dataset:
    """
    The Dataset class represents the main interface for working with loaded
    datasets via the load method in this module. Specifically, it: (1) provides
    an intuitive representation of loaded partitions (including the partitions,
    the number of samples, and associated metadata), and (2) defines wrappers
    for casting numpy arrays into popular datatypes used in machine learning
    (e.g., PyTorch Tensors).

    :func:`__init__`: prepares loaded data
    :func:`__repr__`: shows useful dataset statistics
    """

    def __init__(self, name, partitions):
        """
        This method instantiates Dataset objects and builds a representation
        from the paired metadata.

        :param name: dataset name (i.e., filename)
        :type name: str
        :param partitions: the loaded data partitions
        :type partitions: list of tuples of form: (partition, namedtuple)
        :return: a dataset
        :rtype: Dataset object
        """
        import collections  # Container datatypes

        # set partitions and data
        for part, data in partitions:
            setattr(
                self,
                part,
                collections.namedtuple(part.capitalize(), ("data", "labels"))(
                    data.data, data.labels
                ),
            )
            for field in (f for f in data._fields if f not in {"data", "labels"}):
                setattr(self, field, getattr(data, field))

        # build representation from name and metadata
        samples, features, classes = zip(
            *[
                (
                    f"{part}={data.data.shape[0]}",
                    data.data.shape[1],
                    len(set(data.labels)),
                )
                for part, data in partitions
            ]
        )
        samples = ", ".join(samples)
        transformations = ", ".join(self.transformations)
        self.name = (
            f"{name}(samples=({samples}), features={features[0]}, "
            f"classes={classes[0]}, transformations=({transformations}))"
        )
        return None

    def __repr__(self):
        """
        This method returns a string-based representation of useful metadata
        for debugging.

        :return: dataset statistics
        :rtype: str
        """
        return self.name


def command_line():
    """
    This function defines the main interface for using this repo via the
    command line. Specifically, it defines the available arguments, verifies
    that the parsed arguments are compatible, and calls process to retrieve,
    transform, and save the dataset.

    :return: command-line arguments
    :rtype: argparse.Namesapce
    """
    parser = argparse.ArgumentParser(
        description="Retrieves and processes popular machine learning datasets",
        prog="mlds",
    )
    parser.add_argument(
        "dataset",
        choices=mlds.__available__,
        help="Dataset to retrieve and process",
    )
    parser.add_argument(
        "-d",
        "--data",
        action="append",
        choices=(
            getattr(transformations.Transformer, d)
            for d in (
                "minmaxscaler",
                "onehotencoder",
                "robustscaler",
                "standardscaler",
                "uniformscaler",
            )
        ),
        default=[],
        help="data transformation(s)to apply",
        nargs="*",
        type=lambda d: getattr(transformations.Transformer, d),
    )
    parser.add_argument(
        "-f",
        "--features",
        action="append",
        default=[],
        help="transformable features (or indicies) (can use 'all')",
        metavar="FEATURE",
        nargs="*",
    )
    parser.add_argument(
        "-l",
        "--labels",
        default=[],
        choices=(getattr(transformations.Transformer, d) for d in ("labelencoder",)),
        help="label transformations(s) to apply",
        nargs="*",
        type=lambda d: getattr(transformations.Transformer, d),
    )
    parser.add_argument(
        "--filename",
        help="output file name(s)",
        metavar="DATASET_NAME",
        nargs="*",
    )
    parser.add_argument(
        "--destupefy",
        action="store_true",
        help="cleanup datasets automagically (experimental)",
    )
    parser.add_argument(
        "--version",
        action="version",
        help="Displays module version",
        version="4.0",
    )
    args = parser.parse_args()

    # set default filenames to the product of data and label transformations
    if not args.filename:
        args.filename = (
            f"{args.dataset}_{dt}_{lt}" for dt in args.data for lt in args.labels
        )

    # validate that every data transformation has a corresponding feature set
    if (datalen := len(args.data)) != (featurelen := len(args.features)):
        parser.error(
            "Each data transformation must have a corresponding feature set!"
            f" (Parsed {datalen} data transformations and {featurelen} feature sets)"
        )

    # validate that every dataset has a corresponding filename
    if (setlen := datalen * len(args.labels)) != (filenamelen := len(args.filename)):
        parser.error(
            "Each output dataset must have a corresponding filename!"
            f" (Parsed {setlen} output datasets and {filenamelen} filenames)"
        )

    # print parsed arguments and process the dataset
    print("Arguments:", ", ".join(f"{a}={v}" for a, v in vars(args).items()))
    process(
        dataset=args.dataset,
        data_transforms=args.data,
        destupefy=args.destupefy,
        features=args.features,
        filenames=args.filename,
        labels=args.labels,
    )
    raise SystemExit(0)


def process(dataset, data_transforms, destupefy, features, filenames, label_transforms):
    """
    This function serves as a wrapper for the main interfaces of this repo.
    Specifically, this function: (1) retrieves datasets from the adapter
    package, (2) transforms (and optionally cleans) the data via the transform
    module, and (3) saves the data to disk.

    :param dataset: dataset to download
    :type dataset: str
    :param data_transforms: transformations to apply to the data
    :type data_transforms: tuple of tuples of Transformer callables
    :param destupefy: whether to clean the data (experimental)
    :type destupefy: bool
    :param features: features to manipulate
    :type features: tuple of tuples of strs
    :param filenames: filenames of the saved datasets
    :type filenames: tuple of strs
    :param labels_transforms: transformations to apply to the labels
    :type label_transforms: tuple of tuples of Transformer callables
    :return: None
    :rtype: NoneType
    """

    # retrieve the dataset and get feature names (to resolve "all" argument)
    print(f"Retrieving {dataset}...")
    datadict = getattr(adapters, dataset).retrieve()
    partition = next(iter(datadict))
    feature_names = datadict[partition]["data"].columns
    print(f"Inferred {len(feature_names)} features from {partition} partition.")

    # map "all" keyword to all features except those that are one-hot encoded
    print("Resolving 'all' argument with inferred features...")
    all_features = (
        tuple(feature_names.difference(feature_tuple, sort=False))
        for transform, feature_tuple in zip(data_transforms, features)
        if transformations.Transformer.onehotencoder in transform
    )
    features = (af if of == ("all",) else of for af, of in zip(all_features, features))

    # ensure transformations are only fit to the training set (if applicable)
    print("Instantiating Transformer & applying transformations...")
    transformer = transformations.Transformer(
        data_transforms=data_transforms,
        features=features,
        label_transforms=label_transforms,
    )
    partitions = list(datadict)
    if "train" in datadict:
        has_train = True
        partitions.sorted(key=lambda p: p != "train")
    else:
        has_train = False

    # fit the transformer to the first partition (e.g., training)
    for partition in partitions:
        print(f"Applying transformations to {dataset} {partition} partition...")
        data, labels = datadict[partition].values()
        fit = partition == "train" or not has_train
        transformer.apply(data=data, fit=fit, label=labels)

        # export the dataset, optionally destupefy, and compute metadata
        for (data_t, label_t, x, y), filename in zip(transformer.export(), filenames):
            x, y = transformer.destupefy(fit=fit, x=x, y=y) if destupefy else (x, y)
            metadata = transformer.metadata() | {
                "data transformations": data_t,
                "label transformations": label_t,
            }

            # write the dataset to disk
            save(filename=f"{filename}-{partition}", metadata=metadata, x=x, y=y)
    print(f"{dataset} retrieval, transformation, and export complete!")
    return None


def save(
    dataframe,
    labelframe,
    name,
    metadata={},
    precision=np.float32,
    analytics=False,
    outdir=pathlib.Path(__file__).parent / "out/",
):
    """
    This function is the main exit point from MLDS. It consumes a dataset and
    labels, casts them into numpy arrays (with precision no greater than
    specified), instantiates & populates a dataset object, and writes a
    namedtuple object containing data & metadata to disk with the specified
    filename and output directory. Optionally, analytics are computed and saved
    in the same directory.

    :param dataframe: dataset
    :type dataframe: pandas dataframe
    :param labelframe: labels
    :type labelframe: pandas series
    :param name: filename of dataset (and partition)
    :type name: str
    :param metadata: metadata to be saved alongside the dataset
    :type metadata: dict of various datatypes
    :param precision: maximum dataset precision
    :type precision: numpy type
    :param analytics: whether to compute and save dataset analytics
    :type analytics: bool
    :param outdir: output directory
    :type outdir: pathlib path
    :return: None
    :rtype: NoneType
    """

    # convert to numpy arrays & check if numerical casting is possible (ie not strings)
    print(f"Assembling {name} and casting to (max) {precision.__name__} arrays...")
    data = dataframe.to_numpy()
    labels = labelframe.to_numpy()
    data_precision = (
        precision
        if issubclass(data.dtype.type, np.floating)
        and np.finfo(precision).precision < np.finfo(data.dtype).precision
        else data.dtype
    )
    label_precision = next(
        precision
        for precision in (np.int8, np.int16, np.int32, np.int64, labels.dtype)
        if all(np.can_cast(label, precision) for label in np.unique(labels))
    )

    # set maximum precision (label precision is set based on the number of classes)
    print(
        f"Casting data to {data_precision.__name__}",
        f"and labels to {label_precision.__name__}...",
    )
    data = data.astype(data_precision, order="C")
    labels = labels.astype(label_precision, order="C")

    # populate a dataset object
    print("Populating dataset object...")
    version = subprocess.check_output(
        ("git", "-C", __file__.rstrip("datasets.py"), "rev-parse", "--short", "HEAD"),
        text=True,
    ).strip()
    dataset = utilities.assemble(data, labels, metadata | {"version": version})

    # save the results to disk
    print(f"Pickling dataset & writing {name}.pkl to {outdir}/...")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{name}.pkl", "wb") as f:
        dill.dump(dataset, f)
    print(f"{name}.pkl saved to {outdir}/!")

    # compute analyitcs if desired
    utilities.analyze(
        dataframe,
        labelframe.replace(metadata.get("class_map")),
        name,
        outdir,
    ) if analytics else None
    return None


def assemble(x, y, metadata={}):
    """
    This function populates namedtuples with data & labels, as well as any
    desired metadata.

    :param x: data samples
    :type x: numpy array
    :param y: labels
    :type y: numpy array
    :param metadata: metadata to be stored
    :type metadata: dictionary
    :return: complete dataset with metadata
    :rtype: namedtuple object
    """
    return collections.namedtuple("Dataset", ["data", "labels", *metadata])(
        x, y, **metadata
    )
