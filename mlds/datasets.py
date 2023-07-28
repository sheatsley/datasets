"""
This module defines the core for the machine learning datasets repo. It defines
interfaces for (1) parsing command-line arguments, (2) retrieving datasets, (3)
applying feature and label transformations, (4) and writing data to disk.
"""
import argparse
import pathlib

import dill
import pandas
import sklearn.compose

import mlds
import mlds.transformations as transformations


class Dataset:
    """
    The Dataset class defines the main interfaces for datasets created with this
    repo. It defines methods for accessing data and previewing metadata.

    The Dataset class represents the main interface for working with loaded
    datasets via the load method in this module. Specifically, it: (1) provides
    an intuitive representation of loaded partitions (including the partitions,
    the number of samples, and associated metadata), and (2) defines wrappers
    for casting numpy arrays into popular datatypes used in machine learning
    (e.g., PyTorch Tensors).

    :func:`__init__`: prepares loaded data
    :func:`__repr__`: shows useful dataset statistics
    """

    def __init__(self, datadict, dataset, metadata):
        """
        This method instantiates a Dataset object with attributes to access the
        underlying data and labels, as well as collects metadata attributes to
        be used as a string-based representation.

        :param datadict: the dataset to save
        :type datadict: dict of numpy ndarray objects
        :param dataset: name of the dataset
        :type dataset: str
        :param metadata: metadata to be saved alongside the dataset
        :type metadata: dict of various datatypes
        :return: a dataset
        :rtype: Dataset object
        """
        for partition, data in datadict.items():
            data["partition"] = partition
            data["classes"] = len(set(data["labels"]))
            data["samples"], data["features"] = data["data"].shape
            data["__repr__"] = (
                lambda p: f"{p.partition}(samples={p.samples}, "
                f"features={p.features}, classes={p.classes})"
            )
            setattr(self, partition, type("Partition", (), data)())
        self.dataname = dataset
        self.metadata = metadata
        return None

    def __repr__(self):
        """
        This method returns a string-based representation of useful metadata
        for debugging.

        :return: dataset statistics
        :rtype: str
        """
        partitions = [getattr(self, p) for p in self.metadata["partitions"]]
        info = [(f"{p.samples}", f"{p.features}", f"{p.classes}") for p in partitions]
        samples, features, classes = map(", ".join, zip(*info))
        return (
            f"{self.dataname}(samples=({samples}), features=({features}), "
            f"classes=({classes}), "
            f"partitions=({', '.join(self.metadata['partitions'])}), "
            f"transformations=({', '.join(self.metadata['transformations'])}), "
            f"version={self.metadata['version']})"
        )


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
        choices={getattr(mlds.downloaders, d) for d in mlds.downloaders.__all__},
        help="Dataset to retrieve and process",
        type=lambda d: getattr(mlds.downloaders, d),
    )
    parser.add_argument(
        "--destupefy",
        action="store_true",
        help="clean the dataset automagically (experimental)",
    )
    parser.add_argument(
        "-f",
        "--features",
        action="append",
        default=[],
        help="features-transformation pairs (supports feature names, indices, & 'all')",
        metavar="FEATURE_1,FEATURE_2,...,FEATURE_N DATA_TRANSFORMATION",
        nargs=2,
    )
    parser.add_argument(
        "--filename",
        help="output file name",
        metavar="OUTPUT_FILENAME",
    )
    parser.add_argument(
        "-l",
        "--labels",
        choices=(getattr(transformations, d) for d in ("LabelEncoder",)),
        default=mlds.transformations.IdentityTransformer,
        help="label transformation to apply",
        metavar="LABEL_TRANSFORMATION",
        type=lambda d: getattr(transformations, d),
    )
    parser.add_argument(
        "--version",
        action="version",
        help="Displays package version",
        version=mlds.__version__,
    )
    args = parser.parse_args()

    # default filenames are the dataset concatenated with transformations
    if not args.filename:
        t = [t for _, t in args.features]
        args.filename = f"{args.dataset}_{'_'.join(t)}_{args.labels.__name__}"

    # map transformations to transformations module classes
    valid_transforms = set(
        t
        for t in dir(transformations)
        if isinstance(getattr(transformations, t), type)
        and t not in {"Destupifier", "IdentityTransformer", "LabelEncoder"}
    )
    for i, (f, t) in enumerate(args.features):
        try:
            if t not in valid_transforms:
                raise AttributeError
            args.features[i] = (f, getattr(transformations, t))
        except AttributeError:
            parser.error(f"{t} is not valid! Must be: {', '.join(valid_transforms)}")

    # try to ensure each feature set is associated with only one transformation
    feature_sets = set()
    features, transforms = zip(*args.features)
    features = tuple(tuple(f.split(",")) for f in features)
    for f in features:
        if f in feature_sets:
            parser.error(f"Feature {f} is mapped to more than one transformation!")
        feature_sets.add(f)

    # print parsed arguments and process the dataset
    print("Arguments:", ", ".join(f"{a}={v}" for a, v in vars(args).items()))
    process(
        dataset=args.dataset,
        data_transforms=tuple(transforms),
        destupefy=args.destupefy,
        features=features,
        filename=args.filename,
        label_transform=args.labels,
    )
    raise SystemExit(0)


def process(dataset, data_transforms, destupefy, features, filename, label_transform):
    """
    This function serves as a wrapper for the main interfaces of this repo.
    Specifically, it: (1) retrieves datasets from the downloaders package, (2)
    transforms (and optionally cleans) the data via the transform module, (3)
    casts the pandas dataframes as numpy arrays and (4) saves the data to disk.

    :param dataset: dataset to download
    :type dataset: downloaders module object
    :param data_transforms: transformations to apply to the data
    :type data_transforms: tuple of transformations module classes
    :param destupefy: whether to clean the data (experimental)
    :type destupefy: bool
    :param features: features to transform
    :type features: tuple of tuples of strs
    :param filename: filename of the saved dataset
    :type filename: str
    :param label_transform: transformation to apply to the labels
    :type label_transform: transformations module class
    :return: None
    :rtype: NoneType
    """

    # retrieve the dataset and get feature names (to resolve "all" argument)
    print(f"Retrieving {(dataname := dataset.__name__.split('.').pop())}...")
    datadict = dataset.retrieve()
    partition = next(iter(datadict))
    feature_names = datadict[partition]["data"].columns

    # map "all" keyword to all features except those that are one-hot encoded
    print("Resolving 'all' argument with inferred features...")
    one_hot_features = tuple(
        feature
        for feature_set, transform in zip(features, data_transforms)
        if transform == transformations.OneHotEncoder
        for feature in feature_set
    )
    all_features = feature_names.difference(other=one_hot_features, sort=False)
    features = tuple(tuple(all_features) if f == ("all",) else f for f in features)

    # instantiate transformers and determine if a training set exists
    print(
        "Instantiating Transformers with the following transformations:",
        ", ".join(f"{t.__name__}{f}" for f, t in zip(features, data_transforms)),
    )
    data_transforms = tuple(
        t(sparse_output=False) if t is transformations.OneHotEncoder else t()
        for t in data_transforms
    )
    data_transformers = sklearn.compose.make_column_transformer(
        *zip(data_transforms, features),
        n_jobs=-1,
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    label_transformer = label_transform()
    destupefier = (
        transformations.Destupefier()
        if destupefy
        else transformations.IdentityTransformer()
    )
    partitions = list(datadict)
    if "train" in datadict:
        has_train = True
        partitions.sort(key=lambda p: p != "train")
    else:
        has_train = False

    # fit the transformer to each partitions if a training set doesn't exist
    metadata = {
        "partitions": partitions,
        "transformations": tuple(type(t).__name__ for t in data_transforms),
        "version": mlds.__version__,
    }
    for partition in partitions:
        print(f"Applying transformations to {dataname} {partition} partition...")
        data, labels = datadict[partition].values()
        fit = partition == "train" or not has_train

        # apply transformations and (optionally) clean the data
        if fit:
            data_transformers.fit(data)
            label_transformer.fit(labels)
            destupefier.fit(data)
        x = data_transformers.transform(data)
        y = pandas.Series(label_transformer.transform(labels))
        x, y = destupefier.transform(x, y)

        # correct order of features when using passthrough and one-hot encoding
        one_hot_map = {}
        removed = destupefier.deficient if destupefy else []
        partition_features = feature_names.difference(other=removed, sort=False)
        output_one_hot_features = (
            transformed_features
            for _, t, _ in data_transformers.transformers_
            if type(t) is transformations.OneHotEncoder
            for transformed_features in t.categories_
        )
        for original, new in zip(one_hot_features, output_one_hot_features):
            one_hot_map[original] = list(f"{original}_" + new)
            idx = partition_features.get_loc(original)
            partition_features = partition_features.delete(idx)
            partition_features = partition_features.insert(idx, one_hot_map[original])
        x = x[partition_features]

        # cast dataframes as 32-bit numpy arrays in C-contiguous order
        datadict[partition] = {
            "data": x.to_numpy().astype("float32", order="C"),
            "labels": y.to_numpy().astype("float32", order="C"),
        }

        # set feature names, class name maps, and one-hot-encoding maps as metadata
        metadata[partition] = {"features": list(partition_features)}
        if one_hot_map:
            metadata[partition]["one_hot_map"] = one_hot_map
        if type(label_transformer) is not transformations.IdentityTransformer:
            class_map = {o: n for n, o in enumerate(label_transformer.classes_)}
            metadata[partition]["class_map"] = class_map
        print(f"Transformation complete. Final shape: {x.shape} × {y.shape}")

    # write the dataset to disk
    outdir = pathlib.Path(__file__).parent / "datasets"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{filename}.pkl", "wb") as f:
        dill.dump(Dataset(datadict=datadict, dataset=filename, metadata=metadata), f)
    print(f"{dataname} retrieval, transformation, and export complete!")
    return None
