"""
This module defines the core for the machine learning datasets repo. It defines
interfaces for (1) parsing command-line arguments, (2) retrieving datasets, (3)
applying feature and label transformations, (4) and writing data to disk.
"""
import argparse
import pathlib
import pickle

import sklearn.compose

import mlds
import mlds.adapters as adapters
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
        samples = []
        for partition, data in datadict.items():
            setattr(self, partition, type("Partition", (), data))
            self.classes = len(set(data["labels"]))
            partition_samples, self.features = data["data"].shape
            samples.append(partition_samples)
        self.dataset = dataset
        self.metadata = metadata
        self.samples = ", ".join(samples)
        return None

    def __repr__(self):
        """
        This method returns a string-based representation of useful metadata
        for debugging.

        :return: dataset statistics
        :rtype: str
        """
        return (
            f"{self.dataset}(samples=({self.samples}), features={self.features}, "
            f"classes={self.classes}, transformations=({transformations}))"
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
        choices=mlds.__available__,
        help="Dataset to retrieve and process",
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
        metavar=("FEATURE_1,FEATURE_2,...,FEATURE_N", "TRANSFORMATION"),
        nargs=2,
        type=lambda d: getattr(transformations, d),
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
        help="label transformation to apply",
        metavar="LABEL_TRANSFORMATION",
        type=lambda d: getattr(transformations, d),
    )
    parser.add_argument(
        "--version",
        action="version",
        help="Displays module version",
        version="4.0",
    )
    args = parser.parse_args()

    # default filenames are the dataset concatenated with transformations
    if not args.filename:
        t = [t for _, t in args.features]
        args.filename = f"{args.dataset}_{'_'.join(t)}_{args.labels.__name__}"

    # map transformations to transformations module callables
    for i, (f, t) in enumerate(args.features):
        try:
            args.features[i] = (f, getattr(transformations, t))
        except AttributeError:
            parser.error(
                f"{t} is not a valid transformation! Supported transformations are:"
                "MinMaxScaler, ",
                "OneHotEncoder, ",
                "RobustScaler, ",
                "StandardScaler, ",
                "UniformScaler",
            )

    # ensure each feature set is associated with only one transformation
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
        features=args.features,
        filename=args.filename,
        label_transform=args.labels,
    )
    raise SystemExit(0)


def process(dataset, data_transforms, destupefy, features, filename, label_transform):
    """
    This function serves as a wrapper for the main interfaces of this repo.
    Specifically, it: (1) retrieves datasets from the adapter package, (2)
    transforms (and optionally cleans) the data via the transform module, (3)
    casts the pandas dataframes as numpy arrays and (4) saves the data to disk.

    :param dataset: dataset to download
    :type dataset: str
    :param data_transforms: transformations to apply to the data
    :type data_transforms: tuple of transformations module callables
    :param destupefy: whether to clean the data (experimental)
    :type destupefy: bool
    :param features: features to transform
    :type features: tuple of tuples of strs
    :param filename: filename of the saved dataset
    :type filename: str
    :param label_transform: transformation to apply to the labels
    :type label_transform: transformations module callable
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
    all_features = feature_names.difference(
        feature
        for feature, transform in zip(features, data_transforms)
        if transform == transformations.Transformer.onehotencoder
    )
    features = tuple(tuple(all_features) if f == ("all",) else f for f in features)

    # instantiate transformers and determine if a training set exists
    print("Instantiating Transformers...")
    data_transforms = tuple(
        t(sparse_output=False) if t is transformations.OneHotEncoder else t()
        for t in data_transforms
    )
    data_transformers = sklearn.compose.make_column_transformer(
        *zip(data_transforms, features),
        n_jobs=-1,
        remainder="passthrough",
        verbose_features_out=False,
    ).set_output(tranform="pandas")
    label_transformer = label_transform() if label_transform else None
    destupefier = transformations.Destupefier() if destupefy else None
    partitions = list(datadict)
    if "train" in datadict:
        has_train = True
        partitions.sorted(key=lambda p: p != "train")
    else:
        has_train = False

    # fit the transformer to each partitions if a training set doesn't exist
    metadata = {}
    for partition in partitions:
        print(f"Applying transformations to {dataset} {partition} partition...")
        data, labels = datadict[partition].values()
        fit = partition == "train" or not has_train

        # apply transformations and (optionally) clean the data
        if fit:
            data_transformers.fit(data)
            label_transformer.fit(labels) if label_transform else None
            destupefier.fit(data) if destupefy else None
        x = data_transformers.transform(data)
        y = label_transformer.transform(labels) if label_transform else labels
        x, y = destupefier.transform(x, y) if destupefy else (x, y)

        # correct feature order and cast to numpy arrays
        partition_features = filter(lambda f: f in destupefier.deficient, features)
        for _, t, f in data_transformers.transformers_:
            if type(t) is transformations.OneHotEncoder:
                idx = features.index(f)
                features[idx : idx + 1] = [f for f in t.categories_]
        x = x[partition_features]
        datadict[partition] = {"data": x.to_numpy(), "labels": y.to_numpy()}

        # set feature naems, class name maps, and one-hot-encoding maps as metadata
        metadata[partition] = {"feature_names": partition_features}
        for _, t, f in data_transformers.transformers_:
            if type(t) is transformations.OneHotEncoder:
                onehot_map = metadata[partition].setdefault("onehot_map", {})
                onehot_map[f] = t.categories_
        if label_transformer:
            classes = label_transformer.classes_
            metadata[partition]["class_map"] = {n: o for n, o in enumerate(classes)}
        print(f"Transformation complete. Final shape: {x.shape} × {y.shape}")

    # write the dataset to disk
    outdir = pathlib.Path(__file__).parent / "datasets" / f"{filename}.pkl", "wb"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{filename}.pkl", "wb") as f:
        pickle.dump(Dataset(datadict=datadict, dataset=filename, metadata=metadata), f)
    print(f"{dataset} retrieval, transformation, and export complete!")
    return None
