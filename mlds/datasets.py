"""
This module defines the core for the machine learning datasets repo. It defines
interfaces for (1) parsing command-line arguments, (2) retrieving datasets, (3)
applying feature and label transformations, (4) and writing data to disk.
"""
import pathlib
import subprocess

import numpy as np

import mlds.adapters as adapters
import mlds.transform as transform
import mlds.utilities as utilities


def process(
    dataset,
    destupefy,
    features,
    labels,
    names,
    outdir,
    schemes,
):
    """
    This function serves as a wrapper for the main interfaces of this repo.
    Specifically, this function: (1) retrieves datasets from the adapter
    package, (2) transforms (and optionally cleans) the data via the transform
    module, and (3) saves the data to disk.

    :param dataset: dataset to download
    :type dataset: str
    :param destupefy: whether to clean the data (experimental)
    :type destupefy: bool
    :param features: features to manipulate
    :type features: tuple of tuples of strs
    :param labels: transformations to apply to the labels
    :type labels: tuple of tuples of Transformer callables
    :param names: filenames of the saved datasets
    :type names: tuple of strs
    :param schemes: transformations to apply to the data
    :type schemes: tuple of tuples of Transformer callables
    :return: None
    :rtype: NoneType
    """

    # retrieve the dataset and get feature names (to resolve "all" argument)
    print(f"Retrieving {dataset}...")
    data = getattr(adapters, dataset).retrieve()
    partition = next(iter(data))
    feature_names = data[partition]["data"].columns
    print(f"Inferred {len(feature_names)} features from {partition} partition.")

    # map "all" keyword to all features except those that are one-hot encoded
    print("Resolving 'all' argument with inferred features...")
    ohot_features = [
        feature
        for scheme, feature_list in zip(schemes, features)
        if transform.Transformer.onehotencoder in scheme
        for feature in feature_list
    ]
    all_feat = feat_names.difference(ohot_features, sort=False).tolist()
    features = [all_feat if f == ["all"] else f for f in features]

    # ensure that training preceeds testing to ensure correct transformation fits
    print("Instantiating Transformer & applying transformations...")
    parts = (
        (["train", "test"] + [p for p in data if p not in {"train", "test"}])
        if all(p in data for p in {"train", "test"})
        else list(data)
    )
    transformer = transform.Transformer(features, labels, schemes)

    # apply transformations to each parittion
    for part in parts:
        print(f"Applying transformations to {dataset} {part} partition...")
        transformer.apply(*data[part].values(), part != "test")

        # assemble the transformations (and restore feature names)
        for (transformed_data, transformed_labels, transformations), name in zip(
            transformer.export(), names
        ):
            # if applicable, destupefy
            transformed_data, transformed_labels = (
                transformer.destupefy(
                    transformed_data, transformed_labels, part != "test"
                )
                if destupefy
                else (transformed_data, transformed_labels)
            )

            # read any relevant metadata
            metadata = {
                **transformer.metadata(),
                **{"orgfshape": orgfshape},
                **{"transformations": transformations},
            }

            # save (with analytics, if desired)
            save(
                transformed_data,
                transformed_labels,
                f"{name}-{part}",
                metadata,
                precision=precision,
                analytics=analytics,
                outdir=outdir,
            )
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
