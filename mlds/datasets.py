"""
This module defines the main logic for the machine learning datasets repo. It
consists of (1) parsing arguments, (2) retrieving datasets, (3) feature scaling
applications, (4) one-hot, label, & integer encoding, and (5) writing the
resultant arrays to disk.
Author: Ryan Sheatsley
Mon Feb 28 2022
"""
import dill  # serialize all of python
import numpy as np  # The fundamental package for scientific computing with Python
import pathlib  # Object-oriented filesystem paths
import mlds.retrieve as retrieve  # Download machine learning datasets
import mlds.transform as transform  # Order-preserving data transformations
import mlds.utilities as utilities  # Miscellaneous helper functions
from mlds.utilities import print  # Timestamped printing
import subprocess  # Subprocess management


def main(
    analytics,
    dataset,
    destupefy,
    features,
    labels,
    names,
    outdir,
    precision,
    schemes,
):
    """
    This function represents the heart of Machine Learning Datasets (MLDS).
    Given a dataset, a list of features, a list of label transformations, a
    list of output names, an output directory, numerical precision, a list of
    data transformation schemes, and whether to produce resultant analyitcs and
    apply (experimental) destupefication subroutines, MLDS will perform the
    following steps:

        (1) Retrieve the dataset from either torchvision, tensorflow_datasets,
            or a custom dataset (found in the adapaters directory).
        (2) Apply transformations to specific features and reassmble the
            dataset is in the original order.
        (3) Optionally clean the data and produce basic statistics.
        (4) Save the dataset in the specified output directory with
            the specified name in the specified precision.

    Practically speaking, this function mediates interactions between
    Downloader and Transformer objects, while saving data so that it can be
    readily retrieved by the load function in this module.

    :param analytics: whether analytics are computed and saved
    :type analytics: bool
    :param dataset: dataset to download
    :type dataset: str
    :param destupefy: whether data cleaning is performed (experimental)
    :type destupefy: bool
    :param features: features to manipulate
    :type features: tuple of tuples containing str
    :param labels: transformations to apply to the labels
    :type labels: tuple of tuples of Transformer callables
    :param names: filenames of the saved datasets
    :type names: tuple of str
    :param outdir: ouput directory of saved datasets
    :type outdir: pathlib path
    :param precision: dataset precision
    :type precision: numpy type
    :param schemes: transformations to apply to the data
    :type schemes: tuple of tuples of Transformer callables
    :return: None
    :rtype: NoneType
    """

    # instantiate Downloader and download the dataset
    print(f"Instantiating Downloader & downloading {dataset}...")
    downloader = retrieve.Downloader(dataset)
    data = downloader.download()

    # get features (needed for "all" keyword) from first data partition
    part = next(iter(data))
    feat_names = data[part]["data"].columns
    orgfshape = data.get("fshape", (len(feat_names),))
    data.pop("fshape", None)
    print(f"Inferred {len(feat_names)} features from {part} partition.")

    # resovle "all" keyword to feature names minus those used in one-hot encoding
    print("Resolving 'all' keyword with inferred features...")
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
