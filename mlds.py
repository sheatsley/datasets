"""
This module defines the main entry point for the machine learning datasets
repo. It consists of (1) parsing arguments, (2) retrieving datasets, (3)
feature scaling applications, (4) one-hot, label, & integer encoding, and (5)
writing the resultant arrays to disk.
Author: Ryan Sheatsley
Mon Feb 28 2022
"""
import retrieve  # Download machine learning datasets
import save  # Save (and load) machine learning datasets quickly
import transform  # Apply transformations to machine learning datasets
from utilities import print  # Timestamped printing


def __getattr__(dataset):
    """
    After datasets have been processed by MLDS, this function provides an
    elegant interface for loading them. Per PEP 562
    (https://www.python.org/dev/peps/pep-0562), we can call module functions in
    an attribute-style (without instantiating an object).

    :param dataset: the dataset to retrieve arguments from
    :type dataset: string; one of the method defined below
    :return: complete dataset with metadata
    :rtype: namedtuple object
    """
    return save.read(dataset.lower())


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
    This function represents the heart of MachineLearningDatasets (MLDS). Given
    a dataset, a list of features, a list of label transformations, a list of
    output names, an output directory, numerical precision, a list of data
    transformation schemes, and whether to produce resultant analyitcs and
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

    # extract features (needed for "all" keyword) from first data partition
    part = next(iter(data))
    feat_names = data[part]["data"].columns
    print(f"Inferred {len(feat_names)} features from {part} partition.")

    # resovle "all" keyword to feature names minus those used in one-hot encoding
    print("Resolving 'all' keyword with inferred features...")
    ohot_feat = [
        f for s, f in zip(schemes, features) if transform.Transformer.onehotencoder in s
    ]
    all_feat = feat_names.difference(*ohot_feat, sort=False).tolist()
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
        for (transformed_data, transformed_labels), name in zip(
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
                **{"original_shape": data.get("oshape", transformed_data.shape)},
            }

            # save (with analytics, if desired)
            save.write(
                transformed_data,
                transformed_labels,
                name + "-" + part,
                metadata,
                precision=precision,
                analytics=analytics,
                outdir=outdir,
            )
    print(f"{dataset} retrieval, transformation, and export complete!")
    return None


if __name__ == "__main__":
    """
    This is the entry point when running MachineLearningDataSets via the
    command-line. It first parses arguments from the command line, sets the
    appropriate parameters, and calls main to execute the parsed instructions.

    Example usage:

        $ mlds nslkdd -f duration count -f service --outdir datasets
            -n nslkdd_ss nslkdd_mms -s standardscaler minmaxscaler
            -s onehotencoder -l labelencoder -a --destupefy

    This (1) downloads the NSL-KDD, (2) selects "duration" and "count" as one
    group and "service" as the second group, (3) specifies an alternative
    output directory (instead of "out/"), (4) changes the base dataset name (to
    "nslkdd_ss" and "nslkdd_mms") when saved, (5) creates two copies of the
    dataset: one where "duration" & "count" (subgroup one) are standaridized
    and another where they are rescaled, and, in both copies, "service" (group
    two) is one-hot encoded, (6) encodes labels as integers for both dataset
    copies, and (7) computes basic analytics, and (8) applies destupefication
    (to both dataset copies).
    """
    import arguments  # Command-line Argument parsing

    # parse command-line args and enter main
    main(**arguments.validate_args())
    raise SystemExit(0)