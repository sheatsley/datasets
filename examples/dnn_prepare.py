"""
This script downloads available dataset(s) in MLDS, applies commonly used
transformations for training deep neural networks, and saves them to disk.
"""
import argparse

import mlds


def main(datasets):
    """
    This function is the main entry point for downloading and preparing
    datasets for their use with deep neural networks. Specifically, this calls
    mlds.datasets.process function for each dataset with: (1) the desired
    dataset, (2) UniformScaler for image dataset transformations, and,
    MinMaxScaler, and OneHotEncoder for other dataset transformations (where
    appropriate), (3) data-cleaning for non-image datasets (via the
    Destupifier), (4) sets of features that correspond to the appropriate
    transformer, (5) filenames set to the name of the dataset, and (6) sets the
    label transformation to LabelEncoder (where appropriate).

    :param datasets: the dataset(s) to download and process
    :type datasets: tuple of mlds.downloader module objects
    """

    # define feature transformations for each dataset
    transformations = {
        mlds.downloaders.cicmalmem2022: {
            "data_transforms": (mlds.transformations.MinMaxScaler,),
            "destupefy": True,
            "features": (("all",),),
            "filename": "cicmalmem2022",
            "label_transform": mlds.transformations.LabelEncoder,
        },
        mlds.downloaders.cifar10: {
            "data_transforms": (mlds.transformations.UniformScaler,),
            "destupefy": False,
            "features": (("all",),),
            "filename": "cifar10",
            "label_transform": mlds.transformations.IdentityTransformer,
        },
        mlds.downloaders.fashionmnist: {
            "data_transforms": (mlds.transformations.UniformScaler,),
            "destupefy": False,
            "features": (("all",),),
            "filename": "fashionmnist",
            "label_transform": mlds.transformations.IdentityTransformer,
        },
        mlds.downloaders.mnist: {
            "data_transforms": (mlds.transformations.UniformScaler,),
            "destupefy": False,
            "features": (("all",),),
            "filename": "mnist",
            "label_transform": mlds.transformations.IdentityTransformer,
        },
        mlds.downloaders.nslkdd: {
            "data_transforms": (
                mlds.transformations.MinMaxScaler,
                mlds.transformations.OneHotEncoder,
            ),
            "destupefy": True,
            "features": (
                ("all",),
                (
                    "protocol_type",
                    "service",
                    "flag",
                ),
            ),
            "filename": "nslkdd",
            "label_transform": mlds.transformations.LabelEncoder,
        },
        mlds.downloaders.phishing: {
            "data_transforms": (
                mlds.transformations.MinMaxScaler,
                mlds.transformations.OneHotEncoder,
            ),
            "destupefy": True,
            "features": (
                ("all",),
                (
                    "SubdomainLevelRT",
                    "UrlLengthRT",
                    "PctExtResourceUrlsRT",
                    "AbnormalExtFormActionR",
                    "ExtMetaScriptLinkRT",
                    "PctExtNullSelfRedirectHyperlinksRT",
                ),
            ),
            "filename": "phishing",
            "label_transform": mlds.transformations.IdentityTransformer,
        },
        mlds.downloaders.unswnb15: {
            "data_transforms": (
                mlds.transformations.MinMaxScaler,
                mlds.transformations.OneHotEncoder,
            ),
            "destupefy": True,
            "features": (
                ("all",),
                (
                    "proto",
                    "service",
                    "state",
                ),
            ),
            "filename": "unswnb15",
            "label_transform": mlds.transformations.LabelEncoder,
        },
    }

    # process each dataset
    for idx, dataset in enumerate(datasets, start=1):
        print(f"On dataset {idx}/{len(datasets)}: {dataset.__name__.split('.').pop()}")
        mlds.datasets.process(dataset, **transformations[dataset])
    return None


if __name__ == "__main__":
    """
    This script downloads available dataset(s) in MLDS, applies commonly used
    transformations for deep neural networks, and saves them to disk.
    Specifically, this script: (1) parses command-line arguments, (2) downloads
    dataset(s), (3) applies label encoding to datasets with non-numeric (or
    non-zero-indexed) labels, scales image datasets uniformly to [0, 1] (via
    UniformScaler), while other datasets are scaled to [0, 1] per feature (via
    MinMaxScaler) categorical variables are one-hot encoded (via OneHotEncoder)
    and deficiencies are removed (via Destupifier), and (4) saves the
    transformed dataset(s) to disk as the name of the dataset (which can then
    be subsequently loaded and used in other scripts).
    """
    datasets = tuple(getattr(mlds.downloaders, d) for d in mlds.downloaders.__all__)
    parser = argparse.ArgumentParser(
        description="Download and transform dataset(s) for deep neural networks"
    )
    parser.add_argument(
        "-d",
        "--datasets",
        choices=datasets,
        default=datasets,
        help="Dataset(s) to download",
        metavar=", ".join(mlds.downloaders.__all__),
        nargs="+",
        type=lambda d: getattr(mlds.downloaders, d),
    )
    args = parser.parse_args()
    main(datasets=tuple(args.datasets))
    raise SystemExit(0)
