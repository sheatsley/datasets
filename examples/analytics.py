"""
This script produces analytics for the datasets in MLDS. It produces figures of
feature histograms, and Pearson correlation matricies.
"""
import argparse

import matplotlib.pyplot as plt
import mlds
import numpy as np
import pandas
import seaborn


def main(datasets, override):
    """
    This function is the main entry point for producing analytics for
    downloaded datasets. Specifically, this: (1) skips datasets with over 500
    features unless override is set (as they are likely too large to reasonably
    generate) (2) casts datasets into pandas Dataframes, (3) prepends feature
    names with their index, (4) adds a column for class labels, (5) replaces
    class labels with their corresponding names, (6) melts the dataframe into a
    long format, (7) plots feature histograms, colored by class, and (8)
    produces Pearson correlation matricies.

    :param datasets: the dataset(s) to analyze
    :type datasets: tuple of str
    :param override: whether to generate figures for datasets with over 500 features
    :type override: bool
    :return: None
    :rtype: NoneType
    """
    for idx, dataname in enumerate(datasets, start=1):
        print(f"On dataset {idx}/{len(datasets)}: {dataname}")
        dataset = getattr(mlds, dataname)
        for partition in dataset.metadata["partitions"]:
            if not override and (total := getattr(dataset, partition).features) > 500:
                print(f"Skipping {dataname} {partition} ({total}> 500 features)...")
                continue

            partition_name = f"{dataname}-{partition}"
            class_map = dataset.metadata[partition].get("class_map")
            columns = dataset.metadata[partition]["features"]
            idx_cols = [f"{i}: {f}" for i, f in enumerate(columns)]

            # create dataframe with feature names and class labels
            data = pandas.DataFrame(getattr(dataset, partition).data, columns=idx_cols)
            data.insert(len(idx_cols), "class", getattr(dataset, partition).labels)
            data["class"].replace(inplace=True, to_replace=class_map)
            wide = data.melt(id_vars="class", var_name="feature", value_vars=idx_cols)

            # plot figures
            print(f"Plotting {partition} histograms...")
            plot_histogram(dataname=partition_name, dataset=wide)
            print(f"Plotting {partition} correlation matricies...")
            plot_correlation_matrix(dataname=partition_name, dataset=data.corr())
    return None


def plot_correlation_matrix(dataname, dataset):
    """
    This function plots a Pearson correlation matrix. Specifically, this
    produces a diagonal correlation matrix with colored cells, and a colorbar.

    :param dataname: name of the dataset
    :type dataname: str
    :param dataset: dataset of correlation coefficients
    :type dataset: pandas Dataframe object
    :return: None
    :rtype: NoneType
    """
    size = tuple(len(dataset.columns) * dim for dim in (0.48, 0.39))
    _, ax = plt.subplots(figsize=size)
    cmap = seaborn.diverging_palette(230, 20, as_cmap=True)
    mask = np.triu(np.ones_like(dataset, dtype=bool))
    short_names = tuple(f"{f[:10]}..." if len(f) > 13 else f for f in dataset.columns)
    dataset.columns = short_names
    dataset.index = short_names
    plot = seaborn.heatmap(
        ax=ax,
        cbar_kws=dict(shrink=0.5),
        center=0,
        cmap=cmap,
        data=dataset,
        linewidth=0.5,
        mask=mask,
        square=True,
        vmax=1,
        vmin=-1,
    )
    plot.set(title=f"dataset = {dataname}")
    plot.figure.savefig(
        __file__[:-3] + f"_correlation_matrix_{dataname}.pdf", bbox_inches="tight"
    )
    return None


def plot_histogram(dataname, dataset):
    """
    This function plots a set of histograms. Specifically, this produces
    feature-number of histograms with normalized bar heights, colored per
    class.

    :param dataname: name of the dataset
    :type dataname: str
    :param dataset: dataset to plot histogram for
    :type dataset: pandas Dataframe object
    :return: None
    :rtype: NoneType
    """
    with seaborn.plotting_context(context="notebook", font_scale=2):
        plot = seaborn.displot(
            aspect=1.5,
            bins=10,
            data=dataset,
            col="feature",
            col_wrap=6,
            common_norm=False,
            hue="class",
            stat="percent",
            x="value",
        )
        seaborn.move_legend(
            bbox_to_anchor=(0.25, 0.995),
            frameon=False,
            loc="lower center",
            ncols=len(dataset["class"].unique()),
            obj=plot,
            title="Classes",
        )
        plot.fig.suptitle(f"dataset = {dataname}", x=0.5, y=1)
    plot.set(xlabel=None)
    plot.savefig(__file__[:-3] + f"_histogram_{dataname}.pdf", bbox_inches="tight")
    return None


if __name__ == "__main__":
    """
    This script analyzes downloaded datasets from MLDS. Specifically, this
    script: (1) parses command-line arguments, (2) loads dataset(s) from disk,
    and (3) produces figures of feature histograms, and Pearson correlation
    matricies.
    """
    parser = argparse.ArgumentParser(description="Produce dataset analytics.")
    parser.add_argument(
        "-d",
        "--datasets",
        choices=mlds.__available__,
        default=mlds.__available__,
        help="Dataset(s) to analyze",
        nargs="+",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Try to generate figures for datasets with over 500 features",
    )
    args = parser.parse_args()
    main(datasets=tuple(args.datasets), override=args.override)
    raise SystemExit(0)
