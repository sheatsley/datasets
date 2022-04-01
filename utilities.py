"""
The utilities module defines functions used throughout mlds.
Author: Ryan Sheatsley
Wed Mar 23 2022
"""
import collections  # Container datatypes
import builtins  # Built-in objects
import itertools  # Functions creating iterators for efficient looping
import matplotlib.pyplot as plt  # Visualization with Python
import mpl_toolkits.axes_grid1  # Application-specific functions that extend Matplotlib
import numpy as np  # The fundamental package for scientific computing with Python
import pathlib  # Object-oriented filesystem paths
import time  # Time access and conversions


def analyze(dataframe, labelframe, name, outdir=pathlib.Path("out/")):
    """
    This function analyzes a dataframe to produce various statisitcs of the
    underlying data. Specifically, this computes (1) basic dataset statisics,
    (2) feature histograms, color-coded based on the underlying class with
    class membership statistics, and (3) pearson correlation matricies.

    :param dataframe: dataset
    :type dataframe: pandas dataframe
    :param labelframe: labels
    :type labelframe: pandas series
    :param name: filename of dataset (and partition)
    :type name: str
    :param outdir: output directory
    :type outdir: pathlib path
    :return: None
    :rtype: NoneType
    """

    # (1) compute basic dataset statistics
    print("Computing dataset statistics...")
    fig = statistics(dataframe, name, figsize=(10, len(dataframe.columns) / 5))
    print(f"Statistics computed! Saving as {name}_statistics.pdf...")
    fig.savefig(outdir / f"{name}_statistics.pdf", bbox_inches="tight")

    # (2) compute feature histograms
    print(f"Computing histograms for dataframe of shape {dataframe.shape}...")
    rows = -(-(len(dataframe.columns) + 1) // 6)
    fig = histogram(dataframe, labelframe, name, rows=rows, cols=6)
    print(f"Histogram complete! Saving as {name}_histograms.pdf...")
    fig.savefig(outdir / f"{name}_histograms.pdf", bbox_inches="tight")

    # (3) compute pearson correlation matrix
    print(f"Computing Pearson correlations for dataframe of shape {dataframe.shape}...")
    fullframe = dataframe.join(labelframe.astype("category").cat.codes.rename("label"))
    fig = pearson(fullframe, name, size=len(fullframe.columns) // 2)
    print(f"Pearson correlations complete! Saving as {name}_pearson.pdf...")
    fig.savefig(outdir / f"{name}_pearson.pdf", bbox_inches="tight")
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


def histogram(dataframe, labelframe, name, rows, cols):
    """
    This function creates a histogram for each feature in a given dataframe and
    adds class membership statistics.

    :param dataframe: dataset
    :type dataframe: pandas dataframe
    :param labelframe: labels
    :type labelframe: pandas series
    :param name: title for legend
    :type name: str
    :param rows: number of subplot rows
    :type rows: int
    :param cols: number of subplot columns
    :type cols: int
    :return: plot-ready histogram figures
    :rtype: tuple of matplotlib figure and maplotlib axes
    """

    # compute class ratios and class-based indicies
    class_ratios = labelframe.value_counts(normalize=True)
    sorted_classes = class_ratios.index
    label_idxs = [labelframe == label for label in sorted_classes]

    # compute feature histogram (plot most common class first)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2 * rows))
    for idx, (feature, min_val, max_val, ax) in enumerate(
        zip(dataframe.columns, dataframe.min(), dataframe.max(), axes.flat)
    ):
        print(f"{idx/len(dataframe.columns):.1%} - Analyzing {feature}...", end="\r")
        ax.set_title(f"{idx}: {feature}")
        for label_idx, label in zip(label_idxs, sorted_classes):
            hist, bins = np.histogram(
                dataframe.loc[label_idx, feature], range=(min_val, max_val)
            )
            ax.bar(
                bins[:-1],
                hist / len(dataframe),
                align="edge",
                width=np.diff(bins),
                alpha=0.7,
                ec="k",
                label=label,
            )

    # add legend (with class statistics) & hide remaining axes
    print("Histograms added. Adding legend...")
    handles, labels = axes.flat[idx].get_legend_handles_labels()
    labels = [f"{c} - {r:.2%}" for c, r in zip(sorted_classes, class_ratios)]
    legend = axes.flat[idx + 1]
    legend.set_axis_off()
    legend.legend(
        handles=handles,
        labels=labels,
        frameon=False,
        ncol=len(sorted_classes) // 5 + (1 if len(sorted_classes) % 5 else 0),
        loc="center",
        title=f"{name} Class Distribution",
    )
    for idx in range(idx + 1, len(axes.flat)):
        axes.flat[idx].set_axis_off()
    fig.tight_layout()
    return fig


def pearson(dataframe, name, size):
    """
    This function computes Pearson correlation coefficients and plots the
    result as a heatmap with coefficients inside cells.

    :param dataframe: dataset
    :type dataframe: pandas dataframe
    :param name: figure title
    :type name: str
    :param size: figure size in inches (image is square)
    :type size: int
    :return: plot-ready pearson matrix figure
    :rtype: matplotlib figure
    """

    # compute correlations & compress figsize if needed
    correlations = dataframe.corr().round(2)
    features = len(correlations)
    size = min(2 ** 16 // plt.figure().get_dpi(), size)
    plt.close("all")
    fig, axes = plt.subplots(figsize=(size, size))
    art = axes.matshow(correlations)
    axes.set_yticks(range(features), labels=dataframe.columns)
    axes.set_xticks(range(features), labels=dataframe.columns, rotation=45)
    axes.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    for x, y in itertools.product(*itertools.tee(range(features))):
        print(f"{x/features:.1%} - Analyzing {dataframe.columns[x]}...", end="\r")
        axes.text(x, y, correlations.iloc[x, y], ha="center", va="center", color="w")

    # set title and add colorbar
    print("Correlations computed. Adding colorbar...")
    axes.set_title(f"{name} Pearson Correlation Matrix")
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(axes)
    fig.colorbar(art, cax=divider.append_axes("right", size="1%", pad=0.05))
    return fig


def print(*args, **kwargs):
    """
    This function wraps the print function, prepended with a timestamp.

    :param *args: positional arguments supported by print()
    :type *args: tuple
    :param **kwargs: keyword arguments supported by print()
    :type **kwargs: dictionary
    :return: None
    :rtype: NoneType
    """
    return builtins.print(f"[{time.asctime()}]", *args, **kwargs)


def statistics(dataframe, name, figsize, colors=np.array(["snow", "lightsteelblue"])):
    """
    This function computes basic dataset statistics via pandas decribe
    method.

    :param dataframe: dataset
    :type dataframe: pandas dataframe
    :param name: figure title
    :type name: str
    :param figsize: figure size in inches (width by length)
    :type figsize: tuple of floats
    :param colors: alternating colors for each row in the table
    :type colors: numpy array of str
    :return: plot-ready statistics table
    :rtype: tuple of matplotlib figure and maplotlib axes
    """

    # compute basic dataset statistics
    statsframe = dataframe.describe().T.round(3)
    statsframe.drop(columns="count", inplace=True)
    print("Statistics computed. Configuring table...")

    # configure table and add title
    feat_nums = [f"{f}: {i}" for i, f in enumerate(statsframe.index)]
    repeats = -(-len(statsframe) // len(colors))
    row_col = np.tile(colors, repeats)[: len(statsframe)]
    fig, axes = plt.subplots(figsize=figsize)
    axes.axis("off")
    table = axes.table(
        cellText=statsframe.values,
        colLabels=statsframe.columns,
        rowLabels=feat_nums,
        rowLoc="right",
        cellLoc="center",
        cellColours=row_col.repeat(len(statsframe.columns)).reshape(len(row_col), -1),
        rowColours=row_col,
        loc="center",
    )
    for pos, cell in table.get_celld().items():
        cell.set_height(1 / len(statsframe))
    axes.set_title(f"{name} Dataset Statistics\n")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    """
    This runs some basic unit tests with the functions defined in this module
    """
    print("Test string with implicit date.")
    raise SystemExit(0)
