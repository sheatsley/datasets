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


def analyze(dataframe, labelframe, name, outdir=pathlib.Path("out/"), ppr=4):
    """
    This function analyzes a dataframe to produce various statisitcs of the
    underlying data. Specifically, this computes (1) feature histograms,
    color-coded based on the underlying class, (2) pearson correlation
    matricies, (3) class membership statistics, and (4) basic dataset
    statisics.

    :param dataframe: dataset
    :type dataframe: pandas dataframe
    :param labels: labels
    :type labels: pandas series
    :param name: filename of dataset (and partition)
    :type name: str
    :param outdir: output directory
    :type outdir: pathlib path
    :param ppr: plots per row
    :type ppr: int
    :return: None
    :rtype: NoneType
    """

    # (1): compute class ratios, class-based indicies, and number of figure rows
    print(f"Computing histograms for dataframe of shape {dataframe.shape}...")
    class_ratios = labelframe.value_counts(normalize=True)
    sorted_classes = class_ratios.index
    label_idxs = [labelframe == label for label in sorted_classes]
    rows = -(-(len(dataframe.columns) + 1) // ppr)

    # (1): compute feature histogram (plot most common class first)
    fig, axes = plt.subplots(rows, ppr, figsize=(3 * ppr, 2 * rows))
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

    # (1) & (2): add legend in the last subplot (with class statistics)
    print("Histograms complete! Adding legend and saving...")
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

    # (1) & (2): remove axes from remaining plots, cleanup appearance, and save
    for idx in range(idx + 1, len(axes.flat)):
        axes.flat[idx].set_axis_off()
    dpi = int(min(fig.get_dpi(), 65536 / (3 * ppr), 65536 / (2 * rows)))
    print(f"Fig is too big! Dropping dpi to {dpi}...") if dpi != fig.get_dpi() else None
    fig.set_dpi(dpi)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_histograms", bbox_inches="tight")

    # (3): compute pearson correlation matricies
    print(f"Computing Pearson correlations for dataframe of shape {dataframe.shape}...")
    fullframe = dataframe.join(labelframe.astype("category").cat.codes.rename("label"))
    correlations = fullframe.corr().round(2)
    cols = len(fullframe.columns)
    fig, ax = plt.subplots(figsize=(cols / 2, cols / 2))
    art = ax.matshow(correlations)
    ax.set_yticks(range(cols), labels=fullframe.columns)
    ax.set_xticks(range(cols), labels=fullframe.columns, rotation=45)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    for x, y in itertools.product(*itertools.tee(range(cols))):
        print(f"{x / cols:.1%} - Analyzing {fullframe.columns[x]}...", end="\r")
        ax.text(x, y, correlations.iloc[x, y], ha="center", va="center", color="w")

    # (3) set title, add colorbar, and save
    print("Pearson correlations complete! Adding colorbar and saving...")
    ax.set_title(f"{name} Pearson Correlation Matrix")
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    fig.colorbar(art, cax=divider.append_axes("right", size="1%", pad=0.05))
    dpi = int(min(fig.get_dpi(), 65536 / (cols / 2)))
    print(f"Fig is too big! Dropping dpi to {dpi}...") if dpi != fig.get_dpi() else None
    fig.set_dpi(dpi)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_correlation", bbox_inches="tight")

    # (4) compute basic dataset statistics
    print("Computing dataset statistics...")
    statsframe = dataframe.describe().T.round(3)
    statsframe.drop(columns="count", inplace=True)

    # (4) configure table and save
    print("Statistics computed! Configuring table and saving...")
    feat_nums = [f"{f}: {i}" for i, f in enumerate(statsframe.index)]
    cell_col = np.array(
        [
            ["snow" if i % 2 else "lightsteelblue"] * len(statsframe.columns)
            for i in range(len(statsframe))
        ]
    )
    fig, ax = plt.subplots(figsize=(10, len(statsframe) // 5))
    ax.axis("off")
    ax.table(
        cellText=statsframe.values,
        colLabels=statsframe.columns,
        rowLabels=feat_nums,
        rowLoc="right",
        cellLoc="center",
        cellColours=cell_col,
        rowColours=cell_col[:, 0],
        loc="center",
    )
    dpi = int(min(fig.get_dpi(), 65536 / (len(statsframe) // 5)))
    print(f"Fig is too big! Dropping dpi to {dpi}...") if dpi != fig.get_dpi() else None
    fig.set_dpi(dpi)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_statistics", bbox_inches="tight")
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


if __name__ == "__main__":
    """
    This runs some basic unit tests with the functions defined in this module
    """
    print("Test string with implicit date.")
    raise SystemExit(0)
