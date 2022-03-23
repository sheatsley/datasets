"""
The utils module defines functions used throughout mlds.
Author: Ryan Sheatsley
Fri Jun 18 2021
"""
import collections  # Container datatypes
import builtins  # Built-in objects
import matplotlib.pyplot as plt  # Visualization with Python
import pathlib  # Object-oriented filesystem paths
import time  # Time access and conversions

# TODO
# save n_samples & features, n_classes, class breakdown, means & medians & stds


def analyze(dataframe, labelframe, name, outdir=pathlib.Path("out/"), ppr=4):
    """
    This function analyzes a dataframe to produce various statisitcs of the
    underlying data. Specifically, this computes (1) feature histograms,
    color-coded based on the underlying class, (2) pearson correlation
    matricies, and (3) class membership statistics.

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

    # compute feature histogram (plot most common class first)
    print(f"Computing {name} analytics; this will take some time...")
    print(f"Computing histograms for dataframe of shape {dataframe.shape}...")
    class_ratios = labelframe.value_counts(normalize=True)
    sorted_classes = class_ratios.index
    label_idxs = [labelframe == label for label in sorted_classes]
    rows = int(((dataframe.shape[1] + 1) / ppr) + 0.5)
    fig, axes = plt.subplots(rows, ppr, figsize=(3 * rows, 2 * ppr))
    for idx, (feature, ax) in enumerate(zip(dataframe.columns, axes.flat)):
        print(f"Analyzing {feature}... ({idx/dataframe.shape[1]:.1%})\r", end="")
        ax.set_title(feature)
        for label_idx, label in zip(label_idxs, sorted_classes):
            ax.hist(
                dataframe.loc[label_idx, feature],
                bins="auto",
                density=True,
                alpha=0.7,
                label=label,
            )

    # add legend in the last subplot and save
    print("Histograms complete! Adding legend and saving...")
    handles, labels = axes.flat[idx].get_legend_handles_labels()
    labels = [f"{c} {r:.1%}" for c, r in zip(sorted_classes, class_ratios)]
    legend = axes.flat[idx + 1]
    legend.set_axis_off()
    legend.legend(handles, labels, frameon=False, title="Classes")
    fig.savefig(outdir / (name + "_histograms"))
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
