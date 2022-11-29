"""
This module defines the main entry point for the machine learning datasets
repo. Transforming machine learning datasets into useful representations
requires a substantially large toolset (i..e, imports), while reading
transformed datasets should be as fast as possible. To support these
paradigms, this module segments which dependencies are loaded based
on whether data is to be transformed or simply retrieved.
Author: Ryan Sheatsley
Mon Apr 4 2022
"""


class Dataset:
    """
    The Dataset class represents the main interface for working with loaded
    datasets via the load method in this module. Specifically, it: (1) provides
    an intuitive representation of loaded partitions (including the partitions,
    the number of samples, and associated metadata), and (2) defines wrappers
    for casting numpy arrays into popular datatypes used in machine learning
    (e.g., PyTorch Tensors).

    :func:`__init__`: prepares loaded data
    :func:`__repr__`: shows useful dataset statistics
    """

    def __init__(self, name, partitions):
        """
        This method instantiates Dataset objects and builds a representation
        from the paired metadata.

        :param name: dataset name (i.e., filename)
        :type name: str
        :param partitions: the loaded data partitions
        :type partitions: list of tuples of form: (partition, namedtuple)
        :return: a dataset
        :rtype: Dataset object
        """
        import collections  # Container datatypes

        # set partitions and data
        for part, data in partitions:
            setattr(
                self,
                part,
                collections.namedtuple(part.capitalize(), ("data", "labels"))(
                    data.data, data.labels
                ),
            )
            for field in (f for f in data._fields if f not in {"data", "labels"}):
                setattr(self, field, getattr(data, field))

        # build representation from name and metadata
        samples, features, classes = zip(
            *[
                (
                    f"{part}={data.data.shape[0]}",
                    data.data.shape[1],
                    len(set(data.labels)),
                )
                for part, data in partitions
            ]
        )
        transformations = ", ".join(self.transformations)
        self.name = (
            f"{name}(samples={samples}, features={features[0]}, "
            f"classes={classes[0]}, transformations=({transformations}))"
        )
        return None

    def __repr__(self):
        """
        This method returns a string-based representation of useful metadata
        for debugging.

        :return: dataset statistics
        :rtype: str
        """
        return self.name


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
    if dataset == "__path__":
        raise AttributeError
    return load(dataset.lower())


def load(dataset, out="out/"):
    """
    This function is the main entry point from datasets that been proccessed
    and saved by MLDS. It depickles saved data, and returns the namedtuple
    structure defined in the utilities module (i.e., the assemble function).
    Importantly, loading datasets executes the following based on the dataset
    arugment (parsed as {dataset}-{partition}): (1) if the partition is
    specified, then that partition is returned, else, (2) if the specified
    partition is "all", then *all* partitions are returned, else, if a
    partition is *not* specified and (3) there is only one partition, it is
    returned, else (4) if "train" & "test" are partitions, they are returned,
    else (4) an error is raised.

    :param dataset: the dataset to load
    :type dataset: str
    :param out: directory where the dataset is saved
    :type out: pathlib path
    :return: loaded dataset
    :rtype: namedtuple
    """
    import dill  # serialize all of python
    import pathlib  # Object-oriented filesystem paths

    # check if a partition is specified
    out = pathlib.Path(__file__).parent / "out/" if out == "out/" else pathlib.Path(out)
    print(f"Loading {dataset}.pkl from {out}/...")
    data, part = split if len(split := dataset.rsplit("-", 1)) == 2 else (dataset, None)
    partitions = set(out.glob(f"{dataset}*.pkl"))
    part_stems = [p.stem.rsplit("-")[1] for p in partitions]

    # case 1: the partition is specified
    if part and part != "all":
        with open(out / f"{dataset}.pkl", "rb") as f:
            return Dataset(data, [(part, dill.load(f))])

    # case 2: the partition is "all"
    elif part and part == "all":
        datasets = []
        for part in partitions:
            print(f"Loading {part.stem} partition...")
            with open(part, "rb") as f:
                datasets.append((part, dill.load(f)))
        return Dataset(data, datasets)

    # case 3: there is only one partition
    elif len(partitions) == 1:
        with open(*partitions, "rb") as f:
            return Dataset(data, [("dataset", dill.load(f))])

    # case 4: train and test are available
    elif all(p in part_stems for p in ("train", "test")):
        datasets = []
        for part in ("train", "test"):
            print(f"Loading {part} partition...")
            with open(out / (f"{'-'.join([dataset, part])}.pkl"), "rb") as f:
                datasets.append((part, dill.load(f)))
        return Dataset(data, datasets)

    # case 5: the pickle was not found
    raise FileNotFoundError(f"{dataset} not found in '{out}'! (Is it downloaded?)")


def main():
    """
    This is the entry point when running MachineLearningDataSets via the
    command-line. It first parses arguments from the command line, sets the
    appropriate parameters, and calls main to execute the parsed instructions.

    Example usage:

        $ mlds nslkdd -f duration count -f service --outdir datasets
            -n nslkdd_ss nslkdd_mms -s standardscaler minmaxscaler
            -s onehotencoder -l labelencoder -a -d

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
    import mlds.arguments as arguments  # Command-line Argument parsing
    import mlds.datasets as datasets  # Machine learning dataset transformations

    # parse command-line args and enter main
    raise SystemExit(datasets.main(**arguments.validate_args()))


if __name__ == "__main__":
    main()
