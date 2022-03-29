"""
The templates module provides typical transformations (by setting command line
args) applied to machine learning datasets.
Author: Ryan Sheatsley
Tue Mar 29 2022
"""
import transform  # Order-preserving transformations for machine learning data


class Templates:
    """
    The Templates class provides pre-defined arguments to mlds that apply
    standard transformations to datasets so that they are ready-to-use for
    popular machine learning algorithms. Specifically, this entails: (1)
    continous features are rescaled to [0, 1], (2) categorical variables are
    one-hot encoded, (3) labels are encoded as integers (between 0 and
    number_of_classes-1), (4) transformed dataset file names are set to the
    original dataset name. All attributes are defined within class namespace.
    Transformations for the following datasets are provided:

    (1) nslkdd (NSL-KDD): Network Intrusion Detection
    """

    nslkdd = {
        "features": [["all"], ["protocol_type", "service", "flag"]],
        "labels": [transform.Transformer.labelencoder],
        "names": ["nslkdd"],
        "schemes": [
            [transform.Transformer.minmaxscaler],
            [transform.Transformer.onehotencoder],
        ],
    }


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
