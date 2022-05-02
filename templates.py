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

    (1) cicmalmem2022 (CIC-MalMem-2022): Malware Detection
    (2) fashionmnist (Fashion-MNIST): Fashion Products Recognition
    (3) mnist (MNIST): Handwritten Digit Recognition
    (4) nslkdd (NSL-KDD): Network Intrusion Detection
    (5) phishing (Phishing dataset): Phishing Detection
    (6) unswnb15 (UNSW-NB15): Network Intrusion Detection
    """

    cicmalmem2022 = {
        "features": [["all"]],
        "labels": [transform.Transformer.labelencoder],
        "names": ["cicmalmem2022"],
        "schemes": [[transform.Transformer.minmaxscaler]],
    }
    cifar10 = {
        "features": [["all"]],
        "labels": [transform.Transformer.identity],
        "names": ["cifar10"],
        "schemes": [[transform.Transformer.minmaxscaler]],
    }
    fashionmnist = {
        "features": [["all"]],
        "labels": [transform.Transformer.identity],
        "names": ["fmnist"],
        "schemes": [[transform.Transformer.minmaxscaler]],
    }
    mnist = {
        "features": [["all"]],
        "labels": [transform.Transformer.identity],
        "names": ["mnist"],
        "schemes": [[transform.Transformer.minmaxscaler]],
    }
    nslkdd = {
        "features": [["all"], ["protocol_type", "service", "flag"]],
        "labels": [transform.Transformer.labelencoder],
        "names": ["nslkdd"],
        "schemes": [
            [transform.Transformer.minmaxscaler],
            [transform.Transformer.onehotencoder],
        ],
    }
    phishing = {
        "features": [
            ["all"],
            [
                "SubdomainLevelRT",
                "UrlLengthRT",
                "PctExtResourceUrlsRT",
                "AbnormalExtFormActionR",
                "ExtMetaScriptLinkRT",
                "PctExtNullSelfRedirectHyperlinksRT",
            ],
        ],
        "labels": [transform.Transformer.labelencoder],
        "names": ["phishing"],
        "schemes": [
            [transform.Transformer.minmaxscaler],
            [transform.Transformer.onehotencoder],
        ],
    }
    unswnb15 = {
        "features": [
            ["all"],
            ["proto", "service", "state"],
        ],
        "labels": [transform.Transformer.labelencoder],
        "names": ["unswnb15"],
        "schemes": [
            [transform.Transformer.minmaxscaler],
            [transform.Transformer.onehotencoder],
        ],
    }


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
