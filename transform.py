"""
The transform module applies transformations to machine learning datasets.
Author: Ryan Sheatsley
Tue Feb 15 2022
"""
import itertools  # Functions creating iterators for efficient looping¶
from utilities import print  # Timestamped printing

# TODO
# add print statements
# destupify should cleanse for unknown values: min(drop_samples, drop_columns)


class Transformer:
    """
    This transformer class servs as an intelligent wrapper for scikit-learn's
    data transformation functions. Notably, the transformers (and
    ColumnTransformer compositions) do not implicitly preserve the order of
    features post-transformation. This class enables arbitrary data
    transformation, while conforming to the standard layout the data was
    originally presented in.

    :func:`__init__`: instantiates Transformer objects
    :func:`apply`: applies transformation schemes to the data
    :func:`export`: correctly concatenates transformations to the original data
    :func:`destupify`: automagic data cleaning (experimental)
    :func:`labelencoder`: encode target labels between 0 and n_classses-1
    :func:`minmaxscaler`: scale features to a given range
    :func:`onehotencoder`: encode categorical features as one-hot arrays
    :func:`robustscaler`: scale features with statistics robust to outliers
    :func:`standardscaler`: standardize features to zero mean and unit variance
    """

    def __init__(self, features, schemes):
        """
        This function initializes Transformer objects with the necessary
        information to apply arbitrary transformations to data. Specifically,
        the manipulated features are expected to be tuple of tuples, wherein the
        number of datasets is the product of the lengths of the tuples. This
        enables, for example, creation of multiple datasets with different
        feature transformation schemes (described by the first tuple), all with
        label encoding (described by the second tuple). Consequently, the
        transformations applied to the features within each tuple is described
        by schemes, which is expected to be tuple of length equal to the number
        of tuples (in features) containing Transformer method callables.

        :param features: the features to manipulate
        :type features: tuple of tuples containing indicies
        :param schemes: transformations to apply to the data
        :type schemes: tuple of tuples of Transformer callables
        :return: a prepped transformer
        :rtype: Transformer object
        """
        self.transformations = []
        self.original = None
        self.features = features
        self.schemes = schemes
        return None

    def apply(self, train, test=None):
        """
        This method applies sckilit-learn data transformations, while
        preserving the original layout of the data. Importantly, this method
        applies the transformations and stores them internally. Once the export
        method is called, the transformations are retrieved, concatenated into
        their original indicies, and returned (and thus, this method returns
        nothing).

        :param train: the dataset to transform
        :type train: pandas dataframe
        :param test: the test set to transform (if applicable)
        :type test: pandas dataframe
        :return: None
        :rtype: NoneType
        """
        for scheme, feature in zip(self.schemes, self.features):

            # fit to training, transform to train & test
            print(f"Applying {scheme} to features {feature}...")
            self.tranformations.append(
                (s(train[feature], test[feature] if test else None) for s in scheme)
            )

        # save the original dataframe
        self.original = (train, test if test else None)
        return None

    def export(self, feature_names=None):
        """
        This method properly assembles datasets based on the applied
        transformations. Specifically, it concatenates transformations to
        preserve their original orders and produces n copies of the dataset,
        where n is the product of the length of the tuples in
        self.transformations. It takes no arguments so that the building
        operation (which will memory-intensive) can be called when it is most
        appropriate (other than feature_names, which can optionally set the column
        headers, of which a subset are modified if one-hot encoding was used).

        :param feature_names: names of the features
        :type feature_names: tuple of strings
        :return: the transformed datasets
        :rtype: generator of pandas dataframes
        """
        for features, schemes, transformed in zip(
            self.features,
            itertools.product(*self.schemes),
            itertools.product(*self.transformations),
        ):
            # assemble the dataframe based on the original indicies
            print(f"Exporting {'×'.join(*schemes)}...")
            yield self.original[0].assign()
        return


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
