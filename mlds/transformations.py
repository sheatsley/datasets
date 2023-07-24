"""
This module defines aliases and classes for transforming and cleaning datasets.
"""

import sklearn.preprocessing

# the following sklearn transformations are supported
LabelEncoder = sklearn.preprocessing.LabelEncoder
MinMaxScaler = sklearn.preprocessing.MinMaxScaler
OneHotEncoder = sklearn.preprocessing.OneHotEncoder
RobustScaler = sklearn.preprocessing.RobustScaler
StandardScaler = sklearn.preprocessing.StandardScaler
UniformScaler = sklearn.preprocessing.FunctionTransformer(
    lambda x: (x - x.min()) / (x.max() - x.min())
)


class Destupefier(sklearn.base.TransformerMixin):
    """
    The Destupefier is a stateful transformer, such as those in
    sklearn.preprocessing module. After dataframes have been assembled, this
    class "cleans" them by identifying data deficiencies. Specifically, this
    removes: (1) duplicate columns, (2) duplicate rows, and (3) single-value
    columns. For duplicate rows and columns, the first occurance is kept.

    This class is a stateful transformer as, for datasets that have dedicated
    training and test sets, the same transformations must be applied to both,
    regardless if one exhibits a deficiency and the other does not. For
    example, if a training set has a feature with a single value, then that
    feature will be removed from both the training and test sets.

    :func:`fit`: identifies dataset-wide deficiencies
    :func:`transform`: corrects dataset-wide and parition-specific deficiencies
    """

    def fit(self, dataset, _=None):
        """
        This method identifies deficiencies that must be applied to the entire
        dataset. Specifically, this identifies duplicate and single-value
        columns.

        :param dataset: the training data to fit to
        :type dataset: pandas DataFrame object
        :param _: the training labels to fit to (not used)
        :type _: pandas Series object
        :return: fitted Destupefier
        :rtype: Destupefier object
        """
        print(f"Analyzing {len(dataset.columns)} columns for deficiencies...")
        duplicates = dataset.columns[dataset.T.duplicated()]
        singles = dataset.columns[dataset.nunique() == 1]
        self.deficient = set().union(duplicates).union(singles)
        return self

    def transform(self, dataset, labels):
        """
        This method corrects deficiencies identified in fit. Specifically, this
        removes duplicate and single-value columns, as well as duplicate rows.
        The first occurance of duplicate rows and columns are kept.

        :param dataset: the training data to clean
        :type dataset: pandas DataFrame object
        :param labels: the training labels to clean
        :type labels: pandas Series object
        :return: cleaned dataset
        :rtype: tuple of pandas DataFrame and Series objects
        """
        print(f"Dropping {len(self.deficient)} deficient features...")
        dataset.drop(columns=self.deficient, inplace=True)
        print(f"Scanning {len(dataset)} samples for duplicates...")
        duplicates = dataset.duplicated()
        dataset.drop(index=dataset.index[duplicates], inplace=True)
        dataset.reset_index(inplace=True)
        labels.drop(labels.index[duplicates], inplace=True)
        labels.reset_index(inplace=True)
        return dataset, labels
