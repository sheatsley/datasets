"""
The transform module applies transformations to machine learning datasets.
Author: Ryan Sheatsley
Tue Feb 15 2022
"""
import itertools  # Functions creating iterators for efficient looping¶
import sklearn.preprocessing  # Preprocessing and Normalization
from utilities import print  # Timestamped printing

# TODO
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
    :func:`raw`: no-op (return data unchanged)
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
        self.features = features
        self.schemes = schemes
        self.transformations = []
        self.ohe_f = []
        self.original = None
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

        # save the original dataframe (temporarily convert column headers to strings)
        train.set_axis(train.columns.astype(str), axis=1, inplace=True)
        test.set_axis(test.columns.astype(str), axis=1, inplace=True) if test else None
        self.original = {"train": train, "test": None}
        return None

    def export(self, feature_names=None):
        """
        This method properly assembles datasets based on the applied
        transformations. Specifically, it concatenates transformations to
        preserve their original orders and produces n copies of the dataset,
        where n is the product of the length of the tuples in
        self.transformations. It takes no arguments so that the building
        operation (which will be memory-intensive) can be called when it is
        most appropriate (other than feature_names, which can optionally set
        the column headers, of which a subset are modified if one-hot encoding
        was used).

        :param feature_names: names of the features
        :type feature_names: tuple of strings
        :return: the transformed datasets
        :rtype: generator of pandas dataframes
        """
        for features, schemes, (train_transform, test_transform) in zip(
            itertools.repeat(self.features),
            itertools.product(*self.schemes),
            itertools.product(*self.transformations),
        ):
            # assemble the dataframe based on the original indicies
            print(f"Exporting {'×'.join(*schemes)}...")
            training = self.original["train"].assign(
                **{
                    str(f): t[idx]
                    for idx, (f_tup, t) in enumerate(zip(features, train_transform))
                    for f in f_tup
                }
            )
            testing = (
                self.original["test"].assign(
                    **{
                        str(f): t[idx]
                        for idx, (f_tup, t) in enumerate(zip(features, test_transform))
                        for f in f_tup
                    }
                )
                if test_transform
                else None
            )

            # set feature_names (and correct for one-hot encodings)
            if feature_names:
                try:
                    ohotf = features[schemes.index("onehotencoder")]
                    for idx, ohf in enumerate(ohotf):

                        # this is a dynamically changing list; offset each iteration
                        ohf += sum((len(self.ohe_f[i]) - 1 for i in range(idx)))
                        print(f"Expanding {feature_names[ohf]} to {self.ohe_f[idx]}...")
                        feature_names[ohf : ohf + 1] = self.ohe_f[idx]
                except ValueError:
                    pass
                training.set_axis(feature_names, axis=1, inplace=True)
                testing.set_axis(
                    feature_names, axis=1, inplace=True
                ) if testing else None

            # print final shape as a sanity check and yield
            print(
                f"{'×'.join(*schemes)} export complete! Final shape(s):",
                training.shape,
                testing.shape if testing else "",
            )
            yield (training, testing) if testing else training

    def destupify(self, data):
        """
        This method attempts to clean datasets after they have been processed.
        At this time, destupify performs the following:

            - Removes any single-value columns
            - Removes any identical rows

        Note, this method is experimental.

        :param dataset: the dataset to cleanse
        :type dataset: pandas dataframe
        :return: the cleaned datasets
        :rtype: pandas dataframe
        """
        print("Analyzing dataset for single-value columns...")
        org_s, org_f = data.shape
        data = data[data.columns[data.nunique() > 1]]
        print(f"Dropped {data.shape[1]-org_f} features! Removing duplicate samples...")
        data = data.drop_duplicates()
        print(f"Dropped {data.shape[0]-org_s} samples! (New shape: {data.shape})")
        return data

    def labelencoder(self, train_labels, test_labels=None):
        """
        This method serves as a simple wrapper for scikit-learn's LabelEncoder.
        Notably, we expect both training and testing labels as input (as any
        deficient test set that lacks samples from a given class will still be
        processed correctly).

        :param train_labels: the training labels to transform
        :type train_labels: pandas series
        :param test_labels: the testing labels to transform
        :type test_labels: pandas series
        :return: integer-encoded labels
        :rtype: tuple of pandas series
        """
        encoder = sklearn.preprocessing.LabelEncoder()
        train_labels = encoder.fit_transform(train_labels)
        print(f"Transformed {len(encoder.classes_)} classes.")
        return train_labels, encoder.transform(test_labels) if test_labels else None

    def minmaxscaler(self, train, test=None):
        """
        This method serves as a simple wrapper for scikit-learn's MinMaxScaler.

        :param train: the training data to tranform
        :type train: pandas dataframe
        :param test: the testing data to transform
        :type test: pandas dataframe
        :return: transformed data
        :rtype: tuple of pandas series
        """
        print(f"Applying min-max scaling to data of shape {train.shape}...")
        scaler = sklearn.preprocessing.MinMaxScaler()
        train = scaler.fit_transform(train)
        return train, scaler.transform(test) if test else None

    def onehotencoder(self, train, test=None):
        """
        This method serves as a simple wrapper for scikit-learn's
        OneHotEncoder. Importantly, the categories_ attribute is saved on each
        call to this method to facilitate correct feature labels when
        feature_names is passed into the export method.

        :param train: the training data to tranform
        :type train: pandas dataframe
        :param test: the testing data to transform
        :type test: pandas dataframe
        :return: transformed data
        :rtype: tuple of pandas dataframes
        """
        print(f"Applying one-hot encoding to data of shape {train.shape}...")
        encoder = sklearn.preprocessing.OneHotEncoder()
        train = encoder.fit_transform(train)
        self.ohe_f += encoder.categories_
        print(f"Encoding complete. Expanded shape: {train.shape}")
        return train, encoder.transform(test) if test else None

    def raw(self, train, test=None):
        """
        This method serves as a transformation no-op; the data is returned
        as-is.
        """
        return train, test

    def robustscaler(self, train, test=None):
        """
        This method serves as a simple wrapper for scikit-learn's RobustScaler.

        :param train: the training data to tranform
        :type train: pandas dataframe
        :param test: the testing data to transform
        :type test: pandas dataframe
        :return: transformed data
        :rtype: tuple of pandas dataframes
        """
        print(f"Applying robust scaling to data of shape {train.shape}...")
        scaler = sklearn.preprocessing.RobustScaler()
        train = scaler.fit_transform(train)
        return train, scaler.transform(test) if test else None

    def standardscaler(self, train, test=None):
        """
        This method serves as a simple wrapper for scikit-learn's
        StandardScaler.

        :param train: the training data to tranform
        :type train: pandas dataframe
        :param test: the testing data to transform
        :type test: pandas dataframe
        :return: transformed data
        :rtype: tuple of pandas dataframes
        """
        print(f"Applying standard scaling to data of shape {train.shape}...")
        scaler = sklearn.preprocessing.RobustScaler()
        train = scaler.fit_transform(train)
        return train, scaler.transform(test) if test else None


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
