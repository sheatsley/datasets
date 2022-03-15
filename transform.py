"""
The transform module applies transformations to machine learning datasets.
Author: Ryan Sheatsley
Tue Feb 15 2022
"""
import bisect  # Array bisection algorithm
import itertools  # Functions creating iterators for efficient looping
import sklearn.preprocessing  # Preprocessing and Normalization
from utilities import print  # Timestamped printing

# TODO
# destupefy should cleanse for unknown values: min(drop_samples, drop_columns)


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
    :func:`destupefy`: automagic data cleaning (experimental)
    :func:`labelencoder`: encode target labels between 0 and n_classses-1
    :func:`minmaxscaler`: scale features to a given range
    :func:`onehotencoder`: encode categorical features as one-hot arrays
    :func:`raw`: no-op (return data unchanged)
    :func:`robustscaler`: scale features with statistics robust to outliers
    :func:`standardscaler`: standardize features to zero mean and unit variance
    """

    def __init__(self, features, labels, schemes):
        """
        This function initializes Transformer objects with the necessary
        information to apply arbitrary transformations to data. Specifically,
        the manipulated features are expected to be tuple of tuples, wherein
        the number of datasets is the product of the lengths of the tuples.
        This enables, for example, creation of multiple datasets with different
        feature transformation schemes (described by the first tuple), all with
        label encoding (described by the second tuple). Consequently, the
        transformations applied to the features within each tuple is described
        by schemes, which is expected to be tuple of length equal to the number
        of tuples (in features) containing Transformer method callables.

        :param features: the features to manipulate
        :type features: list of lists containing strings or indicies
        :param labels: transfomrations to apply to the labels
        :type labels: tuple of tuples of Transformer callables
        :param schemes: transformations to apply to the data
        :type schemes: tuple of tuples of Transformer callables
        :return: a prepped transformer
        :rtype: Transformer object
        """
        self.features = features
        self.schemes = schemes
        self.labels = labels

        # instantiate transformers to support transforms for test set
        self.le = sklearn.preprocessing.LabelEncoder()
        self.ohe = sklearn.preprocessing.OneHotEncoder(sparse=False)
        self.mms = sklearn.preprocessing.MinMaxScaler()
        self.rs = sklearn.preprocessing.RobustScaler()
        self.ss = sklearn.preprocessing.StandardScaler()
        return None

    def apply(self, data, labels, fit=True):
        """
        This method applies sckilit-learn data transformations, while
        preserving the original layout of the data. Importantly, this method
        applies the transformations and stores them internally. Once the export
        method is called, the transformations are retrieved, concatenated into
        their original indicies, and returned (and thus, this method returns
        nothing).

        :param data: the dataset to transform
        :type data: pandas dataframe
        :param labels: labels for the training data
        :type labels: pandas series
        :param fit: whether transformers should fit (or just transform) the data
        :type fit: bool
        :return: None
        :rtype: NoneType
        """

        # drop transformations from prior parts
        self.data_transforms = []
        self.label_transforms = []

        # save feature names for export later & pass untouched features to raw
        self.feature_names = data.columns.tolist() if fit else self.feature_names
        untransformed_feat = list(set(self.feature_names).difference(*self.features))
        if untransformed_feat:
            self.features.append(untransformed_feat)
            self.schemes.append(Transformer.raw)
            print(
                "The following features will be passed through:",
                ", ".join(untransformed_feat),
            )

        # fit to all parts except test set, transform everything
        for scheme, feature in zip(self.schemes, self.features):
            print(
                f"Applying {', '.join([s.__name__ for s in scheme])}",
                f"to features: {', '.join(feature)}...",
            )
            self.data_transforms.append(
                [s.__get__(self)(data[feature], fit) for s in scheme]
            )

        # apply label transformations
        for label in self.labels:
            print(f"Applying {label.__name__} to labels...")
            self.label_transforms.append(label.__get__(self)(labels, fit))
        return None

    def export(self, feature_names=None):
        """
        This method properly assembles datasets based on the applied
        transformations. Specifically, it concatenates transformations to
        preserve their original orders and produces n copies of the dataset,
        where n is the product of the lengths of the tuples in
        self.data_transforms. It takes no arguments so that the building
        operation (which will be memory-intensive) can be called when it is
        most appropriate (other than feature_names, which can optionally set
        the column headers, of which a subset are modified if one-hot encoding
        was used).

        :param feature_names: names of the features
        :type feature_names: list of strings
        :return: the transformed datasets
        :rtype: generator of tuples of pandas dataframes
        """
        for features, schemes, transforms in zip(
            itertools.repeat(self.features),
            itertools.product(*self.schemes),
            itertools.product(*self.data_transforms),
        ):
            # assemble the dataframe based on the original indicies
            print(f"Exporting {'×'.join([s.__name__ for s in schemes])}...")
            train_transform, test_transform = zip(*transforms)

            # correct feature indicies if this export uses one-hot encoding
            try:
                ohot_idx = schemes.index(Transformer.onehotencoder)
                ohotf = features[ohot_idx]
                offsets = list(itertools.accumulate(len(f) - 1 for f in self.ohe_f))

                # offset non-onehot features, else expand them
                for idx, f_set in enumerate(features):
                    features[idx] = (
                        [
                            f + ([0] + offsets + [0])[bisect.bisect(ohotf, f)]
                            for f in f_set
                        ]
                        if idx != ohot_idx
                        else [
                            range(
                                f + (offsets[idx - 1] if idx != 0 else 0),
                                f + offsets[idx] + 1,
                            )
                            for idx, f in enumerate(f_set)
                        ]
                    )

            except ValueError:
                pass
            training = self.original["train_data"].assign(
                **{
                    str(f): t.T[idx]
                    for idx, (f_set, t) in enumerate(zip(features, train_transform))
                    for f in f_set
                }
            )
            testing = (
                self.original["test_data"].assign(
                    **{
                        str(f): t.T[idx]
                        for idx, (f_tup, t) in enumerate(zip(features, test_transform))
                        for f in f_tup
                    }
                )
                if test_transform
                else None
            )

            # set feature_names (and correct for one-hot encodings)
            if feature_names is not None:
                try:
                    ohotf = features[schemes.index(Transformer.onehotencoder)]
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
                f"{'×'.join([s.__name__ for s in schemes])} export complete!",
                f"Final shape(s): {training.shape}",
                f"{testing.shape}" if testing is not None else "",
            )

            # yield with each label transformation
            for train_labels, test_labels in self.label_transforms:
                yield training, train_labels, testing, test_labels

    def destupefy(self, train, test=None):
        """
        This method attempts to clean datasets after they have been processed.
        At this time, destupefy performs the following:

            - Removes any single-value columns
            - Removes any identical rows

        Note, this method is experimental.

        :param dataset: the dataset to cleanse
        :type dataset: pandas dataframe
        :return: the cleaned datasets
        :rtype: pandas dataframe
        """

        # doing this indepedently is unsafe; for now, fit to training data only
        print("Analyzing dataset for single-value columns...")
        tr_os, tr_of = train.shape
        _, te_of = test.shape if test is not None else None, None
        single_cols = train.columns[train.nunique() > 1]
        train = train[single_cols]
        test = test[single_cols] if test is not None else None

        # dropping duplicates can be done safely for both traing and testing
        print(f"Dropped {train.shape[1]-tr_os} features! Removing duplicate samples...")
        train = train.drop_duplicates()
        print(f"Dropped {train.shape[0]-tr_of} samples! (New shape: {train.shape})")
        test = test.drop_duplicates() if test is not None else None
        print(
            f"Dropped {test.shape[0]-te_of} test samples! (New shape: {test.shape})"
        ) if test is not None else None
        return train, test

    def labelencoder(self, labels, fit):
        """
        This method serves as a simple wrapper for scikit-learn's LabelEncoder.

        :param labels: the labels to transform
        :type labels: pandas series
        :param fit: whether the transformer should fit before transforming
        :type fit: bool
        :return: integer-encoded labels
        :rtype: numpy array
        """
        print(f"Applying label encoding to {labels.size} samples...")
        data = self.le.fit_transform(labels) if fit else self.le.transform(labels)
        print(f"Transformed {', '.join(self.le.classes_)} to integers.")
        return data

    def minmaxscaler(self, data, fit):
        """
        This method serves as a simple wrapper for scikit-learn's MinMaxScaler.

        :param data: the data to transform
        :type data: pandas dataframe
        :param fit: whether the transformer should fit before transforming
        :type fit: bool
        :return: transformed data
        :rtype: numpy array
        """
        print(f"Applying min-max scaling to data of shape {data.shape}...")
        return self.mms.fit_transform(data) if fit else self.mms.transform(data)

    def onehotencoder(self, data, fit):
        """
        This method serves as a simple wrapper for scikit-learn's
        OneHotEncoder. Importantly, this also alters the class attribute
        feature_names so that it is correct when exporting.

        :param data: the data to transform
        :type data: pandas dataframe
        :param fit: whether the transformer should fit before transforming
        :type fit: bool
        :return: transformed data
        :rtype: numpy array
        """
        print(f"Applying one-hot encoding to data of shape {data.shape}...")
        data = self.ohe.fit_transform(data) if fit else self.ohe.transform(data)
        print(f"Encoding shape expanded to {data.shape}.")

        # ensure feature_names reflects the expanded space
        for idx, ohot_feat in enumerate(self.ohe.feature_names_in_):
            org_idx = self.feature_names.index(ohot_feat)
            self.feature_names[org_idx : org_idx + 1] = self.ohe.categories_[idx]
        return data

    def raw(self, data, fit):
        """
        This method serves as a transformation no-op; the data is returned
        as-is.

        :param data: the data to transform
        :type data: pandas dataframe
        :param fit: whether the transformer should fit before transforming
        :type fit: bool
        :return: transformed data
        :rtype: numpy array
        """
        print(f"Apply raw scaling (no-op) to data of shape {data.shape}...")
        return data

    def robustscaler(self, data, fit):
        """
        This method serves as a simple wrapper for scikit-learn's RobustScaler.

        :param data: the data to transform
        :type data: pandas dataframe
        :param fit: whether the transformer should fit before transforming
        :type fit: bool
        :return: transformed data
        :rtype: numpy array
        """
        print(f"Applying robust scaling to data of shape {data.shape}...")
        return self.rs.fit_transform(data) if fit else self.rs.transform(data)

    def standardscaler(self, data, fit):
        """
        This method serves as a simple wrapper for scikit-learn's
        StandardScaler.

        :param data: the data to transform
        :type data: pandas dataframe
        :param fit: whether the transformer should fit before transforming
        :type fit: bool
        :return: transformed data
        :rtype: numpy array
        """
        print(f"Applying standard scaling to data of shape {data.shape}...")
        return self.ss.fit_transform(data) if fit else self.ss.transform(data)


if __name__ == "__main__":
    """ """
    raise SystemExit(0)
