"""
The transform module applies transformations to machine learning datasets.
Author: Ryan Sheatsley
Tue Feb 15 2022
"""
import itertools  # Functions creating iterators for efficient looping
import pandas  # Python Data Analysis Library
import sklearn.preprocessing  # Preprocessing and Normalization
from utilities import print  # Timestamped printing


class Destupefier(sklearn.base.TransformerMixin):
    """
    The Destupefier is an (experimental) stateful transformer, such as those in
    sklearn.preprocessing module. After dataframes have been assembled, this
    class "cleans" datasets by attempting to identifiy data deficiencies
    through a series of checks. At this time, the following is performed (in
    this order):

        (1) Any unknown values are removed (max of columns and rows are kept).
        (2) Any identical columns are removed (first occurance is kept).
        (3) Any identical rows are removed (first occurance is kept).
        (4) Any single-value columns are removed.

    Notably, we subclass sklearn's TransformerMixin (as opposed to using
    FunctionTransformer directly) for state: if we find, for example, in a
    training set that a feature has a single value, then that feature will be
    removed, and thus, the same transformation needs to be applied to a test
    set (if provided), regardless if the test set has more than one value for
    that feature (a rather pathelogical scenario).

    Moreover, we do not define a fit method as there are a suite of scenarios
    where removing columns or rows (based on invalid values, duplicity, etc.)
    can cause other nonideal scenarios to reappear (such as removing a
    duplicate row could cause unique columns to become duplicates). Thus,
    simply identifying definciencies here offers little utility since some can
    arise as result of addressing others, and so, this method does nothing.

    :func:`__init__`: instantiates Destupifier objects
    :func:`transform`: identifies & corrects deficiencies
    """

    def __init__(self, unknowns=set("")):
        """
        This method instantiates Destupefier transformer objects. It accepts a
        single argument (unknowns), which defines the set of *additional* NA
        values beyond None and NAN (i.e., data entries that are invalid,
        unfilled, errornous, etc.), as Pandas already recognizes None and NAN
        to be invalid values. Such values are removed from the dataframe on
        transform.

        :param unknowns: additional values to be considered invalid
        :type unknowns: set of miscellaneous datatypes
        :return: a Destupefier transformer
        :rtype: Destupefier object
        """
        self.unknowns = unknowns
        return None

    def transform(self, dataset, labels, fit=True):
        """
        This method identifies the deficiencies above and corrects them.
        Importantly, some cleaning procedures (such as removing single-value or
        duplicate features) should be: (1) identically cleaned on the test set,
        and (2) should not be identified on the test set (to ensure both the
        training and test sets are homogenous in the feautre space). Due to the
        fact that some procedures can reveal new deficiencies (described in
        fit), we must explicitly pass if we are fitting within this method.

        :param dataset: the dataset to clean
        :type dataset: pandas dataframe
        :param labels: associated labels
        :type labels: pandas series
        :return: cleaned dataset and labels
        :rtype: tuple containing a pandas dataframe and numpy array
        """

        # only identify columns to remove in non-test partitions
        org_rows, org_cols = dataset.shape
        if fit:

            # test 1: removing invalid values from min(rows, cols)
            print(f"Scanning {dataset.size} values for invalids...")
            dataset.replace(self.unknowns, None, inplace=True)
            na_locs = dataset.isna()
            na_rows = na_locs.any(axis=1).sum() / org_rows
            na_cols = na_locs.any(axis=0).sum() / org_cols
            print(
                f"{na_rows:0.1%} of samples and {na_cols:0.1%}",
                "of features have invalid values. Dropping the minimum...",
            )
            na_col_names = dataset.columns[na_locs.any()]
            dataset.dropna(axis=1 if na_cols < na_rows else 0, inplace=True)
            self.rm_features = set(na_col_names if na_cols < na_rows else [])

            # test 2: identical column removal
            print(f"Scanning {len(dataset.columns)} features for duplicates...")
            dup_features = dataset.columns[dataset.T.duplicated()]
            print(f"Dropping {len(dup_features)} duplicate features...")
            dataset.drop(columns=dup_features, inplace=True)
            self.rm_features = self.rm_features.union(dup_features)
        else:

            # test 1, 2 & 4: drop features dropped in non-test partitions
            print(f"Dropping {len(self.rm_features)} features from test set...")
            dataset.drop(columns=self.rm_features, inplace=True)

            # test ≈1: remove invalid values from rows only for test set
            print(f"Scanning {dataset.size} values for invalids...")
            dataset.replace(self.unknowns, None, inplace=True)
            na_rows = dataset.isna().any(axis=1).sum()
            print(f"Dropping {na_rows} samples with invalid values...")
            dataset.dropna(axis=0, inplace=True)

        # test 3: identical row removal (also remove from labels)
        print(f"Scanning {len(dataset)} samples for duplicates...")
        dup_samples = dataset.duplicated()
        print(f"Dropping {dup_samples.sum()} duplicate samples...")
        dataset.drop(index=dataset.index[dup_samples], inplace=True)
        dataset.reset_index(drop=True, inplace=True)
        labels.drop(labels.index[dup_samples], inplace=True)
        labels.reset_index(drop=True, inplace=True)
        if fit:

            # test 4: single-value column removal
            print(f"Scanning {len(dataset.columns)} features for single values...")
            single_features = dataset.columns[dataset.nunique() == 1]
            print(f"Dropping {len(single_features)} single-valued features...")
            dataset.drop(columns=single_features, inplace=True)
            self.rm_features = self.rm_features.union(single_features)
        return dataset, labels


class Transformer:
    """
    This Transformer class servs as an intelligent wrapper for scikit-learn's
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
        This method initializes Transformer objects with the necessary
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
        self.ds = Destupefier()
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
        self.feature_names = data.columns.tolist()
        untransformed_feat = list(set(self.feature_names).difference(*self.features))
        if untransformed_feat:
            self.features.append(untransformed_feat)
            self.schemes.append([Transformer.identity])
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

    def export(self):
        """
        This method properly assembles datasets based on the applied
        transformations. Specifically, it concatenates transformations to
        preserve their original orders and produces n copies of the dataset,
        where n is the product of the lengths of the tuples in
        self.data_transforms. It takes no arguments so that the building
        operation (which will be memory-intensive) can be called when it is
        most appropriate.

        :return: the transformed datasets
        :rtype: generator of tuples of pandas dataframes & series
        """
        for feature_list, scheme_list, transform_list in zip(
            itertools.repeat(self.features),
            itertools.product(*self.schemes),
            itertools.product(*self.data_transforms),
        ):

            # assemble the dataframe in the original order
            scheme_names = [s.__name__ for s in scheme_list]
            print(f"Exporting {'×'.join(scheme_names)}...")
            dataset = len(self.feature_names) * [None]
            for features, scheme, transform in zip(
                feature_list, scheme_list, transform_list
            ):

                # check if the scheme expanded the features
                if len(features) != len(transform.T):
                    print(f"{scheme.__name__} expanded features! Correcting...")
                    features = list(itertools.chain(*self.ohe.categories_))

                for feature, values in zip(features, transform.T):
                    dataset[self.feature_names.index(feature)] = values

            # convert to pandas dataframe, set header, print shape, and yield
            print("×".join(scheme_names), "assembled. Creating dataframe...")
            dataset = pandas.DataFrame(list(zip(*dataset)), columns=self.feature_names)
            print(f"Dataframe complete. Final shape: {dataset.shape}")

            # yield with each label transformation (as a series)
            for labels, label_scheme in zip(self.label_transforms, self.labels):
                yield dataset, pandas.Series(labels, name="label"), scheme_names + [
                    label_scheme.__name__
                ]

    def destupefy(self, data, labels, fit):
        """
        This method serves as a simple wrapper for Destupefier objects.

        :param data: the dataset to clean
        :type data: pandas dataframe
        :param labels: associated labels
        :type labels: numpy array
        :return: cleaned dataset and labels
        :rtype: tuple containing a pandas dataframe and numpy array
        """
        print(f"Applying destupefication to data of shape {data.shape}...")
        org_rows, org_cols = data.shape
        data, labels = self.ds.transform(data, labels, fit)
        print(
            "Destupefication complete!",
            f"Dropped {org_rows - len(data)}",
            f"samples and {org_cols - len(data.columns)} features.",
            f"New shape: {data.shape}",
        )
        return data, labels

    def identity(self, data, fit):
        """
        This method serves as an identity function, which is always used for
        features that are passed through (i.e., not assocaited with a scheme).

        :param data: the data to transform
        :type data: pandas dataframe
        :param fit: whether the transformer should fit before transforming
        :type fit: bool
        :return: transformed data
        :rtype: numpy array
        """
        print(f"Applying identity to data of shape {data.shape}...")
        return data.to_numpy()

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
        print(f"Encoded {', '.join(self.le.classes_)} as integers.")
        return data

    def metadata(self):
        """
        This method prepares any metadata extracted during the data
        transformation process. At this time, the following attributes are considered
        metadata:

            (1) feature names (from self.feature_names)
            (2) class mappings (from self.le.classes_)
            (3) one-hot encoding maps (from self.ohe.categories_)

        :return: any relevant metadata
        :rtype: dict of various datatypes
        """
        print("Reading transformer metadata...")
        metadata = {"feature_names": self.feature_names}
        if hasattr(self.le, "classes_"):
            metadata["class_map"] = {
                new: old for new, old in enumerate(self.le.classes_)
            }
        if hasattr(self.ohe, "categories_"):
            metadata["onehot_map"] = {
                feature: values.tolist()
                for feature, values in zip(
                    self.ohe.feature_names_in_, self.ohe.categories_
                )
            }
        return metadata

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

        # if categories are ints or floats, apply feature_names_out
        bounds = list(itertools.accumulate(map(len, self.ohe.categories_)))
        slices = [(start, end) for start, end in zip([0] + bounds, bounds)]
        org_feat = [self.ohe.get_feature_names_out()[slice(*s)] for s in slices]
        for idx, new_feat in enumerate(self.ohe.categories_):
            self.ohe.categories_[idx] = (
                org_feat[idx]
                if new_feat[0].translate({45: "", 46: ""}).isdigit()
                else self.ohe.categories_[idx]
            )

        # ensure feature_names reflects the expanded space
        for idx, ohot_feat in enumerate(self.ohe.feature_names_in_):
            org_idx = self.feature_names.index(ohot_feat)
            self.feature_names[org_idx : org_idx + 1] = self.ohe.categories_[idx]
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
