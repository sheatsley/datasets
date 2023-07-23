"""
This module defines classes for transforming and cleaning datasets.
"""
import itertools

import pandas
import sklearn.preprocessing


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

    def fit(self, dataset, _):
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
        self.deficient_features = set().union(duplicates).union(singles)
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
        print(f"Dropping {len(self.deficient_features)} deficient features...")
        dataset.drop(columns=self.deficient_features, inplace=True)
        print(f"Scanning {len(dataset)} samples for duplicates...")
        duplicates = dataset.duplicated()
        dataset.drop(index=dataset.index[duplicates], inplace=True)
        dataset.reset_index(inplace=True)
        labels.drop(labels.index[duplicates], inplace=True)
        labels.reset_index(inplace=True)
        return dataset, labels


class Transformer:
    """
    The Transformer class augments 
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
    :func:`identity`: no-op (return data unchanged)
    :func:`labelencoder`: encode target labels between 0 and n_classses-1
    :func:`minmaxscaler`: scale each feature to a given range
    :func:`onehotencoder`: encode categorical features as one-hot arrays
    :func:`robustscaler`: scale features with statistics robust to outliers
    :func:`standardscaler`: standardize features to zero mean and unit variance
    :func:`uniformscaler`: scale all features to a given range
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
        self.ohe = sklearn.preprocessing.OneHotEncoder(sparse_output=False)
        self.mms = sklearn.preprocessing.MinMaxScaler()
        self.rs = sklearn.preprocessing.RobustScaler()
        self.ss = sklearn.preprocessing.StandardScaler()
        self.us = sklearn.preprocessing.FunctionTransformer(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        return None

    def apply(self, data, fit, labels):
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

        # save feature names for export later & pass untouched features to identity
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

    def uniformscaler(self, data, fit):
        """
        This method is functionally similar to scikit-learn's MinMaxScaler with
        one exception: scikit-learn's MinMaxScaler scales features
        individually, while this method scales all features by the absolute max
        and minimum observed. This is appropriate when all features are
        semantically the same, such as pixels in images.

        :param data: the data to transform
        :type data: pandas dataframe
        :param fit: whether the transformer should fit before transforming
        :type fit: bool
        :return: transformed data
        :rtype: numpy array
        """
        print(f"Applying uniform min-max scaling to data of shape {data.shape}...")
        data = data.to_numpy()
        return self.us.fit_transform(data) if fit else self.us.transform(data)
