""" 
This module contains different attribute preprocessing techniques. At present, these
functions are simply scikit-learn wrappers.
"""


def encode_attributes(x, features):
    """
    Encodes categorical features as a one-hot array
    """
    from numpy import concatenate
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    encoder = ColumnTransformer(
        [("", OneHotEncoder(sparse=False), features)], n_jobs=-1
    )
    return concatenate(
        (x[:, : features[0]], encoder.fit_transform(x), x[:, features[-1] + 1 :]),
        axis=1,
    )


def encode_labels(x, features):
    """
    Maps strings to integers
    """
    from numpy import concatenate
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    encoder = ColumnTransformer([("", OrdinalEncoder(), features)], n_jobs=-1)
    x[:, features] = encoder.fit_transform(x)
    return x
