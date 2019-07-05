""" 
This module contains different attribute preprocessing techniques. At present, these
functions are simply scikit-learn wrappers.
"""


def encode(x, features):
    """
    Encodes categorical features as a one-hot array
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    return ColumnTransformer(
        [("", OneHotEncoder(), features)], remainder="passthrough"
        ).fit_transform(x)
