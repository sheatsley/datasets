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

    encoder = ColumnTransformer(
        [("", OneHotEncoder(), features)], remainder="passthrough"
    )
    if isinstance(x, list):
        scaler.fit(x[0])
        return scaler.transform(x[0]), scaler.transform(x[1])
    else:
        return scaler.fit_transform(x)
