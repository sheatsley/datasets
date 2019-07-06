""" 
This module contains different attribute preprocessing techniques. At present, these
functions are simply scikit-learn wrappers.
"""


def encode(x, features):
    """
    Encodes categorical features as a one-hot array
    """
    from numpy import concatenate
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    encoder = ColumnTransformer(
        [("", OneHotEncoder(sparse=False), features)],
        remainder="passthrough",
        n_jobs=-1,
    )
    if isinstance(x, list):
        encoder.fit(concatenate(x))
        return [encoder.transform(x[0]), encoder.transform(x[1])]
    else:
        return encoder.fit_transform(x)
