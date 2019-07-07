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
    if isinstance(x, list):
        encoder.fit(concatenate(x))

        # insert encodings at appropriate indicies (TODO: could be more general)
        return [
            concatenate(
                (
                    x[0][:, : features[0]],
                    encoder.transform(x[0]),
                    x[0][:, features[-1] + 1 :],
                ),
                axis=1,
            ),
            concatenate(
                (
                    x[1][:, : features[0]],
                    encoder.transform(x[1]),
                    x[1][:, features[-1] + 1 :],
                ),
                axis=1,
            ),
        ]
    else:
        return concatenate(
            (x[:, : features[0]], encoder.fit_transform(x), x[:, features[-1] + 1 :]),
            axis=1,
        )


def encode_labels(x, features):
    """
    Encodes categorical features as a one-hot array
    """
    from numpy import concatenate
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    encoder = ColumnTransformer([("", OrdinalEncoder(), features)], n_jobs=-1)
    if isinstance(x, list):
        encoder.fit(concatenate(x))
        x[0][:, features] = encoder.transform(x[0])
        x[1][:, features] = encoder.transform(x[1])
    else:
        x[:, features] = encoder.fit_transform(x)
    return x
