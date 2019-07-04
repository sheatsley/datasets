""" 
This module contains different attribute preprocessing techniques. At present, these
functions are simply scikit-learn wrappers.
"""


def encode(x):
    """
    Encodes categorical features as a one-hot array
    """
    from sklearn.preprocessing import OneHotEncoder

    return OneHotEncoder().fit_transform(x)
