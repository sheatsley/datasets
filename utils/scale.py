""" 

This module contains different attribute scaling techniques. At present, these
functions are simply scikit-learn wrappers. These functions expect scaling
attributes of one dataset wrt another (e.g., testing wrt training) to be passed
in as a list.

"""


def rescale(x, minimum=0, maximum=1, **kwargs):
    """
    Rescales attributes to range [minimum, maximum]

    Defined as: 
        x' = (x - min(x))/(max(x) - min(x))
    """
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(minimum, maximum))
    if isinstance(x, list):
        scaler.fit(x[0])
        return scaler.transform(x[0]), scaler.transform(x[1])
    else:
        return scaler.fit_transform(x)


def normalization(x, **kwargs):
    """
    Normalizes attributes so that mean(X) = 0

    Defined as: 
        x' = (x - mean(x))/(max(x) - min(x))
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(with_std=False)
    if isinstance(x, list):
        scaler.fit(x[0])
        return scaler.transform(x[0]), scaler.transform(x[1])
    else:
        return scaler.fit_transform(x)


def standardization(x, **kwargs):
    """
    Normalizes attributes so that mean(X) = 0 and std(x) = 1

    Defined as: 
        x' = (x - mean(x))/std(x)
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    if isinstance(x, list):
        scaler.fit(x[0])
        return scaler.transform(x[0]), scaler.transform(x[1])
    else:
        return scaler.fit_transform(x)


def robust_scale(x, minimum=25.0, maximum=75.0, center=True, scale=True, **kwargs):
    """
    Normalize/standardize attributes according to the interquartile range

    Defined as one of: 
        1) x' = (x - mean(x))/(max(x) - min(x)) (scale == False)
        2) x' = (x - mean(x))/std(x)
    where mean(x), max(x), min(x), and std(x) are subject to:
        minimum quantile < x < maximum quantile 
    """
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler(
        quantile_range=(minimum, maximum), with_centering=center, with_scaling=scale
    )
    if isinstance(x, list):
        scaler.fit(x[0])
        return scaler.transform(x[0]), scaler.transform(x[1])
    else:
        return scaler.fit_transform(x)


def unit_norm(x, p="l2", **kwargs):
    """
    Scales attributes so that ||x||_p = 1

    Defined as: 
        x' = x / ||x||_p
    """
    from sklearn.preprocessing import Normalizer

    scaler = Normalizer(norm=p)
    if isinstance(x, list):
        scaler.fit(x[0])
        return scaler.transform(x[0]), scaler.transform(x[1])
    else:
        return scaler.fit_transform(x)
