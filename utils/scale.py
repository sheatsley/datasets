""" 

This module contains different attribute scaling techniques. At present, these
functions are simply scikit-learn wrappers.

"""


def rescale(x, minimum=0, maximum=1):
    """
    Rescales attributes to range [minimum, maximum]

    Defined as: 
        x' = (x - min(x))/(max(x) - min(x))
    """
    from sklearn.preprocessing import minmax_scale

    return minmax_scale(x, feature_range=(minimum, maximum))


def normalization(x):
    """
    Normalizes attributes so that mean(X) = 0

    Defined as: 
        x' = (x - mean(x))/(max(x) - min(x))
    """
    from sklearn.preprocessing import scale

    return scale(x, with_std=False)


def standardization(x):
    """
    Normalizes attributes so that mean(X) = 0 and std(x) = 1

    Defined as: 
        x' = (x - mean(x))/std(x)
    """
    from sklearn.preprocessing import scale

    return scale(x)


def robust_scale(x, minimum=25.0, maximum=75.0, center=True, scale=True):
    """
    Normalize/standardize attributes according to the interquartile range

    Defined as one of: 
        1) x' = (x - mean(x))/(max(x) - min(x)) (scale == False)
        2) x' = (x - mean(x))/std(x)
    where mean(x), max(x), min(x), and std(x) are subject to:
        minimum quantile < x < maximum quantile 
    """
    return robust_scale(
        x, with_centering=center, with_scaling=scale, quantile_range=(minimum, maximum)
    )


def unit_norm(x, p="l2"):
    """
    Scales attributes so that ||x||_p = 1

    Defined as: 
        x' = x / ||x||_p
    """
    from sklearn.preprocessing import normalize

    return normalize(x, norm=p)
