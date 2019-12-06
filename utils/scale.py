""" 

This module contains different attribute scaling techniques. At present, these
functions are simply scikit-learn wrappers. These functions expect scaling
attributes of one dataset wrt another (e.g., testing wrt training) to be passed
in as a list.

"""


def raw(x, features, **kwargs):
    """
    Simply returns the dataset itself
    """
    return x


def normalization(x, features, **kwargs):
    """
    Normalizes attributes so that mean(X) = 0

    Defined as: 
        x' = (x - mean(x))/(max(x) - min(x))
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    scaler = ColumnTransformer(
        [("", StandardScaler(with_std=False), features)], n_jobs=-1
    )
    if isinstance(x, list):
        scaler.fit(x[0])
        x[0][:, features], x[1][:, features] = [
            scaler.transform(x[0]),
            scaler.transform(x[1]),
        ]
    else:
        x[:, features] = scaler.fit_transform(x)
    return x


def rescale(x, features, minimum=0, maximum=1, **kwargs):
    """
    Rescales attributes to range [minimum, maximum]

    Defined as: 
        x' = (x - min(x))/(max(x) - min(x))
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import MinMaxScaler

    scaler = ColumnTransformer(
        [("", MinMaxScaler(feature_range=(minimum, maximum)), features)], n_jobs=-1
    )
    if isinstance(x, list):
        scaler.fit(x[0])
        x[0][:, features], x[1][:, features] = [
            scaler.transform(x[0]),
            scaler.transform(x[1]),
        ]
    else:
        x[:, features] = scaler.fit_transform(x)
    return x


def robust_scale(
    x, features, minimum=25.0, maximum=75.0, center=True, scale=True, **kwargs
):
    """
    Normalize/standardize attributes according to the interquartile range

    Defined as one of: 
        1) x' = (x - mean(x))/(max(x) - min(x)) (scale == False)
        2) x' = (x - mean(x))/std(x)
    where mean(x), max(x), min(x), and std(x) are subject to:
        minimum quantile < x < maximum quantile 
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import RobustScaler

    scaler = ColumnTransformer(
        [
            (
                "",
                RobustScaler(
                    quantile_range=(minimum, maximum),
                    with_centering=center,
                    with_scaling=scale,
                ),
                features,
            )
        ],
        n_jobs=-1,
    )
    if isinstance(x, list):
        scaler.fit(x[0])
        x[0][:, features], x[1][:, features] = [
            scaler.transform(x[0]),
            scaler.transform(x[1]),
        ]
    else:
        x[:, features] = scaler.fit_transform(x)
    return x


def standardization(x, features, **kwargs):
    """
    Normalizes attributes so that mean(X) = 0 and std(x) = 1

    Defined as: 
        x' = (x - mean(x))/std(x)
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    scaler = ColumnTransformer([("", StandardScaler(), features)], n_jobs=-1)
    if isinstance(x, list):
        scaler.fit(x[0])
        x[0][:, features], x[1][:, features] = [
            scaler.transform(x[0]),
            scaler.transform(x[1]),
        ]
    else:
        x[:, features] = scaler.fit_transform(x)
    return x


def unit_norm(x, features, p="l1", single="rescale", **kwargs):
    """
    Scales attributes so that ||x||_p = 1

    Defined as: 
        x' = x / ||x||_p
    
    Feature groups that only contain one feature are
    instead scaled with [single] parameter
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import Normalizer

    # if this isn't defined, then unit_norm probably won't make sense
    if "norm" not in kwargs:
        kwargs["norm"] = [features]

    # for homogenous code, overwrite definition of features (if appropriate)
    for features in kwargs["norm"]:

        # check the norm ranges; use l_p norm if it is a range
        if len(features) > 1:
            scaler = ColumnTransformer([("", Normalizer(norm=p), features)], n_jobs=-1)
            if isinstance(x, list):
                scaler.fit(x[0])
                x[0][:, features], x[1][:, features] = [
                    scaler.transform(x[0]),
                    scaler.transform(x[1]),
                ]
            else:
                x[:, features] = scaler.fit_transform(x)

        # otherwise, use scheme defined by [single]
        else:
            x = globals()[single](x, features, **kwargs)
    return x
