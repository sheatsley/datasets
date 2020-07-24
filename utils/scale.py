""" 
This module contains different attribute scaling techniques. At present, these
functions are simply scikit-learn wrappers. These functions expect scaling
attributes of one dataset wrt another (e.g., testing wrt training) to be passed
in as a list.
"""


def raw(x, features, test, **kwargs):
    """
    Simply returns the dataset itself
    """
    return x


def normalization(x, features, test, **kwargs):
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
    scaler.fit(x[: kwargs["size"]] if test else x)
    x[:, features] = scaler.transform(x)
    return x


def rescale(x, features, test, minimum=0, maximum=1, **kwargs):
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
    scaler.fit(x[: kwargs["size"]] if test else x)
    x[:, features] = scaler.transform(x)
    return x


def robust_scale(
    x, features, test, minimum=25.0, maximum=75.0, center=True, scale=True, **kwargs
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
    scaler.fit(x[: kwargs["size"]] if test else x)
    x[:, features] = scaler.transform(x)
    return x


def standardization(x, features, test, **kwargs):
    """
    Normalizes attributes so that mean(X) = 0 and std(x) = 1

    Defined as:
        x' = (x - mean(x))/std(x)
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    scaler = ColumnTransformer([("", StandardScaler(), features)], n_jobs=-1)
    scaler.fit(x[: kwargs["size"]] if test else x)
    x[:, features] = scaler.transform(x)
    return x


def unit_norm(x, features, p="l1", other="rescale", **kwargs):
    """
    Scales attributes so that ||x||_p = 1

    Defined as:
        x' = x / ||x||_p

    Attribtues that are not included are instead scaled via [other]
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import Normalizer

    # if no features are specified, scale all according to [other]
    try:
        scaler = ColumnTransformer(
            [("", Normalizer(norm=p), list(kwargs["norm"]))], n_jobs=-1
        )
        x[:, kwargs["norm"]] = scaler.fit_transform(x)

        # scale other attributes via [other]
        return globals()[other](
            x,
            tuple(
                (feature for feature in features if feature not in set(kwargs["norm"]))
            ),
            **kwargs
        )
    except:
        return globals()[other](x, features, **kwargs)
