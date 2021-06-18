"""
The utils module defines functions used throughout mlds.
Author: Ryan Sheatsley
Fri Jun 18 2021
"""
import builtins  # Built-in objects
import time  # Time access and conversions


def print(*args, **kwargs):
    """
    This function wraps the print function, prepended with a timestamp.

    :param *args: positional arguments supported by print()
    :type *args: tuple
    :param **kwargs: keyword arguments supported by print()
    :type **kwargs: dictionary
    :return: None
    :rtype: NoneType
    """
    return builtins.print("[" + time.asctime() + "]", *args, **kwargs)


if __name__ == "__main__":
    """
    This runs some basic unit tests with the functions defined in this module
    """
    print("Test string with implicit date.")
    raise SystemExit(0)
