"""Miscellaneous utiity functions."""

import os

from utils.log import debug


def assert_path(path):
    """Ensures that the given path exists.

    Args:
        path (str): The path to check.

    """
    if not os.path.exists(path):
        debug('Creating path {}.'.format(path))
        os.makedirs(path)
    else:
        debug('Path {} exists.'.format(path))


def merge_dictionaries(dict_list):
    dictionary = dict()
    for d in dict_list:
        dictionary.update(d)
    return dictionary
