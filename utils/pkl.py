"""Handles pickling of files."""

import cPickle as pickle
import os

from utils.log import log


def pkl(data, filepath):
    """Saves data to a file."""
    log('Pickling to {}.'.format(filepath))
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, 'w') as outfile:
        pickle.dump(data, outfile)


def check_pkl(filepath):
    """Loads data from a file."""
    if os.path.exists(filepath):
        log('Loading from {}.'.format(filepath))
        with open(filepath) as infile:
            return pickle.load(infile)
