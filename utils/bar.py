"""Utility for generating consistent progress bars throughout the codebase."""\

import progressbar


def progress_bar(max_val):
    """Generates a ProgressBar object for use in long processes."""
    return progressbar.ProgressBar(maxval=max_val,
                                   widgets=[progressbar.Bar('=', '[', ']'),
                                            ' ',
                                            progressbar.Percentage()])
