"""Implements memoization for decorator."""


def memoize(func):
    """Caches previous results from a function."""
    outputs = {}

    def inner(*args):
        """Wrapped function."""
        if args not in outputs:
            outputs[args] = func(*args)
        return outputs[args]
    return inner
