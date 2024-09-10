"""Handles various statistical procedures."""

from utils.log import debug

P_THRESHOLD = 0.05


def standardize_dataframe(df):
    return (df - df.mean()) / df.std()


def bh_correction(tests, threshold=P_THRESHOLD):
    """Performs the Benjamini-Hochberg correction for a set of p-values.

    Args:
        dict: A dictionary of the form {testname (str): pvalue (float)}

    Returns:
        list of str: A list of testnames indicating which p-values hold under
            the Benjamini-Hochberg correction.

    """
    num_tests = len(tests)
    pvalue_dict = {v: k for k, v in tests.iteritems()}
    sorted_pvalues = sorted(tests.values())

    top_rank = 0
    for rank in reversed(xrange(num_tests)):
        pvalue = sorted_pvalues[rank - 1]
        critical = ((float(rank) + 1) / num_tests) * threshold
        if pvalue < critical:
            debug('Top significant rank is {}.'.format(rank))
            top_rank = rank
            break

    results = []
    for pvalue in sorted_pvalues[:top_rank]:
        results.append(pvalue_dict[pvalue])
    return results
