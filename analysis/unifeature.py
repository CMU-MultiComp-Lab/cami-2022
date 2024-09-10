"""Various analyses regarding single features."""

import collections
import itertools
import os

import numpy as np

import scipy.stats

import analysis.session_features
import analysis.statistics

from utils.log import log
from utils.plotting import plot_feature_score, plot_panss_dist

LIWC_SCORES_FIG_PATH = os.path.join('_results', 'liwc_scores')
PPL_SCORES_FIG_PATH = os.path.join('_results', 'ppl_scores')
REPAIR_SCORES_FIG_PATH = os.path.join('_results', 'repair_scores')


def test_pos_neg(transcripts):
    """Tests for a correlation between the positive and negative PANSS scores.

    Args:
        transcripts (list of utils.reader.harvard.Transcript): A list of
            transcript files in the corpus.
    """
    scores = analysis.session_features.get_scores(transcripts)
    result = scipy.stats.spearmanr(scores['pos'], scores['neg'])
    log('Correlation between positive and negative scores:')
    log('\trho = {0[0]}, p = {0[1]}'.format(result))
    plot_panss_dist(scores, os.path.join('_results', 'panss'))


def regression_ppl_scores(transcripts):
    """Runs a regression test for perplexity scores.

    Args:
        transcripts (list of utils.reader.harvard.Transcript): A list of
            transcript files in the corpus.

    """
    log('Running Spearman\'s rho correlation test for perplexity scores and '
        'clinical scores.')

    scores = analysis.session_features.get_scores(transcripts)
    ppls = [transcript.data['perplexity'] for transcript in transcripts]
    ppl_scores = [np.exp2(np.mean(np.log2(ppl))) for ppl in ppls]

    for scale in scores:
        result = scipy.stats.spearmanr(scores[scale], ppl_scores)
        log('{} score and perplexity correlation'.format(scale))
        log('\trho = {0[0]}, p = {0[1]}'.format(result))

        plot_feature_score([scores[scale], ppl_scores], [scale, 'ppl'],
                           os.path.join(PPL_SCORES_FIG_PATH,
                                        '{}.pgf'.format(scale)))


def run_regression_set(scores, features, path):
    """Runs a set of regression tests and corrects with the BH procedure.

    Args:
        scores (dict): A dictionary of the form {scale (str): scores (list of
            float)} containing the list of scores in the corpus.
        features (dict): A dictionary of the form {feature (str): scores (list
            of float)} containing the features of each session to compare
            against.
        path (str): The pathname under which to save the resulting figures.

    """

    results = collections.defaultdict(dict)
    pvalues = collections.defaultdict(dict)

    for scale, feature in itertools.product(scores.keys(), features.keys()):
        result, pvalue = scipy.stats.spearmanr(
            scores[scale], features[feature])
        results[scale][feature] = result
        pvalues[scale][feature] = pvalue

    for scale in scores:
        # for test in pvalues[scale]:
        for test in analysis.statistics.bh_correction(pvalues[scale]):
            log('significant hypothesis test: {} score + {}'.format(scale,
                                                                    test))
            log('\trho = {:.3}, p = {:.3}'.format(results[scale][test],
                                                  pvalues[scale][test]))

            plot_feature_score([scores[scale], features[test]],
                               [scale, test],
                               os.path.join(path, '{}_{}'.format(scale, test)))


def regression_liwc_scores(transcripts):
    """Runs a regression analysis between clinical scores and LIWC features.

    Args:
        transcripts (list of utils.reader.harvard.Transcript): A list of
            transcript files in the corpus.

    """
    log('Running Spearman\'s rho correlation tests for LIWC features and '
        'clinical scores, corrected with the Benjamini-Hochberg procedure.')

    scores = analysis.session_features.get_scores(transcripts)
    features = analysis.session_features.get_session_liwc(transcripts)
    run_regression_set(scores, features, LIWC_SCORES_FIG_PATH)


def regression_repair_scores(transcripts):
    """Runs a regression analysis between clinical scores and repair features.

    Args:
        transcripts (list of utils.reader.harvard.Transcript): A list of
            transcript files in the corpus.

    """
    log('Running Spearman\'s rho correlation tests for repair counts and '
        'clinical scores, corrected with the Benjamini-Hochberg procedure.')

    scores = analysis.session_features.get_scores(transcripts)
    repairs = analysis.session_features.get_session_repairs(transcripts)
    run_regression_set(scores, repairs, REPAIR_SCORES_FIG_PATH)

    return
