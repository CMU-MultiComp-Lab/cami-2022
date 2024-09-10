"""Handles the GLM moderation analysis."""

import itertools
import os

import statsmodels.formula.api as smf

import analysis.session_features
import analysis.statistics

from utils.log import log, debug
from utils.plotting import plot_pred_actual

FIG_PATH = os.path.join('_results', 'glm')

P_THRESHOLD = 0.1

SCORES = ['neg', 'pos']
LIWC = ['cogproc', 'affect', 'power', 'time', 'percept']
REPAIR = ['edit', 'restart', 'repeat']


def report_fp(model_name, result):
    """Prints to log the F-score report for a given model.

    Args:
        model_name (str): The name of the model to print to log.
        result (statsmodels.regression.linear_model.RegressionResults):
            The result object of the model.

    """
    log('\t{}: F({}, {}) = {:.4} (p = {:.4})'
        .format(model_name,
                int(result.df_resid), int(result.df_model),
                result.fvalue, result.f_pvalue))


def run_glm(data, liwc, repair, score):
    """Runs a moderation analysis of "repair ~ score * liwc + score + liwc".

    Args:
        data (pd.DataFrame): The data to run analysis on, one session per row.
        liwc (str): The LIWC feature in question.
        repair (str): The self-repair feature in question.
        score (str): The clinical score in question.

    Returns:
        formula (str): A representation of the model form.
        result (statsmodels.regression.linear_model.RegressionResults):
            The result object of the model.

    """

    X = data[[liwc, score]].copy()
    y = data[repair].copy()

    formula = '{y} ~ {x} * {z}'.format(x=score, y=repair, z=liwc)
    red_formula = '{y} ~ {x} + {z}'.format(x=score, y=repair, z=liwc)

    result = smf.ols(formula=formula, data=data).fit()
    red_result = smf.ols(formula=red_formula, data=data).fit()

    _, p_value, _ = result.compare_f_test(red_result)
    if p_value < P_THRESHOLD:

        log('Significant interaction effect: {} (p = {:.4})'
            .format(formula, p_value))
        report_fp('Reduced model', red_result)
        report_fp('Interaction model', result)

        filepath = os.path.join(FIG_PATH, 'fit_{}_{}_{}.png'.format(score,
                                                                    liwc,
                                                                    repair))
        plot_pred_actual(result.predict(X), y, filepath)

    else:
        debug('No significant interaction effects: {} (p = {:.4})'
              .format(formula, p_value))

    return formula, result


def glms(transcripts):
    """Runs the GLM moderation analyses.

    Args:
        transcripts (list of utils.reader.harvard.Transcript): The list of
            Transcript files representing the sessions.

    """

    log('Running GLMS...')

    data = analysis.session_features.get_session_data(transcripts)
    for score in SCORES:
        for liwc, repair in itertools.product(LIWC, REPAIR):
            run_glm(data, liwc, repair, score)
