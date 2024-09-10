"""Handles plotting of figures for paper presentation."""
import os

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from utils.log import debug
from utils.misc import assert_path

sns.set_style("ticks")
sns.set_context("paper")
sns.set_palette("muted")

# df column - presentation name
FIG_LABELS = {'time': 'Relative Words',
              'percept': 'Percept. Proc. Words',
              'power': 'Power Words',
              'cogproc': 'Cog. Proc. Words',
              'affect': 'Affect Words',
              'pos': 'Positive Score',
              'neg': 'Negative Score',
              'ppl': 'Perplexity',
              'madrs': 'MADRS Score',
              'restart': 'Restarts (per word)',
              'edit': 'Edits (per word)',
              'repeat': 'Repeats (per word)'}


def plot_panss_dist(scores, filepath):
    """Saves a plot of the distribution of PANSS scores.

    Args:
        scores ()
        filepath (str): The filename to save the plot under.
    """
    assert_path(os.path.dirname(filepath))

    fig = plt.figure(figsize=[3.5, 2])
    ax = fig.add_subplot(111)

    bins = np.arange(7, 49, 3)
    for score in ['neg', 'pos']:
        sns.distplot(scores[score], label=FIG_LABELS[score],
                     bins=bins, ax=ax,
                     hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=2))

    ax.set_xlim([7, 49])

    plt.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    sns.despine(offset=3)
    ax.tick_params(left=False, labelleft=False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    debug('Generated plot {}.'.format(filepath))
    plt.savefig(filepath + '.pdf')
    plt.savefig(filepath + '.pgf')


def plot_feature_score(variables, labels, filepath):
    """Saves a plot of features versus scores.

    Args:
        variables (list of list of float): A list of length two indicating the
            X and Y variables to plot.
        labels (list of str): A list of length two indicating the names of the
            X and Y axes.
        filepath (str): The filename to save the plot under.

    """
    assert_path(os.path.dirname(filepath))

    fig = plt.figure(figsize=[1.8, 1.8])
    ax = fig.add_subplot(111)

    sns.regplot(x=np.array(variables[0]),
                y=np.array(variables[1]),
                scatter_kws={'s': 10}, ax=ax)

    plt.xlabel(FIG_LABELS[labels[0]])
    plt.ylabel(FIG_LABELS[labels[1]])

    ax.set_ylim([0, ax.get_ylim()[1]])

    ax.set_aspect(abs(ax.get_xlim()[1] - ax.get_xlim()[0]) /
                  abs(ax.get_ylim()[1] - ax.get_ylim()[0]))

    ax.set_yticklabels(['{:3.0f}%'.format(y * 100)
                        for y in ax.get_yticks()])

    plt.tight_layout()

    debug('Generated plot {}.'.format(filepath))
    plt.savefig(filepath + '.pdf')
    plt.savefig(filepath + '.pgf')
    plt.close()


def plot_pred_actual(pred, actual, filepath):
    """Saves a plot of predicted versus actual values.

    Args:
        pred (pd.Series): The list of predicted values.
        actual (pd.Series): The list of actual values.
        filepath (str): The filename to save the plot under.

    """
    assert_path(os.path.dirname(filepath))

    fig, ax = plt.subplots()
    sns.regplot(pred, actual, fit_reg=False, ax=ax)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    ax.set_aspect('equal')

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    debug('Generated plot {}.'.format(filepath))
    fig.savefig(filepath)
    plt.close()
