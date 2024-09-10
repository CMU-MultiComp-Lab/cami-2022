"""Handles generation of session-wide features."""

import collections

import numpy as np
import pandas as pd

from utils.log import debug
from utils.misc import merge_dictionaries

# a list of all liwc features for reference
LIWC_FEATURES = ['function', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe',
                 'they', 'ipron', 'article', 'prep', 'auxverb', 'adverb',
                 'conj', 'negate', 'verb', 'adj', 'compare', 'interrog',
                 'number', 'quant', 'affect', 'posemo', 'negemo', 'anx',
                 'anger', 'sad', 'social', 'family', 'friend', 'female',
                 'male', 'cogproc', 'insight', 'cause', 'discrep', 'tentat',
                 'certain', 'differ', 'percept', 'see', 'hear', 'feel', 'bio',
                 'body', 'health', 'sexual', 'ingest', 'drives', 'affiliation',
                 'achieve', 'power', 'reward', 'risk', 'focuspast',
                 'focuspresent', 'focusfuture', 'relativ', 'motion', 'space',
                 'time', 'work', 'leisure', 'home', 'money', 'relig', 'death',
                 'informal', 'swear', 'netspeak', 'assent', 'nonflu', 'filler']

# the liwc features of interest
LIWC_FEATURES = ['cogproc', 'affect', 'power',
                 'time', 'percept']


def get_session_repairs(transcripts):
    """Generates all session-wide repair scores for the sessions.

    Args:
        transcripts (list of utils.reader.harvard.Transcript): A list of
            transcript files in the corpus.

    Returns:
        dict: A dictionary of the form {feature (str): scores (list of float)}
            containing the self-repair features of each session.

    """
    debug('Generating session-wide repair scores.')
    repairs = dict()
    for repair in ('edit', 'repeat', 'restart'):
        repairs[repair] = [float(transcript.data[repair].sum()) /
                           transcript.data['WC'].sum()
                           for transcript in transcripts]
    return repairs


def get_session_perplexity(transcripts):
    """Generates all session-wide perplexity scores for the sessions.

    Args:
        transcripts (list of utils.reader.harvard.Transcript): A list of
            transcript files in the corpus.

    Returns:
        dict: A dictionary of the form {feature (str): scores (list of float)}
            containing the perplexity features of each session.

    """
    debug('Generating session-wide perplexity scores.')
    ppl = {'perplexity': []}
    for transcript in transcripts:
        ppl['perplexity'].append(
            np.exp2(np.mean(np.log2(transcript.data['perplexity']))))
    return ppl


def get_scores(transcripts):
    """Returns a dictionary with scores for the given transcripts.

    Args:
        transcripts (list of utils.reader.harvard.Transcript): A list of
            transcript files in the corpus.

    Returns:
        dict: A dictionary of the form {scale (str): scores (list of float)}
            containing the list of clinical scores in the corpus.

    """
    debug('Generating score lists.')
    scores = collections.defaultdict(list)
    for transcript in transcripts:
        for scale, score in transcript.scores.items():
            scores[scale].append(score)
    return scores


def get_session_liwc(transcripts):
    """Generates all session-wide LIWC scores for the sessions.

    Args:
        transcripts (list of utils.reader.harvard.Transcript): A list of
            transcript files in the corpus.

    Returns:
        dict: A dictionary of the form {feature (str): scores (list of float)}
            containing the LIWC features of each session.

    """
    debug('Generating session-wide LIWC scores.')
    features = dict()
    for liwc in LIWC_FEATURES:
        features[liwc] = [sum(transcript.data[liwc] / 100. *
                              transcript.data['WC']) /
                          transcript.data['WC'].sum()
                          for transcript in transcripts]
    return features


def get_session_data(transcripts):
    """Generates a DataFrame of all session-wide data.

    Args:
        transcripts (list of utils.reader.harvard.Transcript): A list of
            transcript files in the corpus.

    Returns:
        pd.DataFrame: A DataFrame containing all session-wide features.

    """

    score_data = get_scores(transcripts)
    liwc_data = get_session_liwc(transcripts)
    repair_data = get_session_repairs(transcripts)
    ppl_data = get_session_perplexity(transcripts)

    data = merge_dictionaries([score_data, liwc_data, repair_data, ppl_data])
    data = pd.DataFrame.from_dict(data)

    return data
