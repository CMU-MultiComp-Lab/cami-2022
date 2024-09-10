"""Handles construction of the Transcript dataframes with all relevant data
for analysis."""

import functools
import os

import pandas as pd

import deep_disfluency.tagger.deep_tagger

from models.backoff import BackoffLM
from utils.bar import progress_bar
from utils.log import log, warn
from utils.memoize import memoize
from utils.pkl import check_pkl, pkl
from utils.reader.harvard import get_transcripts
from utils.reader.switchboard import get_utterances


# location of various data files
SCORES_FILENAME = os.path.join('_data', 'scores.csv')
LIWC_FILENAME = os.path.join('_data', 'liwc.csv')
MODEL_FILENAME = os.path.join('_pkl', 'backoffLM.pkl')
TRANSCRIPT_FILENAME = os.path.join('_pkl', 'transcripts.pkl')

# configuration for the deep-disfluency self-repair model
DD_MODEL = os.path.join('src', 'deep-disfluency', 'deep_disfluency',
                        'experiments', '021', 'epoch_40')
DD_MODEL_CONFIG = os.path.join('src', 'deep-disfluency', 'deep_disfluency',
                               'experiments', 'experiment_configs.csv')
DD_MODEL_CONFIG_NUM = 21

# LIWC features we don't care about
DROP_LIWC = ['AllPunc', 'Period', 'Comma', 'Colon', 'SemiC',
             'QMark', 'Exclam', 'Dash', 'Quote', 'Apostro',
             'Parenth', 'OtherP']


def add_perplexities(transcripts):
    """Appends perplexity scores to each utterance in the transcripts.

    Args:
        transcripts (list of utils.reader.harvard.Transcript): A list of
            transcript files in the corpus.

    Returns:
        list of utils.reader.harvard.Transcript: A list of transcript files
            with the perplexity features appended as columns in the dataframe.

    """
    model = check_pkl(MODEL_FILENAME)
    if not model:
        model = BackoffLM()
        switchboard = get_utterances()
        model.train(switchboard)
        model.save(MODEL_FILENAME)

    log('Calculating perplexities...')

    num_transcripts = len(transcripts)
    progress = progress_bar(max_val=num_transcripts)
    progress.start()

    for idx, transcript in enumerate(transcripts):
        transcript.data['perplexity'] = \
            transcript.data['text'].apply(model.perplexity)
        progress.update(idx + 1)

    progress.finish()
    log('Perplexities appended for {} sessions.'.format(len(transcripts)))


def add_liwc(transcripts):
    """Appends LIWC features to each utterance in the transcripts.

    Args:
        transcripts (list of utils.reader.harvard.Transcript): A list of
            transcript files in the corpus.

    Returns:
        list of utils.reader.harvard.Transcript: A list of transcript files
            with the LIWC features appended as columns in the DataFrame.

    """
    liwc_all = pd.read_csv(LIWC_FILENAME)
    liwc_all = liwc_all.groupby(['Filename'])
    for transcript in transcripts:
        liwc = [dataframe for idx, dataframe in liwc_all
                if idx.split('.')[0] == transcript.session][0]
        liwc = liwc.drop(['Filename', 'Segment'] + DROP_LIWC, axis=1)
        liwc = liwc.reset_index(drop=True)
        transcript.data = transcript.data.merge(liwc, left_index=True,
                                                right_index=True)
    log('LIWC features appended for {} sessions.'.format(len(transcripts)))


@memoize
def tag_utterance_repairs(tagger, utterance):
    """Returns the counts of types of repair in the utterance.

    Args:
        tagger (deep_disfluency.tagger.deep_tagger.DeepDisfluencyTagger): The
            disfluency tagger to use.
        utterance (str): A string utterance to tag disfluencies within.

    Returns:
        list of str: A list of disfluency tags assigned to the utterance, one
            per word, e.g., ['<f/>', '<e/>', '<rms id="12"/>', ...]. See
            https://github.com/dsg-bielefeld/deep_disfluency.
    """
    for word in utterance.split():
        tagger.tag_new_word(word)

    tags = tagger.output_tags
    tagger.reset()
    return tuple(tags)


@memoize
def count_tags(tag, tags):
    """Counts the occurrences of a particular tag in the given utterance.

    Args:
        tag (str): The tag to count instances, e.g., "<e/>".
        tags (list of str): The set of tags to search for the given tag, e.g.,
            ['<f/>', '<e/>', '<rms id="12"/>', ...]. See
            https://github.com/dsg-bielefeld/deep_disfluency.

    Returns:
        int: The number of occurrences in the set.

    """
    return len([t for t in tags if tag in t])


def add_repairs(transcripts):
    """Appends self-repair scores to each utterance in the transcripts.

    Args:
        transcripts (list of utils.reader.harvard.Transcript): A list of
            transcript files in the corpus.

    Returns:
        list of utils.reader.harvard.Transcript: A list of transcript files
            with the self-repair features appended as columns in the DataFrame.

    """

    log('Initializing deep disfluency tagger...')
    tagger = deep_disfluency.tagger.deep_tagger.DeepDisfluencyTagger(
        config_file=DD_MODEL_CONFIG,
        config_number=DD_MODEL_CONFIG_NUM,
        saved_model_dir=DD_MODEL
    )

    log('Tagging disfluencies...')
    progress = progress_bar(max_val=len(transcripts))
    progress.start()

    for idx, transcript in enumerate(transcripts):

        repairs = transcript.data[['text']].copy()
        repairs['tags'] = repairs.applymap(
            functools.partial(tag_utterance_repairs, tagger))

        repairs['edit'] = repairs['tags'].apply(
            functools.partial(count_tags, '<e'))
        repairs['repeat'] = repairs['tags'].apply(
            functools.partial(count_tags, '<rpnrep'))
        repairs['restart'] = repairs['tags'].apply(
            functools.partial(count_tags, '<rps'))

        repairs = repairs.drop(['text'], axis=1)
        transcript.data = transcript.data.merge(repairs, left_index=True,
                                                right_index=True)

        progress.update(idx + 1)

    progress.finish()


def assign_scores(transcripts):
    """Appends clincal scores to transcripts.

    Args:
        transcripts (list of utils.reader.harvard.Transcript): A list of
            transcript files in the corpus.

    Returns:
        list of utils.reader.harvard.Transcript: A list of transcript files
            with the clinical scores appended as columns in the DataFrame.

    """
    panss = pd.read_csv(SCORES_FILENAME, index_col='session')
    filtered_transcripts = []
    for transcript in transcripts:
        try:
            scores = panss.ix[transcript.session]
            transcript.set_scores(scores)
            filtered_transcripts.append(transcript)
        except KeyError:
            warn('No clincal scores found for session {}, '
                 'removing.'.format(transcript.session))
    log('Scores aligned for {} sessions.'.format(len(filtered_transcripts)))
    return filtered_transcripts


def construct_transcripts(root):
    """Constructs the Transcript objects with all relevant data.

    Returns:
        list of utils.reader.harvard.Transcript: A list of transcript files in
            the corpus.

    """
    transcripts = check_pkl(TRANSCRIPT_FILENAME)
    if not transcripts:
        transcripts = assign_scores(get_transcripts(root))
        add_perplexities(transcripts)
        add_liwc(transcripts)
        add_repairs(transcripts)
        pkl(transcripts, TRANSCRIPT_FILENAME)
    return transcripts
