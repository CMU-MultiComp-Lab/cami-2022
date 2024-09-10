"""A module for reading from the Switchboard corpus as transcribed by ISIP."""
import os
import re
import string

from utils.bar import progress_bar
from utils.log import log
from utils.memoize import memoize

SWITCHBOARD_ROOT = os.path.join('_data', 'swb_ms98_transcriptions')

REMOVE_TAGS = [r'\[laughter\]', r'\[noise\]', r'\[silence\]',
               r'\[vocalized-noise\]']


def get_subdirectories(root_dir):
    """Get all subdirectories in a particular directory.

    Args:
        root_dir (str): The root directory to search within.

    Returns:
        list of str: Filepaths pointing to subdirectories in the directory.
    """
    return [os.path.join(root_dir, filename)
            for filename in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, filename))]


def get_transcript_list():
    """Generate a list of filepaths to all transcripts.

    Returns:
        list of str: A list of filepaths pointing to transcripts in the corpus.
    """
    transcripts = [os.path.join(session, transcript)
                   for group in get_subdirectories(SWITCHBOARD_ROOT)
                   for session in get_subdirectories(group)
                   for transcript in os.listdir(session)
                   if 'trans' in transcript]
    return transcripts


@memoize
def clean_utterance(utt):
    """Clean up one utterance for training.

    Args:
        utt (str): Raw copy of the utterance as read from the transcript.

    Returns:
        str: Cleaned version of the utterance, removing punctuation and
            numbers, lowercase, removed noise tags.
    """
    for tag in REMOVE_TAGS:
        utt = re.sub(tag, '', utt)
    utt = re.sub(r'\[laughter-(.*?)\]', r'\1', utt)
    utt = utt.translate(None, string.punctuation)
    utt = ''.join([c for c in utt if not c.isdigit()])
    utt = ' '.join(utt.split())
    utt = utt.lower()
    return utt


def get_transcript_utterances(transcript_fp):
    """Generate a list of all utterances from a transcript.

    Args:
        transcript_fp (str): Filepath pointing to a transcript file.

    Yields:
        str: An utterance from the transcript
    """
    with open(transcript_fp) as infile:
        lines = infile.readlines()

    for line in lines:
        utt = ' '.join(line.split()[3:])
        utt = clean_utterance(utt)
        if utt:
            yield utt


def get_utterances():
    """Gather all utterances in the corpus.

    Returns:
        list of str: All utterances from the corpus.
    """
    transcripts = get_transcript_list()

    log('Loading Switchboard corpus...')
    num_files = len(transcripts)
    progress = progress_bar(max_val=num_files)
    progress.start()

    utts = []
    for idx, transcript in enumerate(transcripts):
        for utt in get_transcript_utterances(transcript):
            utts.append(utt)
        progress.update(idx + 1)

    progress.finish()
    log('Loaded {} Switchboard transcripts.'.format(num_files))
    return utts
