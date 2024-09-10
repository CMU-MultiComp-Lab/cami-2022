"""Contains a reader for the HM dataset and the associated Transcript class."""

import os
import re
import string

import numpy as np
import pandas as pd

from utils.log import debug
from utils.memoize import memoize

HM_REMOVE_TAGS = [r"\[inaudible\]", r"\[laughter\]", r"\[crosstalk\]", r"\[redacted\]"]

COLUMN_NAMES = ["speaker", "timestamp", "text"]


class Transcript(object):
    """Holds a transcript session and all associated data and scores.

    Args:
        filepath (str): Path to a CSV file containing the utterances to import.

    Attributes:
        scores (dict): Clinical scores associated with the transcript.
        session (str): Name of the session, e.g., 240521_MS0059.
    """

    def __init__(self, filepath):
        self.data = pd.read_csv(filepath, delimiter="\t", names=COLUMN_NAMES)

        self.scores = dict()
        self.session = os.path.basename(filepath).split(".")[0]

        self.__clean()

        # TODO: handle this more elegantly
        self.data = self.data[self.data["speaker"] == "Participant"]

        debug("Initialized session {}.".format(self.session))

    @staticmethod
    @memoize
    def __clean_text(text):
        """Cleans a utterance for processing.

        Args:
            text (str): The utterance to clean.

        Returns:
            str: The cleaned utterance.

        """
        for tag in HM_REMOVE_TAGS:
            text = re.sub(tag, "", text)
        text = text.translate(None, string.punctuation)
        text = " ".join(text.split())
        text = text.lower()
        text = text.decode("utf8", "ignore")
        return text

    def __clean(self):
        """Cleans all utterances in the transcript."""
        self.data["text"] = self.data["text"].apply(Transcript.__clean_text)
        self.data["text"].replace("", np.nan, inplace=True)
        self.data.dropna(subset=["text"], inplace=True)
        debug("Cleaned session {}.".format(self.session))

    def set_scores(self, scores):
        """Assigns clinicial scores to the transcript session.

        Args:
            scores (dict): A dictionary of the form {scale (str): score
                (float)} containing all clinical scores to attach to the
                session.

        """
        debug("Transcript {} scores:")
        for scale, score in scores.items():
            self.scores[scale] = score
            debug("\t{} score: {}".format(scale, score))

    def export_lines(self):
        """Exports utterances to plaintext lines.

        Returns:
            list of str: A list of utterance text lines in the transcript.

        """
        debug("Exporting session {} to file.".format(self.session))
        return self.data["text"].tolist()


def get_transcript_list(root):
    """Gathers a list of all HM transcript locations.

    Args:
        root (str): The root directory to search within.

    Returns:
        list of str: A list of filepaths where transcripts are located.

    """
    filepaths = []
    for filename in os.listdir(root):
        path = os.path.join(root, filename)
        if os.path.isfile(path):
            filepaths.append(path)
    debug("Found {} transcripts.".format(len(filepaths)))
    return filepaths


def get_transcripts(root):
    """Gathers a list of all Transcript objects.

    Returns:
        list of Transcript: A list of transcript files in the corpus.

    """
    filepaths = get_transcript_list(root)
    transcripts = []
    for filepath in filepaths:
        transcripts.append(Transcript(filepath))
    debug("Loaded {} transcripts.".format(len(transcripts)))
    return transcripts
