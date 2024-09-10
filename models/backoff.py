"""Handles a backoff language model."""

import collections

import numpy as np
import nltk

from utils.bar import progress_bar
from utils.log import log, debug, error
from utils.pkl import pkl

OOV_TOKEN = '__oov__'


class BackoffLM(dict):
    """Handles a backoff language model.

    Args:
        oov (str): Token to use for out-of-vocabulary words.
        min_freq (int): Minimum number of appearances for a word to be
            included in the vocabulary.
        n (int): Default n-gram size.

    Attributes:
        vocab (collections.Counter): Keeps track of the number of times each
            word is observed, as well as which words are in the vocabulary.
        oov (str): Token to use for out-of-vocabulary words.
        min_freq (int): Minimum number of appearances for a word to be
            included in the vocabulary.
        n (int): Default n-gram size.
        is_normalized (bool): Tracks whether the model has been normalized
            (cannot train further after normalization).

    """

    def __init__(self, oov=OOV_TOKEN, min_freq=2, ngram_size=3):

        super(BackoffLM, self).__init__()
        self.vocab = collections.Counter()
        self.oov = oov
        self.min_freq = min_freq
        self.ngram_size = ngram_size
        self.is_normalized = False

        log('Initialized {}-gram BackoffLM.'.format(self.ngram_size))

    def __to_vocab(self, text):
        """Constrains text to words contained in the vocabulary of the model.

        Args:
            text (tuple of str or str): A word or tuple of words to constrain.

        Returns:
            tuple of str or str: Input constrained to the vocabulary of the
                model.

        """
        if isinstance(text, tuple):
            return tuple([self.__to_vocab(word) for word in text])
        return text if text in self.vocab else self.oov

    def __train_vocab(self, utts):
        """Generate a vocabulary for the model.

        Args:
            utts (iterable of str): A list of utterances to train the model's
                vocabulary upon.

        """
        log('Training vocabulary...')

        num_utts = len(utts)
        progress = progress_bar(max_val=num_utts)
        progress.start()

        # add words from utterance to the vocabulary counter
        for idx, utt in enumerate(utts):
            self.vocab.update([None] + utt.split())
            progress.update(idx + 1)

        progress.finish()

        # treat tokens observed less than self.min_freq as oov
        oov_words = set([w for (w, c) in self.vocab.items()
                         if c < self.min_freq])
        self.vocab[self.oov] = 0
        for word in oov_words:
            self.vocab[self.oov] += self.vocab[word]
            del self.vocab[word]

        log('Found {} words with a frequency of at least {}.'
            .format(len(self.vocab), self.min_freq))
        debug('Removed {} words for a total of {} oov tokens.'
              .format(len(oov_words), self.vocab[self.oov]))

    @staticmethod
    def __generate_ngrams(text, max_ngram_size):
        """Generates all ngrams of size max_ngram_size or lower.

        Args:
            text (str): The text to generate ngrams from.
            max_ngram_size (int): The maximum size of the ngrams to generate.

        Yields:
            tuple of str: An ngram in the given text.

        """
        for ngram_size in xrange(max_ngram_size):
            for ngram in nltk.ngrams(text.split(), ngram_size + 1,
                                     pad_left=True, pad_right=True):
                yield ngram

    def train(self, utts):
        """Train the backoff model upon a series of utterances.

        Args:
            utts (iterable of str): A list of utterances to train the model
                upon.

        """
        if self.is_normalized:
            error('Cannot train model that is already normalized.')
            exit()

        self.__train_vocab(utts)

        log('Training model...')

        num_utts = len(utts)
        progress = progress_bar(max_val=num_utts)
        progress.start()

        for idx, utt in enumerate(utts):
            for ngram in BackoffLM.__generate_ngrams(utt, self.ngram_size):
                context = ngram[:-1]
                word = ngram[-1]
                self.__observe(word, context)

            progress.update(idx + 1)

        progress.finish()
        self.__normalize()
        log("Trained on {} utterances.".format(num_utts))

    def __observe(self, word, context):
        """Train upon a single ngram.

        Args:
            word (str): The word of interest.
            context (iterable of str): The ngram context for the observed word.

        """
        context = self.__to_vocab(context)
        word = self.__to_vocab(word)

        if context not in self:
            self[context] = collections.defaultdict(float)
        self[context][word] += 1

    def __normalize(self):
        """Normalizes the model from counts to probabilities."""
        for context in self:
            count = sum(self[context].values())
            denom = float(count + len(self.vocab))

            for word in self[context]:
                self[context][word] = (self[context][word] + 1) / denom
            self[context][self.oov] = 1.0 / len(self.vocab)
        self.is_normalized = True
        debug('Normalized model.')

    def prob(self, word, context):
        """Determines the probability of a word appearing in a given context.

        Args:
            word (str): The word that was observed.
            context (iterable of str): The words representing the context in
                which the word was observed.

        Returns:
            float: Probability of given word in context.

        """
        if not self.is_normalized:
            error('Cannot determine probabilities; model has not been '
                  'normalized.')
            exit()

        context = self.__to_vocab(context)
        word = self.__to_vocab(word)

        # backing off until we see a context we know
        while context not in self and len(context) > 1:
            context = tuple(context[1:])

        if word in self[context]:
            return self[context][word]
        return self.__uni_p(word)

    def __uni_p(self, word):
        """Determines the unigram probability of a given word.

        Args:
            word (str): The word of interest.

        Returns:
            float: The unigram probability of the word.

        """
        return float(self.vocab[word]) / sum(self.vocab.values())

    def perplexity(self, utt):
        """Calculates the perplexity of a single utterance.

        Args:
            utt (str): The utterance to calculate the perplexity of.

        Returns:
            float: The perplexity of the given utterance.

        """
        entropy = 0.0
        text = utt.split()
        ngrams = nltk.ngrams(text, self.ngram_size,
                             pad_left=True, pad_right=True)
        log_p = 0.0
        for ngram in list(ngrams)[:-1]:
            context = ngram[:-1]
            word = ngram[-1]
            log_p += np.log2(self.prob(word, context))
        entropy = -log_p / (len(text) + 1)
        return pow(2.0, entropy)

    def perplexity_multi(self, utts):
        """Calculates the perplexity of a set of utterances.

        Args:
            utts (iterable of str): The set of utterances to calculate the
                perplexity of.

        Returns:
            float: The perplexity of the given set of utterances.

        """
        perplexities = []
        for utt in utts:
            perplexities.append(self.perplexity(utt))
        return np.exp2(np.mean(np.log2(perplexities)))

    def save(self, filename):
        """Saves the model to filename.

        Args:
            filename (str): The filename to save the model to.

        """
        pkl(self, filename)
