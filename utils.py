# Utilities for the DIConf Notebooks

import sys
import nltk
import string

from nltk.corpus import wordnet as wn


##########################################################################
## Data Loading Utility
##########################################################################


def documents(corpus, fileids=None, categories=None, norm=False):
    """
    Generator yields tokenized documents in the corpus.
    """
    for doc in corpus.docs(fileids=fileids, categories=categories):
        doc = [
            token
            for paragraph in doc
            for sentence in paragraph
            for token in sentence
        ]

        if norm:
            yield normalize(doc)
        else:
            yield doc


def utterances(corpus, fileids=None, categories=None, norm=False):
    """
    Generator yeilds tokenized sentences in the corpus.
    """
    for sent in corpus.sents(fileids=fileids, categories=categories):
        if norm:
            yield normalize(sent)
        else:
            yield doc


def labels(corpus, fileids=None, categories=None):
    """
    Generator yields labels for the given subset of docs.
    """
    for fileid in corpus._resolve(fileids, categories):
        yield corpus.categories(fileids=[fileid])[0]


##########################################################################
## Helper Functions
##########################################################################

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg


##########################################################################
## Preprocessing
##########################################################################

class Lemmatizer(object):
    """
    Wraps the nltk.WordNetLemmatizer to provide added functionality like the
    discovery of the part of speech of the word to lemmatize.
    """

    def __init__(self):
        self._wordnet = nltk.WordNetLemmatizer()
        self._cache   = {}

    def tagwn(self, tag):
        """
        Returns the WordNet tag from the Penn Treebank tag.
        """

        return {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

    def poswn(self, word):
        """
        Computes the part of speech for the given word.
        """
        return self.tagwn(nltk.pos_tag([word])[0][1])

    def lemmatize(self, word, tag=None):
        """
        Lemmatizes the word; if no tag is given, then computes the tag.
        """
        if (word, tag) in self._cache:
            return self._cache[(word, tag)]

        tag   = self.tagwn(tag) if tag else self.poswn(word)
        lemma = self._wordnet.lemmatize(word, tag)

        self._cache[(word, tag)] = lemma
        return lemma


PUNCTUATION = string.punctuation + "“”’—"

class Normalizer(object):

    def __init__(self, stopwords=None, punct=None):
        self.stopwords = set(stopwords or nltk.corpus.stopwords.words('english'))
        self.punct = str.maketrans('', '', PUNCTUATION)
        self.lemmatizer = Lemmatizer()

    def strip_punctuation(self, token):
        return token.translate(self.punct)

    def strip_stopwords(self, token):
        if token in self.stopwords:
            return ""
        return token

    def __call__(self, tokens):
        tokens = map(lambda s: self.lemmatizer.lemmatize(*s), tokens)
        tokens = map(lambda s: s.strip(), tokens)
        tokens = map(lambda s: s.lower(), tokens)
        tokens = map(lambda s: self.strip_stopwords(s), tokens)
        tokens = map(lambda s: self.strip_punctuation(s), tokens)
        return list(filter(None, tokens))

normalize = Normalizer()
