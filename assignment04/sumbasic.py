#!/usr/bin/env python

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
from numpy import ndarray
from string import punctuation
import argparse
import numpy as np
import os
import string
import sys

STOP_WORDS = (list(stopwords.words('english')))
STOP_WORDS.append(None)
LEMMATIZER = WordNetLemmatizer()
PUNCTUATION = punctuation

def get_sentences_and_tokens(filename):
    with open(filename) as f:
        sentences = sent_tokenize(f.read())

    tokens = []
    for sentence in sentences:
        processed_sentence = preprocess(word_tokenize(sentence))
        tokens.append(processed_sentence)

    return(sentences, tokens)


def preprocess(tokens):
    """
    gets a document and produces preprocessed tokens by removing punctuation
    and cardinals, lemmatizing, and removing stop words.
    """
    tokens = [t for t in tokens if t not in PUNCTUATION and t != '@card@']
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    tokens = list(filter(lambda x: x not in STOP_WORDS, tokens))
    tokens = [t.lower() for t in tokens]

    return(tokens)


def get_word_probs(sentences):
    """gets p of each word (freq / # of total tokens)"""

    # make the corpus a non-nested list
    corpus = []
    for sentence in sentences:
        corpus.extend(sentence)

    # FreqDist does some of the heavy lifting
    word_freq = FreqDist(corpus)
    word_ps = {}

    # store in a custom dict so we can update probabilities
    for word in word_freq.keys():
        word_ps[word] = word_freq.freq(word)

    return(word_ps)


def count_words(summary):
    """counts words in summary, whether it be a list, or a string"""
    if isinstance(summary, list):
        return(sum(len(sentence.split()) for sentence in summary))

    return(len(summary.split()))


def print_first(sentences, length=100):
    """just returns the first sentences (up to length)"""
    summary = []

    for sentence in sentences:
        if count_words(summary) + count_words(sentence) > length:
            break
        summary.append(sentence)

    return(' '.join(summary))


def sumbasic(sentences, tokens, word_p, length=100, use_max_prob=True, update_ps=True):
    """
    implements one of 4 possible variants of the sumbasic algorithm.
    the default settings are for 'classic' sumbasic.
    """
    summary = []

    while count_words(summary) < length and len(sentences) > 0:

        # our 2 selection criteria (loop over n sentences to evaluate each time)
        max_prob_word = max(word_p, key=word_p.get)
        best_weight = 0

        for i, sentence in enumerate(sentences):

            # 2. calculate weight (mean probability) of sentence
            weight = sum(word_p[word] for word in tokens[i]) / len(tokens[i])

            # 3. pick sentence with highest weight & has max_prob_word
            if weight > best_weight:
                if use_max_prob:
                    if max_prob_word in tokens[i]:
                        highest_weight = weight
                        best_sentence = sentence
                        idx = i
                else:
                    highest_weight = weight
                    best_sentence = sentence
                    idx = i


        # 4. update probabilities by squaring them
        if update_ps:
            for word in tokens[idx]:
                word_p[word] = word_p[word]**2

        # store results, update remaining sentences
        summary.append(best_sentence)
        sentences.remove(best_sentence)
        tokens.remove(tokens[idx])


    return(' '.join(summary))


def main(method, file_n, length):

    sentences, tokens = [], []

    for document in file_n:
        these_sentences, this_token_set = get_sentences_and_tokens(document)
        sentences.extend(these_sentences)
        tokens.extend(this_token_set)

    word_p = get_word_probs(tokens)

    if method == 'orig':
        summary = sumbasic(sentences, tokens, word_p, length=length)
    elif method == 'best-avg':
        summary = sumbasic(sentences, tokens, word_p, length=length, use_max_prob=False)
    elif method == 'simplified':
        summary = sumbasic(sentences, tokens, word_p, length=length, update_ps=False)
    elif method == 'leading':
        summary = print_first(sentences, length=length)
    else:
        sys.exit('invalid method provided: {orig, best-avg, simplified, leading}')

    print(summary.strip('\n'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, help="{leading, simplified, best-avg, orig}")
    parser.add_argument('file_n', help="predix of articles to summarize", nargs='+')
    parser.add_argument('--length', type=int, default=100, help='output summary length')
    args = parser.parse_args()

    main(args.method, args.file_n, args.length)

