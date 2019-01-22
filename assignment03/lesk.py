#!/usr/bin/env python
"""
data: https://www.cs.york.ac.uk/semeval-2013/task12/
"""
from gensim.models import Word2Vec
from nltk.corpus import semcor
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.wsd import lesk as nltk_lesk
from os.path import expanduser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from string import punctuation as PUNCTUATION
import argparse
import codecs
import gensim.downloader as api
import logging
import math
import nltk
import numpy as np
import os
import random
import sklearn
import sys
import xml.etree.cElementTree as ET

logging.basicConfig(level=logging.WARN, format="[%(name)s:%(funcName)s:%(lineno)s] %(levelname)s: %(message)s")
logger = logging.getLogger(os.path.basename(__file__))

STOP_WORDS = (list(stopwords.words('english')))
STOP_WORDS.append(None)
TOKENIZER = RegexpTokenizer(r'[\w_-]+')
LEMMATIZER = WordNetLemmatizer()
WORD2VEC = api.load("word2vec-google-news-300") # this is slow

class WSDInstance:

    def __init__(self, my_id, lemma, context, index):
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.index = index      # index of lemma within the context


    def __str__(self):
        '''For printing purposes'''
        return('%s\t%s\t%s\t%d'.format(
            self.id, self.lemma, ' '.join(self.context), self.index))


def get_semcor_corpus():
    """builds a corpus of word frequencies using SemCor"""
    corpus = []
    for sentence in semcor.sents():
        sentence_proc = preprocess(' '.join(sentence))
        for word in sentence_proc:
            corpus.append(word.lower())

    word_freq = FreqDist(corpus)

    corpus_freqs = {}
    freqs, words = [], []

    for word in corpus:
        freqs.append(word_freq[word])
        words.append(word)

    # laplace smoothing
    freqs = np.array(freqs)
    freqs += 1

    # compute inverse weighting
    N = len(word_freq)
    freqs =  np.log((1+N) / freqs)

    for freq, word in zip(freqs, words):
        corpus_freqs[word] = freq

    return(corpus_freqs)


def get_dev_corpus(dev):
    """builds a corpus of word frequencies using dev set"""
    corpus = []
    for sample in dev.keys():
        wsd = dev[sample]
        corpus.extend(preprocess(' '.join(wsd.context)))

    word_freq = FreqDist(corpus)

    freqs, words = [], []

    # get the frequency of each word in the corpus
    corpus_freqs = {}

    for word in corpus:
        freqs.append(word_freq[word])
        words.append(word)

    # laplace smoothing
    freqs = np.array(freqs)
    freqs += 1

    # compute inverse weighting
    N = len(word_freq)
    freqs =  np.log((1+N) / freqs)

    for freq, word in zip(freqs, words):
        corpus_freqs[word] = freq

    return(corpus_freqs)


def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is
    a dict, where the keys are the ids, and the values are instances of
    WSDInstance.
    '''
    tree = ET.parse(f)
    root = tree.getroot()

    dev_instances = {}
    test_instances = {}

    for text in root:

        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances

        for sentence in text:
            # construct sentence context
            #context = [to_ascii(el.attrib['lemma']) for el in sentence]
            context = [el.attrib['lemma'] for el in sentence]

            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    #lemma = to_ascii(el.attrib['lemma'])
                    lemma = el.attrib['lemma']
                    instances[my_id] = WSDInstance(my_id, lemma, context, i)

    return(dev_instances, test_instances)


def load_key(f):
    """load the solutions as dicts, key=id, value=list of correct sense keys"""
    dev_key = {}
    test_key = {}

    for line in open(f):
        if len(line) <= 1:
            continue
        #print (line)

        doc, my_id, sense_key = line.strip().split(' ', 2)

        if doc == 'd001':
            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()

    return(dev_key, test_key)


def intersect(a, b):
    """intersection between two sets"""
    return(len(set(a) & set(b)))


def weighted_intersect(a, b, weights):
    """intersection between two sets, weighted by inverse document frequency"""
    score = 0
    a = set(a) # remove duplicate words so we don't double-count
    for word in a:
        score += len(list(filter(lambda x: x == word, b))) * weights[word]

    logger.debug('a={}, b={}, score={}'.format(a, b, score))

    return(score)


def cosine(a, b):
    """cosine similarity between two vectors: a dot b / ||a||*||b||"""
    sum_xx, sum_yy, sum_xy = 1, 0, 0

    for i in range(len(a)):
        x = a[i]
        y = b[i]

        sum_xx += x*x
        sum_yy += y*y
        sum_xy += x*y

    return(sum_xy / math.sqrt(sum_xx * sum_yy))


def distributional(a, b, corpus):
    """
    computes overlap of two sets using the summed cosine distance using
    Google's 2013 word2vec model, which contains distributional information
    about word co-occourance.

    Approach inspired by An Enhanced Lesk Word Sense Disambiguation Algorithm
    through a Distributional Semantic Model, Basile et al 2014, COLING.
    """
    vec_a = np.zeros(300) # 300 is size of WORD2VEC embedding
    vec_b = np.zeros(300) #
    n_a = 0
    n_b = 0

    for word in a:
        try:
            vec_a += WORD2VEC.get_vector(word) * corpus[word]
            n_a += 1
        except Exception as e:
            logger.warning(e)
            pass

    for word in b:
        try:
            vec_b += WORD2VEC.get_vector(word) * corpus[word]
            n_b += 1
        except Exception as e:
            logger.warning(e)
            pass

    vec_a /= n_a # take mean vector over all matched words
    vec_b /= n_b #

    return(cosine(vec_a, vec_b))


def accuracy(X, y):

    hits = 0.0
    total = float(len(y))

    for predictions, targets in zip(X, y):
        for prediction in predictions:
            for target in targets:
                logger.debug('hits={}, target={}, prediction={}'.format(hits, target, prediction))
                if prediction == target:
                    hits += 1.0

    return((hits/total)*100)


def print_results(X, y, filename):
    """print results to a file and to the screen, for science"""
    columns = X.keys()
    with open(filename, 'w') as f:

        f.write(','.join(columns) + '\n')

        for i, col in enumerate(columns):

            acc_score = accuracy(X[col], y)
            f.write('{}'.format(acc_score))
            if i < len(columns)-1:
                f.write(',')

            print('accuracy: {}={:.2f}'.format(col, acc_score))


def get_context_freqs(target, context):
    """
    + gets the senses associated with the target word
    + define the corpus as word frequencies in all definitions
    + for each word in the context, find the inverse definition frequency

    loosely inspired by:

    [1] https://stackoverflow.com/questions/15551195/how-to-get-the-wordnet-sense-frequency-of-a-synset-in-nltk
    [2] An Enhanced Lesk Word Sense Disambiguation Algorithm through a
    Distributional Semantic Model
    [3] https://medium.freecodecamp.org/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3
    """
    synsets = wn.synsets(target)

    corpus = []
    for synset in synsets:
        corpus.extend(preprocess(synset.definition()))

        for example in synset.examples():
            corpus.extend(preprocess(example))

    word_freq = FreqDist(corpus)

    freqs, words = [], []

    # get the frequency of each context word
    context_freqs = {}

    for word in context:
        freqs.append(word_freq[word])
        words.append(word)

    # laplace smoothing
    freqs = np.array(freqs)
    freqs += 1

    # compute inverse weighting
    N = len(synsets)
    freqs =  np.log((1+N) / freqs)

    for freq, word in zip(freqs, words):
        context_freqs[word] = freq

    return(context_freqs)


def get_sense_freqs(word, corpus):
    """
    for each word sense, gets the frequency of occourance in the synsnet,
    normalized by number of occouranced in corpus
    uses laplace smoothing on frequencies
    inspired by https://stackoverflow.com/questions/15551195/how-to-get-the-wordnet-sense-frequency-of-a-synset-in-nltk
    and An Enhanced Lesk Word Sense Disambiguation Algorithm through a
    Distributional Semantic Model
    """
    synsets = wn.synsets(word)
    sense_freqs = {}
    freqs = []
    keys = []

    # search all synsets
    for s in synsets:
        freq = 0

        for lemma in s.lemmas():
            freq += lemma.count()

        freqs.append(freq)
        keys.append(get_first_sense_key(s))

    # laplace smoothing
    freqs = np.array(freqs)
    freqs += 1
    # corpus[word] is the count of the word in corpus
    # len(corpus) is the number of matching synsets
    try:
        freqs = freqs / (corpus[word] + len(freqs))
    except:
        freqs = freqs / (1 + len(freqs))

    for freq, key in zip(freqs, keys):
        sense_freqs[key[0]] = freq

    return(sense_freqs)


def preprocess(definition):
    """
    preprocesses definition by: tokenizing, removing punctuation and cardinals,
    lemmatizing, and removing stop words.
    """
    tokens = TOKENIZER.tokenize(definition)
    tokens = [t for t in tokens if t not in PUNCTUATION and t != '@card@']
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    tokens = list(filter(lambda x: x not in STOP_WORDS, tokens))
    tokens = [t.lower() for t in tokens]

    return(tokens)


def get_sense_keys(synset):
    """returns the sense key(s) as a list for the given synset"""
    keys = []
    for lemma in synset.lemmas():
        keys.append(lemma.key())
    return(keys)


def get_first_sense_key(synset):
    """returns the first sense key as a list for the given synset"""
    return([synset.lemmas()[0].key()])


def baseline(wsd):
    """returns most common sense for input wsd.lemma"""
    synset = wn.synsets(wsd.lemma)
    if len(synset) > 0:
        return(get_first_sense_key(synset[0]))
    else:
        logger.debug('synset empty for {}'.format(wsd.lemma))
        return(None)


def lesk_builtin(wsd):
    """returns word sense for synset found using lesk's algorithm"""
    synset = nltk_lesk(wsd.context, wsd.lemma)
    if synset is not None:
        return get_first_sense_key(synset)
    else:
        logger.debug('synset empty for {}'.format(wsd.lemma))
        return(None)


def lesk(wsd, corpus, lesk_lamb=1, syns_lamb=0, tidf_lamb=0, word_lamb=0):
    """
    wsd:       a word sense disambiguation object
    corpus:    a list of words that represent our dev_set corpus (preprocessed)
    lesk_lamb: weight of the simplified lesk overlap signal      (0 = off)
    syns_lamb: weight of the synset distributional signal        (0 = off)
    tidf_lamb: weifht of the tf-idf synset distributional signal (0 = off)
    word_lamb: weight of the word2vec distributional signal      (0 = off)

    NB: lesk/syns/tidf/word lamb will always be forced to sum to 1, and at least
    one must be nonzero.
    """
    synsets = wn.synsets(wsd.lemma)

    wsd.lemma = wsd.lemma.lower()

    logger.info('** disambiguating {} with the following settings:'.format(wsd.lemma))
    logger.info('   lesk_lamb={}, syns_lamb={}, tidf_lamb={} word_lamb={}'.format(
        lesk_lamb, syns_lamb, tidf_lamb, word_lamb))

    if len(synsets) == 0:
        logger.warning('initial synset probe empty for {}'.format(wsd.lemma))
        return(None)

    # enforce probability (all lambdas must sum to 1)
    total_lamb = lesk_lamb + syns_lamb + word_lamb

    if total_lamb == 0:
        lesk_lamb += 0.1
        syns_lamb += 0.1
        word_lamb += 0.1
        total_lamb = lesk_lamb + syns_lamb + word_lamb
    else:
        lesk_lamb = lesk_lamb / total_lamb
        syns_lamb = syns_lamb / total_lamb
        word_lamb = word_lamb / total_lamb

    logger.debug('lesk / syns / word lambdas: {} / {} / {}'.format(
        lesk_lamb, syns_lamb, word_lamb))

    # remove lemma, punctuation from context (TODO: move PUNTUATION to preproc?)
    context = list(filter(lambda x: wsd.lemma not in x, wsd.context))
    context = preprocess(' '.join(context))

    # get the distribution of the different senses (more fine-grained than
    # simply using the most frequent sense by default)
    if syns_lamb > 0:
        synset_freqs = get_sense_freqs(wsd.lemma, corpus)

    # get the inverse document frequencies of the context words
    if tidf_lamb > 0:
        tidf_freqs = get_context_freqs(wsd.lemma, context)

    # initialize with most frequent best sense for word
    best_synset = synsets[0]
    logger.debug('using initial synset as best: {}'.format(best_synset))

    # using our three sources of information, search for the best score
    best_score = 0
    for synset in synsets:

        # include gloss of sense
        signature = preprocess(synset.definition())

        # include examples of sense
        for example in synset.examples():
            signature += preprocess(example)

        logger.info('final context:   {}'.format(' '.join(context)))
        logger.info('final signature: {}'.format(' '.join(signature)))

        # classic lesk overlap
        if lesk_lamb > 0:
            lesk_score = intersect(context, signature)

            # laplace smoothing of classic if using either distributional method
            # converts overlap to a 'probability', weighted by lesk_lamb
            if any([word_lamb, tidf_lamb, syns_lamb]):
                logger.debug('lesk score pre-smooth:  {}'.format(lesk_score))
                lesk_score = (lesk_score+1) / (len(context)+1)
                logger.debug('lesk score post-smooth: {}'.format(lesk_score))
                lesk_score = lesk_lamb * lesk_score
        else:
            lesk_score = 0

        # synset distribution score
        if syns_lamb > 0:
            syns_score = syns_lamb * synset_freqs[get_first_sense_key(synset)[0]]
        else:
            syns_score = 0

        # tf-idf lesk overlap
        if tidf_lamb > 0:
            tidf_score = tidf_lamb * weighted_intersect(context, signature, tidf_freqs)
        else:
            tidf_score = 0

        # word2vec cosine similarity
        if word_lamb > 0:
            word_score = word_lamb * distributional(context, signature, corpus)
        else:
            word_score = 0

        total_score = lesk_score + syns_score + tidf_score + word_score

        if total_score > best_score:
            logger.info('(lesk+syns+word)={:.2f}+{:.2f}+{:.2f}, {} wins'.format(
                lesk_score, syns_score, word_score, synset))

            best_score = total_score
            best_synset = synset

    if best_synset is None:
        logger.warning('synset empty for {}'.format(wsd.lemma))
        return(None)

    logger.info('final definition found {}:{}\n'.format(
        best_synset, best_synset.definition()))

    return(get_first_sense_key(best_synset))


def run(X, name, results):
    """runs algo, stores results in X['name'], and returns other information"""
    X[name].append(results)
    return(X)


def main():
    """
    - [x] implement and apply Lesks algorithm to SemEval 2013 Shared Task using
          nltk and wordnet 3.0
    - [x] tokenize, lemmatize, remove stop words
    - [x] compare (for WSD) using accuracy.:
        - [x] the most frequent sense baseline: this is the sense indicated as
              #1 in the synset according to WordNet
        - [x] NLTKs implementation of Lesks algorithm (nltk.wsd.lesk)
        - [x] custom Lesk implementation
    - [x] There is sometimes more than one correct sense annotated in the key.
          Any are correct -- ALWAYS USE THE FIRST.
    - [x] Develop 2 additional methods for word sense disamiguation.
        - [x] One of them must combine distributional information about the
              frequency of word senses, and the standard Lesk's algorithm.
        - [x] Another method of my own design.
    - [x] Convert gold standard keys to word sense keys:
          https://wordnet.princeton.edu/man/senseidx.5WN.html
    - [x] convert multi-words to first_second (deal with upper/lower case)
    - [x] id d001+ is dev. remaining cases are test.
    - [x] final report: 2 pages explaining experiments with 3 models, and
          discussion. include sample output, analysis, improvement suggestions.
          Report justifying design etc. s.a. what to include in the ense and
          context representations, how to compute overlap, and trade off
          distributional and lesk signal.
    """

    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}

    logger.info('lesk go ;) ** wordnet version: {}'.format(wn.get_version()))

    # holds results
    X_dev = {'baseline': [], 'lesk_official': [],
             'custom:lesk': [], 'custom:lesk+tidf': [], 'custom:lesk+syns': [], 'custom:lesk+word': []}
    X_test = {'baseline': [], 'lesk_official': [],
             'custom:lesk': [], 'custom:lesk+tidf': [], 'custom:lesk+syns': [], 'custom:lesk+word': []}

    y_dev = []
    y_test = []

    # build corpus using dev set (try SemCor?)
    #corpus = get_dev_corpus(dev_instances)
    corpus = get_semcor_corpus()

    n = 10

    # run experiments DEV SET
    for sample in dev_instances.keys():

        wsd = dev_instances[sample]

        # baseline experiment
        X_dev = run(X_dev, 'baseline', baseline(wsd))

        # using NLTK's builtin lesk algorithm
        X_dev = run(X_dev, 'lesk_official', lesk_builtin(wsd))

        # using the simplified lesk implementation with no modifications
        X_dev = run(X_dev, 'custom:lesk', lesk(wsd, corpus))

        # add y values
        y_dev.append(dev_key[sample])

    # run experiments TEST SET
    for sample in test_instances.keys():

        wsd = test_instances[sample]

        # baseline experiment
        X_test = run(X_test, 'baseline', baseline(wsd))

        # using NLTK's builtin lesk algorithm
        X_test = run(X_test, 'lesk_official', lesk_builtin(wsd))

        # using the simplified lesk implementation with no modifications
        X_test = run(X_test, 'custom:lesk', lesk(wsd, corpus))

        # add y values
        y_test.append(test_key[sample])


    # using the simplified lesk implementation and synset distributions
    # grid search
    lesk_lambs = [random.random()+0.01 for _ in range(0, n)]
    syns_lambs = [random.random()+0.01 for _ in range(0, n)]
    best_lesk_lamb = 0
    best_syns_lamb = 0
    best_acc = 0

    for lesk_lamb in lesk_lambs:
        for syns_lamb in syns_lambs:

            predictions = []
            for sample in dev_instances.keys():
                wsd = dev_instances[sample]
                predictions.append(lesk(wsd, corpus, lesk_lamb=lesk_lamb, syns_lamb=syns_lamb))

            acc_score = accuracy(predictions, y_dev)
            if acc_score > best_acc:
                best_acc = acc_score
                best_lesk_lamb = lesk_lamb
                best_syns_lamb = syns_lamb

    logger.info('GRID SEARCH DONE -- lesk+syns: best_acc={}, lesk_lamb={}, syns_lamb={}'.format(
        best_acc, best_lesk_lamb, best_syns_lamb ))

    for sample in dev_instances.keys():
        wsd = dev_instances[sample]
        X_dev = run(X_dev, 'custom:lesk+syns', lesk(wsd, corpus, lesk_lamb=best_lesk_lamb, syns_lamb=best_syns_lamb))

    for sample in test_instances.keys():
        wsd = test_instances[sample]
        X_test = run(X_test, 'custom:lesk+syns', lesk(wsd, corpus, lesk_lamb=best_lesk_lamb, syns_lamb=best_syns_lamb))

    # using the simplified lesk implementation and tf-idf of dev corpus
    #X = run(X, 'custom:lesk+tidf', lesk(wsd, corpus, lesk_lamb=0.3, tidf_lamb=0.7))




    # using the simplified lesk implementation and weighted word2vec distance
    # grid search
    lesk_lambs = [random.random()+0.01 for _ in range(0, n)]
    word_lambs = [random.random()+0.01 for _ in range(0, n)]
    best_lesk_lamb = 0
    best_word_lamb = 0
    best_acc = 0

    for lesk_lamb in lesk_lambs:
        for word_lamb in word_lambs:

            predictions = []
            for sample in dev_instances.keys():
                wsd = dev_instances[sample]
                predictions.append(lesk(wsd, corpus, lesk_lamb=lesk_lamb, word_lamb=word_lamb))

            acc_score = accuracy(predictions, y_dev)
            if acc_score > best_acc:
                best_acc = acc_score
                best_lesk_lamb = lesk_lamb
                best_word_lamb = word_lamb

    logger.info('GRID SEARCH DONE -- lesk+word: best_acc={}, lesk_lamb={}, word_lamb={}'.format(
        best_acc, best_lesk_lamb, best_word_lamb ))

    for sample in dev_instances.keys():
        wsd = dev_instances[sample]
        X_dev = run(X_dev, 'custom:lesk+word', lesk(wsd, corpus, lesk_lamb=best_lesk_lamb, word_lamb=best_word_lamb))

    for sample in test_instances.keys():
        wsd = test_instances[sample]
        X_test = run(X_test, 'custom:lesk+word', lesk(wsd, corpus, lesk_lamb=best_lesk_lamb, word_lamb=best_word_lamb))

    print_results(X_dev, y_dev, 'dev_results.csv')
    print_results(X_test, y_test, 'test_results.csv')

    logger.warning('jackie <3 u')
    import IPython; IPython.embed()


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-v", "--verbose", action="count",
        help="increase output verbosity")
    args = argparser.parse_args()

    if args.verbose == None:
        logger.setLevel(logging.WARN)
    elif args.verbose > 1:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    log_fname = "lesk_log.txt"
    if os.path.isfile(log_fname):
        os.remove(log_fname)
    log_hdl = logging.FileHandler(log_fname)
    log_hdl.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(log_hdl)

    main()

