#!/usr/bin/env python
import os, sys
import numpy as np
from scipy import stats
import pandas as pd

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import seaborn as sns

NEGFILE = 'rt-polarity.neg'
POSFILE = 'rt-polarity.pos'

FOLDS = 10
INNER = 3
N_CV = 100

MIN_DF = stats.randint(1, 11)
NGRAMS = ((1, 1), (1, 2), (2, 2))
STOPWS = (set(stopwords.words('english')), None)

SETTINGS_NB = {
    'pre__min_df': MIN_DF,
    'pre__ngram_range': NGRAMS,
    'pre__stop_words': STOPWS,
    'clf__alpha': stats.uniform(0.5, 2)
}

SETTINGS_SVC = {
    'pre__min_df': MIN_DF,
    'pre__ngram_range': NGRAMS,
    'pre__stop_words': STOPWS,
    'clf__C': stats.uniform(10e-5, 100),
}

SETTINGS_LR = {
    'pre__min_df': MIN_DF,
    'pre__ngram_range': NGRAMS,
    'pre__stop_words': STOPWS,
    'clf__C': stats.uniform(10e-5, 100),
}


def get_data():
    """
    Returns a preprocessed X with the entire corpus and a y with
    neg=0/pos=1 labels.
    """

    with open(NEGFILE, mode='r', encoding='cp1252') as f:
        neg_data = f.readlines()

    with open(POSFILE, mode='r', encoding='cp1252') as f:
        pos_data = f.readlines()

    n_neg = len(neg_data)
    n_pos = len(pos_data)
    n = n_neg + n_pos
    y = np.zeros(n)
    y[n_pos:] = 1

    X = np.concatenate([np.array(neg_data), np.array(pos_data)])

    return(X, y)


def classify(X, y, clf_type='nbc'):
    """
    Preprocess the input documents to extract feature vector representations of
    them. Your features should be N-gram counts, for N<=2.

    1. Experiment with the complexity of the N-gram features (i.e., unigrams,
       or unigrams and bigrams): `gram_min` + `gram_max`
    2. Experiment with removing stop words. (see NLTK)
    3. Remove infrequently occurring words and bigrams as features. You may tune
       the threshold at which to remove infrequent words and bigrams.
    4. Search over hyperparameters for the three models (nb, svm, lr) to
       find the best performing model.

    All 4 of the above are done in the context of 10-fold cross validation on
    the data. On the training data, 3-fold cross validation is done to find the
    optimal hyperparameters (using randomized CV), which are then tested on
    held-out data.
    """

    if clf_type == 'nbc':
        clf = BernoulliNB()
        params = SETTINGS_NB
    elif clf_type == 'svc':
        clf = LinearSVC()
        params = SETTINGS_SVC
    elif clf_type == 'lrc':
        clf = LogisticRegression()
        params = SETTINGS_LR
    else:
        raise Exception('invalid clf {}: {nbc, svc, lrc}'.format(clf_type))

    # pipeline runs preprocessing and model during every CV loop
    pipe = Pipeline([
        ('pre', CountVectorizer()),
        ('clf', clf),
    ])

    model = RandomizedSearchCV(
        pipe, params, n_jobs=-1, n_iter=N_CV, cv=INNER, scoring='f1_macro'
    )

    results = {
        'test':  {'loss': [], 'accuracy': [], 'confusion': [], 'errors': []},
        'train': {'loss': [], 'accuracy': [], 'confusion': []},
        'cv': {}
    }

    kf = StratifiedKFold(n_splits=FOLDS, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        print("[{}] {}/{}".format(clf_type, i+1, FOLDS))

        # split training and test sets
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # fit model
        model.fit(X_train, y_train)

        # save the best parameters from the inner-fold cross validation
        best_params = model.best_estimator_.get_params()
        for p in sorted(params.keys()):
            results['cv'][p] = best_params[p]

        # make predictions on train and test set
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        # record some misclassified sentences
        idx_errors = np.where(y_test_pred != y_test)[0]
        np.random.shuffle(idx_errors)
        errors = X_test[idx_errors[:5]]
        results['test']['errors'].extend(errors)

        # store results
        results['test']['loss'].append(log_loss(y_test, y_test_pred))
        results['test']['accuracy'].append(accuracy_score(y_test, y_test_pred))
        results['test']['confusion'].append(confusion_matrix(y_test, y_test_pred))
        results['train']['loss'].append(log_loss(y_train, y_train_pred))
        results['train']['accuracy'].append(accuracy_score(y_train, y_train_pred))
        results['train']['confusion'].append(confusion_matrix(y_train, y_train_pred))

    return(results)


def main():

    X, y = get_data()

    nb_results = classify(X, y, clf_type='nbc')
    sv_results = classify(X, y, clf_type='svc')
    lr_results = classify(X, y, clf_type='lrc')

    db = pd.DataFrame()
    accs = np.concatenate([
        nb_results['train']['accuracy'], nb_results['test']['accuracy'],
        sv_results['train']['accuracy'], sv_results['test']['accuracy'],
        lr_results['train']['accuracy'], lr_results['test']['accuracy']]
    )

    labels = ['train'] * FOLDS + ['test'] * FOLDS
    labels = np.array(labels * 3)

    models = np.array(['nieve bayes'] * FOLDS * 2 +
             ['linear SVM'] * FOLDS * 2 +
             ['logistic regression'] * FOLDS * 2)

    db['accuracy'] = accs
    db['phase'] = labels
    db['model'] = models

    sns.boxplot(
        x="model", y="accuracy", hue="phase", palette=["grey", "red"], data=db
    )
    sns.despine(offset=10, trim=True)
    plt.title('Average model performance across {} folds for the 3 models'.format(
        FOLDS))
    plt.tight_layout()
    plt.savefig('accs.jpg')

    c_matrix = np.stack(nb_results['test']['confusion'], axis=0)
    c_matrix_mean = np.mean(c_matrix, axis=0)
    c_matrix_std = np.std(c_matrix, axis=0)

    import IPython; IPython.embed()

if __name__ == '__main__':
    main()

