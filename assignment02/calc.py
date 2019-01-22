#!/usr/bin/env python

import pandas as pd
import numpy as np
from copy import copy

def get_alpha(words, tags, init, a, B, filename, fwd=True):
    """forward algorithm"""

    T = np.zeros((len(tags), len(words)))

    # first compute trelis matrix
    for i, word in enumerate(words):
        for j, tag in enumerate(tags):

            if i == 0:
                T[j, i] = init[tag]*B[word][tag]

            else:
                # used for EM
                if fwd:
                    # use np.multiply
                    T[j, i] = np.sum(np.array([
                        T[0, i-1] * a[tag][tags[0]],
                        T[1, i-1] * a[tag][tags[1]],
                        T[2, i-1] * a[tag][tags[2]],
                        T[3, i-1] * a[tag][tags[3]]]) * B[word][tag])

                # used for viterbi
                else:
                    # use np.multiply
                    T[j, i] = np.max(np.array([
                        T[0, i-1] * a[tag][tags[0]] * B[word][tag],
                        T[1, i-1] * a[tag][tags[1]] * B[word][tag],
                        T[2, i-1] * a[tag][tags[2]] * B[word][tag],
                        T[3, i-1] * a[tag][tags[3]] * B[word][tag]
                    ]))

    with open(filename, 'w') as f:

        # next generate csv of 'computations'
        lines = [','+','.join(words)+'\n']

        for j, tag in enumerate(tags):
            line = '{},'.format(tag)
            for i, word in enumerate(words):
                if i == 0:
                    line += '{:.2f} * {:.2f} = {:.2f},'.format(
                        init[tag], B[word][tag], T[j, i])
                else:
                    # used for EM
                    if fwd:
                        line += '({:.2f}*{:.2f} + {:.2f}*{:.2f} + {:.2f}*{:.2f} + {:.2f}*{:.2f}){:.2f} = {:.5f},'.format(
                            T[0, i-1], a[tag][tags[0]],
                            T[1, i-1], a[tag][tags[1]],
                            T[2, i-1], a[tag][tags[2]],
                            T[3, i-1], a[tag][tags[3]], B[word][tag], T[j, i]
                        )
                    # used for viterbi
                    else:
                        line += 'max({:.2f}*{:.2f}*{:.2f}; {:.2f}*{:.2f}*{:.2f}; {:.2f}*{:.2f}*{:.2f}; {:.2f}*{:.2f}*{:.2f}) = {:.5f},'.format(
                            T[0, i-1], a[tag][tags[0]], B[word][tag],
                            T[1, i-1], a[tag][tags[1]], B[word][tag],
                            T[2, i-1], a[tag][tags[2]], B[word][tag],
                            T[3, i-1], a[tag][tags[3]], B[word][tag],
                            T[j, i]
                        )
            line += '\n'
            lines.append(line)

        f.writelines(lines)
        f.close()

    return(T)


def get_betas(words, tags, a, B, filename):
    """backward algorithm"""

    words.reverse() # words are reversed so I can iterate forwards

    T = np.zeros((len(tags), len(words)))

    for i, word in enumerate(words):

        # when i=0, we're at the last step. We can't look back, so
        # we fill the final column with 1s and move on
        if i == 0:
            T[:, -1] = 1 # fill end probabilities
            continue

        # reverse indexing
        idx = T.shape[1] - 1 - i

        # use np.multiply
        for j, tag in enumerate(tags):
            T[j, idx] = np.sum(np.array([
                T[0, idx+1] * a[tags[0]][tag] * B[words[i-1]][tags[0]],
                T[1, idx+1] * a[tags[1]][tag] * B[words[i-1]][tags[1]],
                T[2, idx+1] * a[tags[2]][tag] * B[words[i-1]][tags[2]],
                T[3, idx+1] * a[tags[3]][tag] * B[words[i-1]][tags[3]],
            ]))

    # write output for report
    words.reverse()
    output = pd.DataFrame(T, index=tags, columns=words)
    output.to_csv(filename, index=True, float_format='%.3f')

    return(T)


def get_squig(words, tags, alpha, beta, gamma, a, B, p_o_o):
    """
    otherwise known as 'xi', probability of transitioning between
    two states at each time step
    """
    squig = np.zeros((len(tags), len(tags), len(words)-1))

    for t, word in enumerate(words[:-1]):
        for i, tagi in enumerate(tags):
            for j, tagj in enumerate(tags):
                squig[i,j,t] = (alpha[i, t] * a[tagi][tagj] *
                                B[words[t+1]][tagj] * beta[j, t+1]) / p_o_o

    return(squig)


def get_gamma(sentence, tags, alpha, betas, p_o_o, filename):
    """
    probability of being in a state at a point in time given the
    observed sequence
    """
    gamma = (alpha * betas) /  p_o_o
    output = pd.DataFrame(gamma, index=tags, columns=sentence)
    output.to_csv(filename, index=True, float_format='%.3f')

    return(gamma)


def soft_em(sentences, tags, init, a, B):
    """processes ALL sentences to compute one updated init, a, and B."""

    new_init = np.zeros(len(tags))
    new_a = []
    gammas = []
    squigs = []

    new_B = copy(B)
    new_B[:] = 1        # start with 1 for laplace smoothing
    #B_changes = copy(B) #
    #B_changes[:] = 0    # used to keep track of which states we actually updated

    for n, sentence in enumerate(sentences):
        print('em on sentence {}'.format(sentence))
        alpha = get_alpha(sentence, tags, init, a, B, 'alphas_em_{}.csv'.format(n), fwd=True)
        betas = get_betas(sentence, tags, a, B, 'betas_em_{}.csv'.format(n))
        p_o_o = np.sum(alpha[:, -1])
        print('p_o_o={}'.format(p_o_o))
        gamma = get_gamma(sentence, tags, alpha, betas, p_o_o, 'gamma_em_{}.csv'.format(n))
        squig = get_squig(sentence, tags, alpha, betas, gamma, a, B, p_o_o)

        gammas.append(gamma)
        squigs.append(squig)

        sentence = np.array(sentence)

        # updates to B
        for i, tag in enumerate(tags):
            for word in np.unique(sentence):
                idx = np.where(sentence == word)[0]
                # each sentence is added + normalized one-by-one
                new_B[word][tag] += np.sum(gamma[i, idx]) / np.sum(gamma[i, :])

                # old way -- didn't sum to 1, so I think it was incorrect
                #if B_changes[word][tag] == 0:
                #    # we replace the old value in B
                #    new_B[word][tag] = np.sum(gamma[i, idx]) / np.sum(gamma[i, :])
                #else:
                #    # we add (so we can divide by n later)
                #    new_B[word][tag] += np.sum(gamma[i, idx]) / np.sum(gamma[i, :])
                #B_changes[word][tag] += 1

    # normalize init
    for gamma in gammas:
        new_init += gamma[:, 0]
    new_init /= len(sentences)

    # normalize a
    squigs = np.concatenate(squigs, axis=2)
    gammas = np.concatenate(gammas, axis=1)
    new_a = np.sum(squigs, axis=2) / np.repeat(
        np.atleast_2d(np.sum(gammas, axis=1)).T, len(tags), axis=1)

    # smooth B
    new_B = new_B.div(new_B.sum(axis=1), axis=0).fillna(0) # laplace smoothing

    # old way -- didn't sum to 1, so I think it was incorrect
    #sentences = np.array(sentences)
    #for i, tag in enumerate(tags):
    #    for words in np.unique(sentences.ravel()):
    #        if B_changes[word][tag] > 0:
    #            new_B[word][tag] /= B_changes[word][tag]

    # not used anymore
    #sentence_lengths = np.array([len(x) for x in sentences])
    #sentence_weights = np.sum(sentence_lengths / np.max(sentence_lengths))

    # convert formats
    new_init = dict(zip(tags, new_init))
    new_a = pd.DataFrame(new_a, index=tags, columns=tags)
    print('initial probabilities after EM: {}'.format(new_init))

    new_a.to_csv('new_a_em.csv', index=True, float_format='%.3f')
    new_B.to_csv('new_B_em.csv', index=True, float_format='%.3f')

    return(new_init, new_a, new_B)


if __name__ == "__main__":

    # !! initial probabilities
    tags = {'C': 2, 'N': 0, 'V': 2, 'J': 0}
    count = sum(tags.values())
    n = len(tags.keys())

    tag_prob = {}
    tag_smoothed_prob = {}

    for key in tags.keys():
        tag_prob[key] = tags[key] / count
        tag_smoothed_prob[key] = (tags[key] + 1) / (count+n)

    print('unsmoothed initial probabilities:\n{}'.format(tag_prob))
    print('smoothed initial probabilities:\n{}'.format(tag_smoothed_prob))
    init = tag_smoothed_prob


    # !! transition probabilities
    df = pd.read_csv('transitions.csv', index_col=0)
    df_smoothed = df + 1
    df = df.div(df.sum(axis=1), axis=0).fillna(0)
    df_smoothed = df_smoothed.div(df_smoothed.sum(axis=1), axis=0).fillna(0)
    df.to_csv('transitions_prob.csv',
        index=True, float_format='%.3f')
    df_smoothed.to_csv('transitions_smoothed_prob.csv',
        index=True, float_format='%.3f')
    a = df_smoothed # saved for later


    # !! emission probabilities
    df = pd.read_csv('emissions.csv', index_col=0)
    df_smoothed = df + 1
    df = df.div(df.sum(axis=1), axis=0).fillna(0)
    df_smoothed = df_smoothed.div(df_smoothed.sum(axis=1), axis=0).fillna(0)
    df.to_csv('emissions_prob.csv',
        index=True, float_format='%.3f')
    df_smoothed.to_csv('emissions_smoothed_prob.csv',
        index=True, float_format='%.3f')
    B = df_smoothed # saved for later

    tags = ["C", "N", "V", "J"]
    alpha_0 = get_alpha(["that", "is", "good"], tags, init, a, B, 'trellis.csv')

    # !! expectation maximization
    # make new emissions matrix with BAD
    B = pd.read_csv('emissions.csv', index_col=0)
    B['bad'] = 0
    B = B + 1
    B = B.div(B.sum(axis=1), axis=0).fillna(0)
    B.to_csv('emissions_smoothed_prob_newword.csv',
        index=True, float_format='%.3f')

    # !! compute EM with new sentences
    sentences = [['bad', 'is', 'not', 'good'], ['is', 'it', 'bad']]
    new_init, new_a, new_B = soft_em(sentences, tags, init, a, B)

    # probability of a new sequence:
    sentence = ['that', 'is', 'bad']
    alpha = get_alpha(
        sentence, tags, new_init, new_a, new_B, 'new_sentence_prob.csv', fwd=True)
    likelihood_of_sentence = np.sum(alpha[:, -1])
    print('likelihood of sentence = {}'.format(likelihood_of_sentence))

    print('***complete, drop to interactive prompt***')
    print('(type "quit()" to escape)')
    import IPython; IPython.embed()

