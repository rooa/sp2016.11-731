#!/usr/bin/env python
import argparse
import sys
import models
import heapq
from collections import namedtuple, defaultdict


parser = argparse.ArgumentParser(description='Simple phrase based decoder.')
parser.add_argument('-i', '--input', dest='input', default='data/input',
                    help='File containing sentences to translate (default=data/input)')
parser.add_argument('-t', '--translation-model', dest='tm', default='data/tm',
                    help='File containing translation model (default=data/tm)')
parser.add_argument('-s', '--stack-size', dest='s',
                    default=5, type=int, help='Maximum stack size (default=1)')
parser.add_argument('-n', '--num_sentences', dest='num_sents', default=sys.maxint,
                    type=int, help='Number of sentences to decode (default=no limit)')
parser.add_argument('-l', '--language-model', dest='lm', default='data/lm',
                    help='File containing ARPA-format language model (default=data/lm)')
parser.add_argument('-v', '--verbose', dest='verbose',
                    action='store_true', default=False, help='Verbose mode (default=off)')
opts = parser.parse_args()

tm = models.TM(opts.tm, sys.maxint)
lm = models.LM(opts.lm)
sys.stderr.write('Decoding %s...\n' % (opts.input,))
input_sents = [tuple(line.strip().split())
               for line in open(opts.input).readlines()[:opts.num_sents]]


def contains_sublist(lst, sublst):
    n = len(sublst)
    return any((sublst == lst[i:i + n]) for i in xrange(len(lst) - n + 1))


def new_hyp(h, phrase, idx, size, fromIndex, toIndex, span_future_logprob):
    '''
    input:
        h: current hypothesis block
        phrase: a phrase to translate
        idx: index of the ending of the phrase
        size: sentence length
    output:
        logprob: The resulting log-probability for h + phrase
        lm_state: The resulting sequence of words translated
        hypothesis: hypothesis block for stacks that contains information above
    '''
    global lm
    logprob = h.logprob + phrase.logprob
    lm_state = h.lm_state
    # copy the bit vectors, and modify it by turning on all bits between fromIndex and toIndex
    new_bit_array = list(h.coverage_vector)
    for covered_src_word_index in range(fromIndex, toIndex):
        new_bit_array[covered_src_word_index] = True
    new_bit_array = tuple(new_bit_array)
    for word in phrase.english.split():
        (lm_state, word_logprob) = lm.score(lm_state, word)
        logprob += word_logprob
    future_logprob = compute_future_logprob(new_bit_array, span_future_logprob)
    heuristic_logprob = future_logprob
    logprob += lm.end(lm_state) if idx == size else 0.0
    return logprob, lm_state, hypothesis(logprob, heuristic_logprob, new_bit_array, future_logprob, lm_state, h, phrase)


def extract_english_recursive(h):
    return '' if h.predecessor is None else '%s%s ' % (extract_english_recursive(h.predecessor), h.phrase.english)


def extract_tm_logprob(h):
    return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)


def precompute_span_future_logprob(f):
    span_future_logprob = defaultdict(lambda: defaultdict(float))
    # example: f == ('a', 'b', 'c'); len(f) == 3; span_size in [1, 2, 3]
    for span_size in range(1, len(f) + 1):  # decide the span size
        # example: span_size == 2; i in [0, 1]
        for i in range(0, len(f) - (span_size - 1)):  # decide the starting point
            # example: i == 1; j in [2, 3]
            for j in range(i + 1, len(f) + 1):  # decide the ending point
                # initialize the logprob of this span to -1000000000000000000
                span_future_logprob[i][j] = -10000000000000
                # for each possible decomposition of the range(i, j)
                # example: i == 0; j == 3; k in [1, 2]
                for k in range(i + 1, j):
                    # if it turns out translating individual components in this range has a better logprob, use that decomposition
                    if span_future_logprob[i][k] + span_future_logprob[k][j] > span_future_logprob[i][j]:
                        span_future_logprob[i][j] = span_future_logprob[i][k] + span_future_logprob[k][j]
                # for each possible translation of f[i:j]
                if f[i:j] not in tm:
                    continue
                for phrase in tm[f[i:j]]:
                    local_lm_logprob = 0
                    lm_state = ()
                    for word in phrase.english.split():
                        (lm_state, word_logprob) = lm.score(lm_state, word)
                        local_lm_logprob += word_logprob
                    phraseCost = local_lm_logprob + phrase.logprob
                    if phraseCost > span_future_logprob[i][j]:
                        span_future_logprob[i][j] = phraseCost
    return span_future_logprob


def compute_future_logprob(coverage_vector, span_future_logprob):
    future_logprob = 0
    insideUncoveredSpan = False
    uncoveredSpanStartsAt = -1
    for i in range(0, len(coverage_vector)):
        if coverage_vector[i] and insideUncoveredSpan:
            future_logprob += span_future_logprob[uncoveredSpanStartsAt][i]
            insideUncoveredSpan = False
            uncoveredSpanStartsAt = -1
        elif not coverage_vector[i] and not insideUncoveredSpan:
            insideUncoveredSpan = True
            uncoveredSpanStartsAt = i
        if insideUncoveredSpan and i == len(coverage_vector) - 1:
            future_logprob += span_future_logprob[uncoveredSpanStartsAt][len(coverage_vector)]
    return future_logprob


def extract_phrases(h):
    """ Return a list of phrases (en, es) in the hypothesis Recursive """
    if not h.predecessor:
        return []
    else:
        ec = extract_phrases(h.predecessor)
        ec.extend([(h.phrase.english, h.src)])
        return ec


hypothesis = namedtuple('hypothesis', 'logprob, heuristic_logprob, coverage_vector, future_logprob, lm_state, predecessor, phrase')

'''
for f in input_sents:
    lm_state = ()
    result = []
    for word in f:
        (lm_state, word_logprob) = lm.score(lm_state, word)
        result.append(str(word_logprob))
    print ' '.join(f)
    print ' '.join(result)
'''


for f in input_sents:
    N = len(f)
    stacks = [{} for _ in f] + [{}]

    span_future_logprob = precompute_span_future_logprob(f)  # precompute everything

    # Place empty hypothesis
    coverage_vector = tuple([False for _ in f])
    initial_hypothesis = hypothesis(0.0, 0.0, coverage_vector, 0.0, lm.begin(), None, None)
    stacks[0][lm.begin()] = initial_hypothesis

    for i, stack in enumerate(stacks[:-1]):
        # If i is not the end of the sentence, you can look at 2 words ahead
        if i + 2 < len(stacks):
            # Pick top [opts.s] hypotheses
            for h in heapq.nlargest(opts.s, stacks[i].itervalues(), key=lambda h: h.logprob + h.heuristic_logprob):
                for k in xrange(i + 1, N + 1):
                    # If a translation is found in f[i:k], find the next phrase starting from k until j, and make a new hypothesis till that point (take into account the reordering)
                    if f[i:k] in tm:
                        for j in xrange(k + 1, N + 1):
                            # If f[k:j] is a valid phrase, make a new hypothesis
                            if f[k:j] in tm:
                                reordering_hyps = []
                                for phrase in tm[f[k:j]]:
                                    _, _, new_hypothesis = new_hyp(h, phrase, j, N, k, j, span_future_logprob)
                                    reordering_hyps.append(new_hypothesis)
                                    # sys.stderr.write(str(new_hypothesis.heuristic_logprob) + "\n")
                                for hi in reordering_hyps:  # TODO: Modify this
                                    # Consider skipped f[i:k] and append them
                                    for phrase in tm[f[i:k]]:
                                        logprob, lm_state, new_hypothesis = new_hyp(hi, phrase, k, N, i, k, span_future_logprob)
                                        # logprob, lm_state, new_hypothesis = new_hyp(hi, phrase, k, N)
                                        # second case is recombination
                                        if lm_state not in stacks[j] or (stacks[j][lm_state].logprob + stacks[j][lm_state].heuristic_logprob) < (new_hypothesis.logprob + new_hypothesis.heuristic_logprob):
                                            # Finally add hypothesis with reordering
                                            stacks[j][lm_state] = new_hypothesis

        # Add hypotheses with no reordering
        for h in heapq.nlargest(opts.s, stacks[i].itervalues(), key=lambda h: h.logprob + h.heuristic_logprob):
            for k in xrange(i + 1, N + 1):
                if f[i:k] in tm:
                    for phrase in tm[f[i:k]]:
                        logprob, lm_state, new_hypothesis = new_hyp(h, phrase, k, N, i, k, span_future_logprob)

                        if lm_state not in stacks[k] or (stacks[k][lm_state].logprob + stacks[k][lm_state].heuristic_logprob) < (new_hypothesis.logprob + new_hypothesis.heuristic_logprob):
                            stacks[k][lm_state] = new_hypothesis

    # find best translation by looking at the best scoring hypothesis
    # on the last stack
    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)

    translation = extract_phrases(winner)
    # TODO


    print extract_english_recursive(winner)

    if opts.verbose:
        tm_logprob = extract_tm_logprob(winner)
        sys.stderr.write('LM = %f, TM = %f, Total = %f\n' %
                         (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
