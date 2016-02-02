#!/usr/bin/env python
# -*- coding: utf-8 -*-

import optparse
import sys
from collections import defaultdict
import numpy as np

bitext_original = None
bitext_dev = None
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)
p_e_given_f = defaultdict(lambda: defaultdict(float))
weighted_counts = defaultdict(float)


def read_corpus(opts):
    global bitext_original, bitext_dev, f_count, e_count, fe_count, p_e_given_f
    sys.stderr.write("Training with IBM Model 1...")
    bitext_original = [[sentence.lower().strip().split() for sentence in pair.split(' ||| ')]
                       for pair in open(opts.bitext)]
    bitext_dev = bitext_original[:opts.num_sents]
    for (n, (f, e)) in enumerate(bitext_dev):
        for f_i in set(f):
            f_count[f_i] += 1
            for e_j in set(e):
                fe_count[(f_i, e_j)] += 1
        for e_j in set(e):
            e_count[e_j] += 1
        if n % 500 == 0:
            sys.stderr.write(".")

    num_e_count = len(e_count.keys())
    for n, (f_i, e_i) in enumerate(fe_count.keys()):
        p_e_given_f[f_i][e_i] = 1.0 / num_e_count
        if n % 5000 == 0:
            sys.stderr.write(".")


def expectation_sent(f_text, e_text):
    global p_e_given_f
    n = len(f_text)
    p_a_given_e = 1.0 / (n)
    res = {}
    for (i, e_i) in enumerate(e_text):
        a_i_distribution = [(p_a_given_e * p_e_given_f[f_i][e_i]) for f_i in f_text]
        p_ai_given_fe = map(lambda x: x / sum(a_i_distribution), a_i_distribution)
        res.update({(f_j, e_i): p_ai_given_fe[j] for (j, f_j) in enumerate(f_text)})
    return res


def EM():
    global bitext_dev, p_e_given_f
    weighted_counts = defaultdict(lambda: defaultdict(float))
    f_vocabularies = set()
    # E-step, calculating weighted scores
    for (f_text, e_text) in bitext_dev:
        res = expectation_sent(f_text, e_text)
        for ((f, e), score) in res.iteritems():
            weighted_counts[f][e] += score
            f_vocabularies.add(f)

    sys.stderr.write("E-step is over.\n")
    sys.stderr.write("German Vocabulary size: %d\n" % len(f_vocabularies))
    # M-step, updating table
    for f_i in f_vocabularies:
        # row = filter(lambda ((x, y), z): x == f_i, weighted_counts.iteritems())   # SLOW
        row = weighted_counts[f_i]
        denominator = sum(row.values())
        weighted_counts[f_i] = {k: (v / denominator) for (k, v) in row.iteritems()}

    p_e_given_f = weighted_counts
    sys.stderr.write("M-step is over\n")


def align_sent(f_text, e_text):
    global p_e_given_f
    res = []
    for (j, e) in enumerate(e_text):
        i = np.argmax([p_e_given_f[f][e] for f in f_text])
        res.append((i, j))
    return res


def align():
    global bitext_original, p_e_given_f
    for (f_text, e_text) in bitext_original:
        res = align_sent(f_text, e_text)
        sys.stdout.write(" ".join(["-".join([str(i), str(j)]) for (i, j) in res]))
        sys.stdout.write("\n")


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en",
                         help="Parallel corpus (default data/dev-test-train.de-en)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint,
                         type="int", help="Number of sentences to use for training and alignment")
    (opts, _) = optparser.parse_args()
    read_corpus(opts)
    for i in range(5):
        EM()
    sys.stderr.write(sorted(p_e_given_f['der'].items(), key=lambda x: x[1], reverse=True)[:5])
    align()
