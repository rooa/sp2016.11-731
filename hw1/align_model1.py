#!/usr/bin/env python
# -*- coding: utf-8 -*-

import optparse
import sys
from collections import defaultdict
import numpy as np

bitext = None
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)
p_e_given_f = defaultdict(float)
weighted_counts = defaultdict(float)

def read_corpus(opts):
    global bitext, f_count, e_count, fe_count, p_e_given_f
    sys.stderr.write("Training with IBM Model 1...")
    bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')]
              for pair in open(opts.bitext)][:opts.num_sents]

    for (n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            f_count[f_i] += 1
            for e_j in set(e):
                fe_count[(f_i, e_j)] += 1
        for e_j in set(e):
            e_count[e_j] += 1
        if n % 500 == 0:
            sys.stderr.write(".")

    p_e_given_f = defaultdict(float)
    num_e_count = len(e_count.keys())
    for n, (f_i, e_i) in enumerate(fe_count.keys()):
        p_e_given_f[(f_i, e_i)] = 1.0 / num_e_count
        if n % 500 == 0:
            sys.stderr.write(".")


def expectation_sent(f_text, e_text):
    global p_e_given_f
    n = len(f_text)
    p_a_given_e = 1.0 / (n)
    res = {}
    for (i, e_i) in enumerate(e_text):
        a_i_distribution = [(p_a_given_e * p_e_given_f[(f_i, e_i)]) for f_i in f_text]
        p_ai_given_fe = map(lambda x: x / sum(a_i_distribution), a_i_distribution)
        res.update({(f_j, e_i): p_ai_given_fe[j] for (j, f_j) in enumerate(f_text)})


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en",
                         help="Parallel corpus (default data/dev-test-train.de-en)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint,
                         type="int", help="Number of sentences to use for training and alignment")
    (opts, _) = optparser.parse_args()
    read_corpus(opts)
    expectation_sent(bitext[0][0], bitext[0][1])
