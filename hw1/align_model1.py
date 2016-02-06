#!/usr/bin/env python
# -*- coding: utf-8 -*-

import optparse
import sys
from collections import defaultdict
import numpy as np
from nltk.stem.snowball import SnowballStemmer

bitext_original = None
bitext_stemmed = None
bitext_dev = None
bitext_test = None
e_count = defaultdict(int)
fe_count = defaultdict(int)
p_e_given_f = defaultdict(lambda: defaultdict(float))
weighted_counts = defaultdict(float)

deu_stemmer = SnowballStemmer("german", ignore_stopwords=True)
eng_stemmer = SnowballStemmer("english", ignore_stopwords=True)

# Ge -> En
supplemental_pair_txt = "data/german_most_freq_translated_pair_500.txt"
# En -> De, then flipped
supplemental_pair_txt2 = "data/english_most_freq_translated_pair_500.txt"


def read_corpus(opts):
    global bitext_original, bitext_stemmed, bitext_dev, f_count, e_count, fe_count, p_e_given_f
    sys.stderr.write("Training with IBM Model 1...\n")
    bitext_original = [[sentence.decode('utf8').strip().split() for sentence in pair.split(' ||| ')]
                       for pair in open(opts.bitext)][:opts.num_sents]
    bitext_test = [[["NULL"] + [deu_stemmer.stem(i) for i in f],
                    [eng_stemmer.stem(i) for i in e]]
                   for (f, e) in bitext_original]

    supplemental_pairs = [[sentence.decode('utf8').strip().split() for sentence in pair.split(' ||| ')]
                          for pair in open(supplemental_pair_txt)]
    supplemental_pairs_2 = [[sentence.decode('utf8').strip().split() for sentence in pair.split(' ||| ')]
                            for pair in open(supplemental_pair_txt2)]
    bitext_original += (supplemental_pairs + supplemental_pairs_2) * opts.rept_count
    bitext_stemmed = [[["NULL"] + [deu_stemmer.stem(i) for i in f],
                       [eng_stemmer.stem(i) for i in e]]
                      for (f, e) in bitext_original]

    sys.stderr.write("Stemming finished.\n")
    bitext_dev = bitext_stemmed
    for (n, (f, e)) in enumerate(bitext_dev):
        for f_i in set(f):
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
        if i > 0:
            res.append((i - 1, j))
    return res


def align():
    global bitext_test, p_e_given_f
    for (f_text, e_text) in bitext_test:
        res = align_sent(f_text, e_text)
        sys.stdout.write(" ".join(["-".join([str(i), str(j)]) for (i, j) in res]))
        sys.stdout.write("\n")


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en",
                         help="Parallel corpus (default data/dev-test-train.de-en)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint,
                         type="int", help="Number of sentences to use for training and alignment")
    optparser.add_option("-r", "--repeat_count", dest="rept_count", default=sys.maxint,
                         type="int", help="Repetion count for most frequent words")
    (opts, _) = optparser.parse_args()
    read_corpus(opts)
    for i in range(6):
        EM()
    sys.stderr.write(str(sorted(p_e_given_f['der'].items(), key=lambda x: x[1], reverse=True)[:5]))
    align()
