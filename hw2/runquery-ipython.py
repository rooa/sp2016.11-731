#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle as pickle
import numpy as np
import subprocess
import optparse
import tempfile
import nltk
import sys


def load_data():
    prefix = "/usr0/home/hiroakih/data/sp2016.11-731/hw2/data/"
    train = prefix + "train-test.hyp1-hyp2-ref"
    ans = prefix + "train.gold"
    train_set = [[sentence.strip() for sentence in pair.split(' ||| ')] for pair in open(train)]
    print train_set[0]
    ans_set = [int(i) for i in open(ans)]
    return train_set, ans_set

# def sentences():
#     with open(opts.train, "r") as f:
#         for pair in f:
#             yield [sentence.strip().split() for sentence in pair.split(' ||| ')]
#
# # note: the -n option does not work in the original code
# for h1, h2, ref in islice(sentences(), opts.num_sentences):


def run_command(string):
    s = nltk.word_tokenize(string.decode('utf8'))
    tmp = tempfile.NamedTemporaryFile(prefix='tmp.txt', dir='.', delete=False)
    tmp.write(" ".join(s).encode('utf8') + "\n")
    name = tmp.name
    tmp.close()
    supposed_text = open(tmp.name, 'r').read()
    command = ['/usr0/home/hiroakih/tools/mosesdecoder/bin/query',
               '-n',
               '-l lazy',
               "/usr0/home/hiroakih/tools/mosesdecoder/3corpus_unkified_small.binary",
               '<',
               tmp.name]
    raw_result = subprocess.check_output(" ".join(command), shell=True, stderr=subprocess.STDOUT)
    perplexity_string = raw_result.split("\n")[1].strip()
    result = float(perplexity_string[perplexity_string.find(":") + 1:].strip())
    subprocess.check_output("rm -f " + tmp.name, shell=True, stderr=subprocess.STDOUT)
    return result


def main():
    sents, labels = load_data()
    sys.stderr.write("Dataset loaded...\n")
    res_array = np.empty((len(sents), 3), dtype=np.float64)

    for (i, (s1, s2, ref)) in enumerate(sents):
        perplex_s1 = run_command(s1)
        perplex_s2 = run_command(s2)
        assert type(perplex_s1) == float
        res_array[i, 0] = perplex_s1
        res_array[i, 1] = perplex_s2
        res_array[i, 2] = perplex_s1 - perplex_s2
        if i % 100 == 0:
            sys.stderr.write(".")
    sys.stderr.write("Training Data generated...\n")

    hw2data = {"feature": res_array, "label": np.array(labels)}
    with open("/usr0/home/hiroakih/data/hw2data.pickle", "wb") as f:
        pickle.dump(hw2data, f)

    return res_array


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-b", "--binary", dest="bin", default="/usr0/home/hiroakih/tools/mosesdecoder/3corpus_unkified_small.binary", help="Binary file to use")
    optparser.add_option("-i", "--input", dest="input", default="/usr0/home/hiroakih/data/sp2016.11-731/hw2/data/",
                         help="Threshold for aligning with Dice's coefficient (default=0.5)")
    optparser.add_option('-n', '--num_sentences', dest="num_sentences", default=None, type=int, help='Number of hypothesis pairs to evaluate')
    (opts, _) = optparser.parse_args()
    main(opts)
