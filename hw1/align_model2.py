#!/usr/bin/env python
# -*- coding: utf-8 -*-

import optparse
import sys
import cPickle as pickle
from collections import defaultdict
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import os


def dd():
    return defaultdict(float)

bitext_original = None
bitext_stemmed = None
bitext_dev = None
bitext_test = None
f_count = defaultdict(int)
fe_count = defaultdict(int)
ilm_count = defaultdict(int)
jilm_count = defaultdict(int)
p_e_given_f = defaultdict(dd)
p_j_given_i_l_m = {} 
#weighted_counts = defaultdict(float)

deu_stemmer = SnowballStemmer("german", ignore_stopwords=True)
eng_stemmer = SnowballStemmer("english", ignore_stopwords=True)

# Ge -> En
supplemental_pair_txt = "data/german_most_freq_translated_pair_500.txt"
# En -> De, then flipped
supplemental_pair_txt2 = "data/english_most_freq_translated_pair_500.txt"


def read_corpus(opts):
    global bitext_original, bitext_stemmed, bitext_dev, bitext_test 
    sys.stderr.write("Training with IBM Model 2...\n")
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

def initialise_counts():
    global bitext_dev, fe_count, f_count, ilm_count, jilm_count 
    for (n, (f, e)) in enumerate(bitext_dev):
        for f_i in f:
            for e_j in e:
                fe_count[(f_i, e_j)] = 0
        for f_i in f:
            f_count[f_i] = 0 
        if n % 500 == 0:
            sys.stderr.write(".")
    ilm_count_updates = defaultdict(bool)
    jilm_count_updates = defaultdict(bool)
    for n, (f, e) in enumerate(bitext_dev):
        for eng_position in range(len(e)): # i
            for deu_position in range(len(f)): # j
                ilm_key = str(eng_position) + '_' + str(len(f)) + '_' + str(len(e))
                if not(ilm_count_updates[ilm_key]):
                    ilm_count[ilm_key] = 0
                    ilm_count_updates[ilm_key] = True
                jilm_key = str(deu_position) + '_' + str(eng_position) + '_' + str(len(f)) + '_' + str(len(e))
                if not(jilm_count_updates[jilm_key]):
                    jilm_count[jilm_key] = 0
                    jilm_count_updates[jilm_key] = True 
    sys.stderr.write('\nFinish initialising counts\n')
    sys.stderr.write('Number of keys for fe_count: ' + str(len(fe_count)) + '\n')
    sys.stderr.write('Number of keys for f_count: ' + str(len(f_count)) + '\n')
    sys.stderr.write('Number of keys for ilm_count: ' + str(len(ilm_count)) + '\n')
    sys.stderr.write('Number of keys for jilm_count: ' + str(len(jilm_count)) + '\n')
    sys.stderr.flush()

def initialise_params(bias, pret):
    global bitext_dev, p_e_given_f, p_j_given_i_l_m, fe_count, jilm_count

    # initialise p_e_given_f by loading pickled parameters
    with open(pret, "rb") as f:
        p_e_given_f = pickle.load(f)
    num_keys = 0
    for f in p_e_given_f:
        num_keys += len(p_e_given_f[f])
    sys.stderr.write("Number of keys for p_e_given_f: " + str(num_keys) + '\n')
    sys.stderr.flush()
        
    # initialise p_e_given_f from scratch. TODO: make sure the probability sums to one
    #for n, (f_i, e_i) in enumerate(fe_count.keys()):
    #    p_e_given_f[f_i][e_i] = np.absolute(np.random.randn()) 
    #    if n % 5000 == 0:
    #        sys.stderr.write(".")
    # initialise p_j_given_i_l_m

    for k, (f_sent, e_sent) in enumerate(bitext_dev):
        for i, eng_word in enumerate(e_sent): # for each english word and its position in the english sentence    
            denominator = 0.0
            for j, deu_word in enumerate(f_sent): # for each german word and its position in the german sentence
                dict_key = str(j) + '_' + str(i) + '_' + str(len(f_sent)) + '_' + str(len(e_sent))
                curr_val = np.absolute(np.random.randn())
                while curr_val == 0.0:
                    curr_val = np.absolute(np.random.randn())    
                if i == (j-1):
                    curr_val *= bias 
                denominator += curr_val
                p_j_given_i_l_m[dict_key] = curr_val 
            for j, deu_word in enumerate(f_sent):
                dict_key = str(j) + '_' + str(i) + '_' + str(len(f_sent)) + '_' + str(len(e_sent))
                assert not(p_j_given_i_l_m[dict_key] == 0.0)
                p_j_given_i_l_m[dict_key] /= denominator 
       
    sys.stderr.write("\nFinish initialising parameters\n")
    sys.stderr.write('Number of keys for p_j_given_i_l_m: ' + str(len(p_j_given_i_l_m)) + '\n')
    sys.stderr.flush()
    

def expectation_corpus(bitext_dev): # for the whole corpus 
    global p_e_given_f, p_j_given_i_l_m, f_count, fe_count, ilm_count, jilm_count 
    delta_k_i_j = defaultdict(float)
    for k, (f_sent, e_sent) in enumerate(bitext_dev): # for each sentence in the parallel corpus
        # compute the denominator (normalising term) for this sentence
        for i, eng_word in enumerate(e_sent): # for each english word and its position in the english sentence
            denominator = 0.0
            # calculate the normaliser (called denominator) by marginalising (looping over the german words)
            for j, deu_word in enumerate(f_sent): # for each german word and its position in the german sentence
                dict_key = str(j) + '_' + str(i) + '_' + str(len(f_sent)) + '_' + str(len(e_sent)) # retrieve the key to the p_j_given_i_l_m dictionary based on the positions
                #assert not(p_e_given_f[deu_word][eng_word] == 0.0)
                #assert not(p_j_given_i_l_m[dict_key] == 0.0)
                curr_val = p_j_given_i_l_m[dict_key] * p_e_given_f[deu_word][eng_word]         
                delta_k_i_j[str(k) + '_' + str(i) + '_' + str(j)] = curr_val 
                denominator += curr_val
            # re-loop on the german words and then normalise and add the counts (expectation basically infers the counts given the parameters
            for j, deu_word in enumerate(f_sent): # for each german word and its position in the german sentence
                # normalise with the computed denominator at the previous step
                delta_k_i_j[str(k) + '_' + str(i) + '_' + str(j)] /= denominator 
                # initialise the increment value (normalised delta(k, i, j) in the Collins note)
                increment_value = delta_k_i_j[str(k) + '_' + str(i) + '_' + str(j)]
                # increment the f_count dict with the increment value
                assert deu_word in f_count # sanity check
                f_count[deu_word] += increment_value 
                # increment the fe_count dict with the increment value
                assert (deu_word, eng_word) in fe_count # sanity check
                fe_count[(deu_word, eng_word)] += increment_value
                # increment the ilm_count dict with the increment value
                assert str(i) + '_' + str(len(f_sent)) + '_' + str(len(e_sent)) in ilm_count
                ilm_count[str(i) + '_' + str(len(f_sent)) + '_' + str(len(e_sent))] += increment_value 
                # increment the jilm_count dict with the increment value
                assert str(j) + '_' + str(i) + '_' + str(len(f_sent)) + '_' + str(len(e_sent)) in jilm_count
                jilm_count[str(j) + '_' + str(i) + '_' + str(len(f_sent)) + '_' + str(len(e_sent))] += increment_value

def EM():
    global bitext_dev, p_e_given_f, p_j_given_i_l_m, f_count, fe_count, ilm_count, jilm_count
    sys.stderr.write('length of bitext dev: ' + str(len(bitext_dev)) + '\n')
    # get the expected counts in the E-step
    expectation_corpus(bitext_dev)
    # the counts are now updated, maximise the parameters in the M-step
    p_e_given_f_updates = defaultdict(bool) 
    p_j_given_i_l_m_updates = defaultdict(bool)
    for f_sent, e_sent in bitext_dev:
        for j, f_word in enumerate(f_sent):
            for i, e_word in enumerate(e_sent):
                if not(p_e_given_f_updates[(f_word, e_word)]):
                    #assert f_count[f_word] != 0 and f_count[f_word] != 0.0 # sanity check
                    p_e_given_f[f_word][e_word] = float(fe_count[(f_word, e_word)]) / float(f_count[f_word])             
                    p_e_given_f_updates[(f_word, e_word)] = True
                key_jilm = str(j) + '_' + str(i) + '_' + str(len(f_sent)) + '_' + str(len(e_sent))
                if not(p_j_given_i_l_m_updates[key_jilm]):
                    key_ilm = str(i) + '_' + str(len(f_sent)) + '_' + str(len(e_sent))
                    #assert not(jilm_count[key_jilm] == 0) and not(ilm_count[key_ilm] == 0)
                    assert not(ilm_count[key_ilm] == 0)
                    p_j_given_i_l_m[key_jilm] = float(jilm_count[key_jilm]) / float(ilm_count[key_ilm]) 
                    p_j_given_i_l_m_updates[key_jilm] = True

def align_sent(f_text, e_text):
    global p_e_given_f
    res = []
    for (i, eng_word) in enumerate(e_text):
        # for each english word, pick the most probable german word that is aligned to it 
        max_val = -np.inf
        max_idx = -1
        for (j, deu_word) in enumerate(f_text): 
            dict_key = str(j) + '_' + str(i) + '_' + str(len(f_text)) + '_' + str(len(e_text)) # retrieve the key to the p_j_given_i_l_m dictionary based on the positions
            #assert not(p_j_given_i_l_m[dict_key] == 0.0) and not(p_e_given_f[deu_word][eng_word] == 0.0)
            curr_val = p_j_given_i_l_m[dict_key] * p_e_given_f[deu_word][eng_word] 
            assert curr_val >= 0.0
            if curr_val > max_val:
                max_val = curr_val
                max_idx = j
        assert max_idx != -1
        if max_idx > 0:
            res.append((max_idx - 1, i))
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
    optparser.add_option("-i", "--num_iter", dest="num_iter", default=13,
                         type="int", help="number of iterations to run the EM for")
    optparser.add_option("-a", "--alpha", dest="alpha", default=3,
                         type="int", help="how much more weight we assign to diagonal entries ")
    optparser.add_option("-p", "--pretrained", dest="pretrained", default="p_e_given_f_EM3.pickle",
                         help="pre-trained translation table probabilities ")
    (opts, _) = optparser.parse_args()
    sys.stderr.write('Process ID: ' + str(os.getpid()) + '\n')
    sys.stderr.write('Number of iterations: ' + str(opts.num_iter) + '\n')
    sys.stderr.write('Repetition counts: ' + str(opts.rept_count) + '\n')
    sys.stderr.write('Alpha: ' + str(opts.alpha) + '\n')
    sys.stderr.flush()
    read_corpus(opts)
    initialise_counts()
    initialise_params(opts.alpha, opts.pretrained)
    sys.stderr.write('Number of iterations: ' + str(opts.num_iter) + '\n')
    sys.stderr.flush()
    sys.stderr.write('')
    for i in range(opts.num_iter):
        sys.stderr.write(str(i) + '\n')
        sys.stderr.flush()
        EM()
        initialise_counts()
        with open("pickled_params/p_e_given_f." + str(os.getpid())  + "." + str(opts.rept_count) + "_" + str(opts.num_iter) + "_" + str(opts.alpha)  + ".pickle", "wb") as file_:
            pickle.dump(p_e_given_f, file_)
        with open("pickled_params/p_j_given_i_l_m." + str(os.getpid())  + "_" + str(opts.rept_count) + "_" + str(opts.num_iter) + "_" + str(opts.alpha)  + ".pickle", "wb") as file_:
            pickle.dump(p_j_given_i_l_m, file_)
    #sys.stderr.write(str(sorted(p_e_given_f['der'].items(), key=lambda x: x[1], reverse=True)[:5]))
    align()
