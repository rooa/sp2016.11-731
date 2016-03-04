#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import sys
import optparse


def unkify(tokens, words_dict):
    final = []
    for token in tokens:
        # only process the train singletons and unknown words
        if len(token.rstrip()) == 0:
            final.append('UNK')
        else:
            numCaps = 0
            hasDigit = False
            hasDash = False
            hasLower = False
            for char in token.rstrip():
                if char.isdigit():
                    hasDigit = True
                elif char == '-':
                    hasDash = True
                elif char.isalpha():
                    if char.islower():
                        hasLower = True
                    elif char.isupper():
                        numCaps += 1
            result = 'UNK'
            lower = token.rstrip().lower()
            ch0 = token.rstrip()[0]
            if ch0.isupper():
                if numCaps == 1:
                    result = result + '-INITC'
                    if words_dict.has_key(lower):
                        result = result + '-KNOWNLC'
                else:
                    result = result + '-CAPS'
            elif not(ch0.isalpha()) and numCaps > 0:
                result = result + '-CAPS'
            elif hasLower:
                result = result + '-LC'
            if hasDigit:
                result = result + '-NUM'
            if hasDash:
                result = result + '-DASH'
            if lower[-1] == 's' and len(lower) >= 3:
                ch2 = lower[-2]
                if not(ch2 == 's') and not(ch2 == 'i') and not(ch2 == 'u'):
                    result = result + '-s'
            elif len(lower) >= 5 and not(hasDash) and not(hasDigit and numCaps > 0):
                if lower[-2:] == 'ed':
                    result = result + '-ed'
                elif lower[-3:] == 'ing':
                    result = result + '-ing'
                elif lower[-3:] == 'ion':
                    result = result + '-ion'
                elif lower[-2:] == 'er':
                    result = result + '-er'
                elif lower[-3:] == 'est':
                    result = result + '-est'
                elif lower[-2:] == 'ly':
                    result = result + '-ly'
                elif lower[-3:] == 'ity':
                    result = result + '-ity'
                elif lower[-1] == 'y':
                    result = result + '-y'
                elif lower[-2:] == 'al':
                    result = result + '-al'
            final.append(result)
        #else:
        #    final.append(token.rstrip())
    assert len(final) == len(tokens)
    return final


def unkify2(tokens, words_dict):
    final = []
    for token in tokens:
        if words_dict.has_key(token):
            final.append(token)
            continue
        # only process the train singletons and unknown words
        if len(token.rstrip()) == 0:
            final.append('UNK')
        else:
            numCaps = 0
            hasDigit = False
            hasDash = False
            hasLower = False
            for char in token.rstrip():
                if char.isdigit():
                    hasDigit = True
                elif char == '-':
                    hasDash = True
                elif char.isalpha():
                    if char.islower():
                        hasLower = True
                    elif char.isupper():
                        numCaps += 1
            result = 'UNK'
            lower = token.rstrip().lower()
            ch0 = token.rstrip()[0]
            if ch0.isupper():
                if numCaps == 1:
                    result = result + '-INITC'
                    if words_dict.has_key(lower):
                        result = result + '-KNOWNLC'
                else:
                    result = result + '-CAPS'
            elif not(ch0.isalpha()) and numCaps > 0:
                result = result + '-CAPS'
            elif hasLower:
                result = result + '-LC'
            if hasDigit:
                result = result + '-NUM'
            if hasDash:
                result = result + '-DASH'
            if lower[-1] == 's' and len(lower) >= 3:
                ch2 = lower[-2]
                if not(ch2 == 's') and not(ch2 == 'i') and not(ch2 == 'u'):
                    result = result + '-s'
            elif len(lower) >= 5 and not(hasDash) and not(hasDigit and numCaps > 0):
                if lower[-2:] == 'ed':
                    result = result + '-ed'
                elif lower[-3:] == 'ing':
                    result = result + '-ing'
                elif lower[-3:] == 'ion':
                    result = result + '-ion'
                elif lower[-2:] == 'er':
                    result = result + '-er'
                elif lower[-3:] == 'est':
                    result = result + '-est'
                elif lower[-2:] == 'ly':
                    result = result + '-ly'
                elif lower[-3:] == 'ity':
                    result = result + '-ity'
                elif lower[-1] == 'y':
                    result = result + '-y'
                elif lower[-2:] == 'al':
                    result = result + '-al'
            final.append(result)
        #else:
        #    final.append(token.rstrip())
    assert len(final) == len(tokens)
    return final


def main(opts):
    f1 = open(opts.ap, "r").read()
    f2 = open(opts.afp, "r").read()
    f3 = open(opts.nyt, "r").read()
    print "File read..."

    # tok_f1 = nltk.word_tokenize(f1)
    # tok_f2 = nltk.word_tokenize(f2)
    # tok_f3 = nltk.word_tokenize(f3)
    # print "Tokenize done.."

    tok_f1 = f1.strip().split()
    tok_f2 = f2.strip().split()
    tok_f3 = f3.strip().split()

    corpus = tok_f1 + tok_f2 + tok_f3
    print corpus[:10]
    words_list = nltk.FreqDist(corpus)
    print "Original: " , words_list.items()[:10]
    cutted = {k: v for (k, v) in words_list.items() if v > int(opts.cutoff)}
    print "After cutting: ", cutted.items()[:10]
    print "FreqDist constructed..."

    final = unkify2(corpus, cutted)
    print "UNKified.."
    print final[:10]

    fout = open(opts.output, "w")
    fout.write(" ".join(final))
    fout.close()
    print "Output done, Terminate..."

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--cutoff", dest="cutoff", default=5,
                         help="Cutoff value for UNK-ifying vocabs")
    optparser.add_option("-o", "--output", dest="output", default="~/data/3corpus_unkified.txt",
                         help="Output destination")
    optparser.add_option("-A", "--AP", dest="ap", default="/usr0/home/akuncoro/all_english_corpora/APNews_tokenized.txt",
                         help="AP")
    optparser.add_option("-f", "--AFP", dest="afp", default="/usr0/home/akuncoro/all_english_corpora/AFP_tokenized.txt",
                         help="AFP")
    optparser.add_option("-n", "--nyt", dest="nyt", default="/usr0/home/akuncoro/all_english_corpora/NYT_tokenized.txt",
                         help="NYT")

    (opts, _) = optparser.parse_args()
    main(opts)
