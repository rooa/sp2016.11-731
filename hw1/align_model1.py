#!/usr/bin/env python
# -*- coding: utf-8 -*-

import optparse
import sys
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en",
                     help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint,
                     type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

sys.stderr.write("Training with IBM Model 1...")
bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')]
          for pair in open(opts.bitext)][:opts.num_sents]
