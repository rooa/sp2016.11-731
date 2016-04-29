#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from tqdm import tqdm


f = open(sys.argv[1], "r").read().strip().split("\n")
out_f = open(sys.argv[1] + "_tagless", "w")

for line in tqdm(f):
    tokens = line.split()
    out_f.write(" ".join(w for w in tokens[1:-1]) + "\n")
out_f.close()
