#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('/usr0/home/hiroakih/UltraDeep')

import theano
import theano.tensor as T
# import theano.sandbox.cuda
import numpy as np
import scipy.io as sio
import codecs
import math
from copy import deepcopy
from collections import Counter
from tqdm import tqdm
import time

# theano.sandbox.cuda.use("gpu3")

# import from SandeepLearn
from network import LSTM
from layer import HiddenLayer, EmbeddingLayer
from learning_method import LearningMethod

from bleu import bleu_stats, bleu
# from config import src_embed_dim, tgt_embed_dim, src_lstm_op_dim, tgt_lstm_op_dim, beta
import config
# data
train_src = [line.strip().split() for line in codecs.open('data/train.src', 'r', encoding='utf8')]
train_tgt = [line.strip().split() for line in codecs.open('data/train.tgt', 'r', encoding='utf8')]
dev_src = [line.strip().split() for line in codecs.open('data/dev.src', 'r', encoding='utf8')]
dev_tgt = [line.strip().split() for line in codecs.open('data/dev.tgt', 'r', encoding='utf8')]
test_src = [line.strip().split() for line in codecs.open('data/test.src', 'r', encoding='utf8')]


def create_word_table(corpus):
    vocab = set()
    for line in corpus:
        for word in line:
            vocab.add(word)

    word2idx, idx2word = dict(), dict()

    for idx, word in enumerate(vocab):
        word2idx[word] = idx
        idx2word[idx] = word

    return word2idx, idx2word


def get_validation_bleu(hypotheses):
    stats = numpy.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, dev_tgt):
        hyp, ref = (hyp.strip().split(), ref.strip().split())
        stats += numpy.array(bleu_stats(hyp, ref))
    return "%.2f" % (100 * bleu(stats))


def get_predictions(src_word2idx, tgt_word2idx, tgt_idx2word, f, mode="validation"):
    if mode == 'validation':
        data = dev_src
    elif mode == 'test':
        data = test_src
    else:
        raise NotImplementedError("Pick validation / test.\n")

    predictions = []
    for idx, sent in enumerate(data):
        if idx % 300 == 0:
            sys.stderr.write("%s, %s\n" % (idx, len(data)))
        src_words = np.array([src_word2idx[x] for x in sent]).astype(np.int32)
        current_outputs = [tgt_word2idx['<s>']]

        while True:
            # The best choice
            next_word = f(src_words, current_outputs).argmax(axis=1)[-1]
            current_outputs.append(next_word)
            if next_word == tgt_word2idx['</s>'] or len(current_outputs) >= 15:
                predictions.append([tgt_idx2word[x] for x in current_outputs])
                break
    return predictions


class NMTAttention():
    """docstring for NMTAttention"""
    def __init__(self, **kwargs):  # src_embed_dim, tgt_embed_dim, src_lstm_op_dim, tgt_lstm_op_dim, beta):
        self.src_embed_dim = kwargs['src_embed_dim']
        self.tgt_embed_dim = kwargs['tgt_embed_dim']
        self.src_lstm_op_dim = kwargs['src_lstm_op_dim']
        self tgt_lstm_op_dim = kwargs['tgt_lstm_op_dim']
        self.beta = kwargs['beta']
        self.src_word2idx = kwargs['src_word2idx']
        self.tgt_idx2word = kwargs['tgt_idx2word']
        self.n_src_vocab = len(self.src_word2idx.keys())
        self.n_tgt_vocab = len(self.tgt_idx2word.keys())

    def build_model1(self):
        # LookupTable to Embedding
        src_embedding_layer = EmbeddingLayer(input_dim=self.n_src_vocab, output_dim=self.src_embed_dim, name='src_embedding')
        tgt_embedding_layer = EmbeddingLayer(input_dim=self.n_tgt_vocab, output_dim=self.tgt_embed_dim, name='src_embedding')

        # LSTMs
        src_lstm_forward = LSTM(input_dim=self.src_embed_dim, output_dim=self.src_lstm_op_dim)
        src_lstm_backward = LSTM(input_dim=self.src_embed_dim, output_dim=self.src_lstm_op_dim)
        tgt_lstm = LSTM(input_dim=self.tgt_embed_dim, output_dim=self.tgt_lstm_op_dim)
        sys.stderr.write(str(tgt_lstm.params) + "\n")  # TODO

        # From target LSTM to target word indexes
        # Input: target LSTM output dim + Attention from BiLSTM
        proj_layer = FullyConnectedLayer(input_dim=tgt_lstm_op_dim + 2 * src_lstm_op_dim, output_dim=self.n_tgt_vocab, activation='softmax')

        params = src_embedding_layer.params + tgt_embedding_layer.params + src_lstm_forward.params + src_lstm_backward.params + tgt_lstm.params[:-1] + proj_layer.params

        # declare input variables
        src_ip = T.ivector()
        tgt_ip = T.ivector()
        tgt_op = T.ivector()

        # lookup table -> embedding
        src_embed_ip = src_embedding_layer.fprop(src_ip)
        tgt_embed_ip = tgt_embedding_layer.fprop(tgt_ip)

        # embedding -> source BiLSTM
        src_lstm_forward.fprop(src_embed_ip)
        src_lstm_backward.fprop(src_embed_ip[::-1, :])
        # Concatenate foward/backward. (Flip backward again to get corresponding h for the same word)
        encoderh = T.concatenate((src_lstm_forward.h, src_lstm_backward.h[::-1, :]), axis=1)

        # End of source BiLSTM -> target LSTM
        tgt_lstm.h_0 = encoderh[-1]
        tgt_lstm.fprop(tgt_embed_ip)

        # Attention
        # Read http://arxiv.org/abs/1508.04025
        attention = tgt_lstm.h.dot(encoderh.transpose())
        attention = attention.dot(encoderh)

        # Order preference?
        decoderh = T.concatenate((attention, tgt_lstm.h), axis=1)

        # LSTM output -> target word
        proj_op = proj_layer.fprop(decoder)

        # Cost + regularization
        cost = T.nnet.categorical_crossentropy(proj_op, tgt_op).mean()
        cost += beta * T.mean((tgt_lstm.h[:-1] ** 2 - tgt_lstm.h[1:] ** 2) ** 2)

        return dict({'cost': cost,
                     'src_ip': src_ip,
                     'tgt_ip': tgt_ip,
                     'tgt_op': tgt_op,
                     'params': params,
                     'proj_op': proj_op})

    # Attempt to do local attention
    def build_model2(self):
        # LookupTable to Embedding
        src_embedding_layer = EmbeddingLayer(input_dim=self.n_src_vocab, output_dim=self.src_embed_dim, name='src_embedding')
        tgt_embedding_layer = EmbeddingLayer(input_dim=self.n_tgt_vocab, output_dim=self.tgt_embed_dim, name='src_embedding')

        # LSTMs
        src_lstm_forward = LSTM(input_dim=self.src_embed_dim, output_dim=self.src_lstm_op_dim)
        src_lstm_backward = LSTM(input_dim=self.src_embed_dim, output_dim=self.src_lstm_op_dim)
        tgt_lstm = LSTM(input_dim=self.tgt_embed_dim, output_dim=self.tgt_lstm_op_dim)
        sys.stderr.write(str(tgt_lstm.params) + "\n")  # TODO

        # From target LSTM to target word indexes
        # Input: target LSTM output dim + Attention from BiLSTM
        proj_layer = FullyConnectedLayer(input_dim=self.tgt_lstm_op_dim + 2 * self.src_lstm_op_dim, output_dim=self.n_tgt_vocab, activation='softmax')

        params = src_embedding_layer.params + tgt_embedding_layer.params + src_lstm_forward.params + src_lstm_backward.params + tgt_lstm.params[:-1] + proj_layer.params

        # declare input variables
        src_ip = T.ivector()
        tgt_ip = T.ivector()
        tgt_op = T.ivector()

        # lookup table -> embedding
        src_embed_ip = src_embedding_layer.fprop(src_ip)
        tgt_embed_ip = tgt_embedding_layer.fprop(tgt_ip)

        # embedding -> source BiLSTM
        src_lstm_forward.fprop(src_embed_ip)
        src_lstm_backward.fprop(src_embed_ip[::-1, :])
        # Concatenate foward/backward. (Flip backward again to get corresponding h for the same word)
        encoderh = T.concatenate((src_lstm_forward.h, src_lstm_backward.h[::-1, :]), axis=1)

        # End of source BiLSTM -> target LSTM
        tgt_lstm.h_0 = encoderh[-1]
        tgt_lstm.fprop(tgt_embed_ip)

        # Attention
        # Read http://arxiv.org/abs/1508.04025
        attention = tgt_lstm.h.dot(encoderh.transpose())
        attention = attention.dot(encoderh)

        # Order preference?
        decoderh = T.concatenate((attention, tgt_lstm.h), axis=1)

        # LSTM output -> target word
        proj_op = proj_layer.fprop(decoderh)

        # Cost + regularization
        cost = T.nnet.categorical_crossentropy(proj_op, tgt_op).mean()
        cost += self.beta * T.mean((tgt_lstm.h[:-1] ** 2 - tgt_lstm.h[1:] ** 2) ** 2)

        return dict({'cost': cost,
                     'src_ip': src_ip,
                     'tgt_ip': tgt_ip,
                     'tgt_op': tgt_op,
                     'params': params,
                     'proj_op': proj_op})


def main():
    src_word2idx, src_idx2word = create_word_table(train_src)
    tgt_word2idx, tgt_idx2word = create_word_table(train_tgt)
    sys.stderr.write("Lookup table constructed." + "\n")
    NMT = NMTAttention(src_embed_dim=config.src_embed_dim, tgt_embed_dim=config.tgt_embed_dim,
                       src_lstm_op_dim=config.src_lstm_op_dim, tgt_lstm_op_dim=config.tgt_lstm_op_dim,
                       src_word2idx=src_word2idx, tgt_idx2word=tgt_idx2word, beta=config.beta)
    variables = NMT.build_model1()

    # Objective, and construct a function
    updates = Optimizer(clip=5.0).adam(cost=variables['cost'], params=variables['params'])
    f_train = theano.function(inputs=[variables['src_ip'], variables['tgt_ip'], variables['tgt_op']], outputs=variables['cost'], updates=updates)
    f_eval = theano.function(inputs=[variables['src_ip'], variables['tgt_ip']], outputs=variables['proj_layer'])

    all_costs = []
    log = open('train.log', 'w')
    n_epochs = 100

    best_valid_predictions = None
    best_valid_score = -1
    best_test_predictions = None

    for epoch in xrange(n_epochs):
        ts = time.time()
        sys.stderr.write("====== Epoch %d ======" % epoch + "\n")
        # Shuffle order
        indices = range(len(train_src))
        np.random.shuffle(indices)
        train_src_sents = [train_src[i] for i in indices]
        train_tgt_sents = [train_tgt[i] for i in indices]
        costs = []

        # For all the sentences
        for i in xrange(len(train_src_sents)):
            new_cost = f_train(
                np.array([src_word2idx[w] for w in train_src_sents[i]]).astype(np.int32),
                np.array([tgt_word2idx[w] for w in train_tgt_sents[i]][:-1]).astype(np.int32),
                np.array([tgt_word2idx[w] for w in train_tgt_sents[i]][1:]).astype(np.int32),
            )
            all_costs.append((i, new_cost))
            costs.append(new_cost)

            if i % 300 == 0:
                sys.stderr.write("%d, %f" % (i, np.mean(costs)) + "\n")
                costs = []

            if i % 10000 == 0 and i != 0:
                valid_preds = get_predictions(src_word2idx, tgt_word2idx, tgt_idx2word, f_eval, mode="validation")
                bleu = get_validation_bleu(valid_preds)
                sys.stderr.write('Epoch %d BLEU on Validation : %s\n' % (epoch, bleu))
                if float(bleu) >= best_valid_score:
                    best_valid_score = float(get_validation_bleu(valid_preds))
                    best_valid_predictions = deepcopy(valid_preds)
                    best_test_predictions = deepcopy(get_predictions(src_word2idx, tgt_word2idx, tgt_idx2word, f_eval, mode="test"))
                    sys.stderr.write('Found new best validation score %f ' % (best_valid_score) + "\n")
                log.write('Epoch %d BLEU on Validation : %s \n' % (epoch, i, bleu))

        # Compute time it takes for one epoch
        te = time.time()
        sys.stderr.write('Elapsed time for one epoch: %f\n' % (te - ts))

        # Store after epoch
        fout = open('output' + str(epoch) + '.txt', 'w')
        for line in best_test_predictions:
            fout.write(' '.join(line) + '\n')
        fout.close()

    log.close()
