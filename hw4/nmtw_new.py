# coding: utf-8

import os
import sys
import numpy as np
import theano.tensor as T
import codecs
import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu3")
from collections import Counter
import math
import copy

sys.path.append('/usr0/home/hiroakih/UltraDeep/')

from network import LSTM
from layer import HiddenLayer, EmbeddingLayer
from learning_method import LearningMethod

train_src = [line.strip().split()
             for line in codecs.open('data/train.src', 'r', encoding='utf8')]
train_tgt = [line.strip().split()
             for line in codecs.open('data/train.tgt', 'r', encoding='utf8')]
dev_src = [line.strip().split()
           for line in codecs.open('data/dev.src', 'r', encoding='utf8')]
dev_tgt = [line.strip().split()
           for line in codecs.open('data/dev.tgt', 'r', encoding='utf8')]
test_src = [line.strip().split()
            for line in codecs.open('data/test.src', 'r', encoding='utf8')]


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


def bleu_stats(hypothesis, reference):
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in xrange(1, 5):
        s_ngrams = Counter([tuple(hypothesis[i:i + n])
                            for i in xrange(len(hypothesis) + 1 - n)])
        r_ngrams = Counter([tuple(reference[i:i + n])
                            for i in xrange(len(reference) + 1 - n)])
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    if len(filter(lambda x: x == 0, stats)) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum([math.log(float(x) / y)
                         for x, y in zip(stats[2::2], stats[3::2])]) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_validation_bleu(hypotheses):
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, dev_tgt):
        stats += np.array(bleu_stats(hyp, ref))
    return "%.2f" % (100 * bleu(stats))


def main():

    source_word2idx, source_idx2word = create_word_table(train_src)
    target_word2idx, target_idx2word = create_word_table(train_tgt)
    sys.stderr.write("Lookup table constructed." + "\n")

    src_emb_dim = 256  # source word embedding dimension
    tgt_emb_dim = 256  # target word embedding dimension
    src_lstm_hid_dim = 512  # source LSTMs hidden dimension
    tgt_lstm_hid_dim = 2 * src_lstm_hid_dim  # target LSTM hidden dimension
    dropout = 0.5  # dropout rate

    n_src = len(source_word2idx)  # number of words in the source language
    n_tgt = len(target_word2idx)  # number of words in the target language

    # Parameters
    params = []

    # Source words + target words embeddings layer
    # lookup table for source words
    src_lookup = EmbeddingLayer(n_src, src_emb_dim, name='src_lookup')
    # lookup table for target words
    tgt_lookup = EmbeddingLayer(n_tgt, tgt_emb_dim, name='tgt_lookup')
    params += src_lookup.params + tgt_lookup.params

    # LSTMs
    src_lstm_for = LSTM(src_emb_dim, src_lstm_hid_dim, name='src_lstm_for', with_batch=False)
    src_lstm_rev = LSTM(src_emb_dim, src_lstm_hid_dim, name='src_lstm_rev', with_batch=False)
    tgt_lstm = LSTM(2 * tgt_emb_dim, tgt_lstm_hid_dim, name='tgt_lstm', with_batch=False)
    params += src_lstm_for.params + src_lstm_rev.params + tgt_lstm.params[:-1]

    # Projection layers
    proj_layer1 = HiddenLayer(tgt_lstm_hid_dim + 2 * src_lstm_hid_dim, n_tgt, name='proj_layer1', activation='softmax')
    proj_layer2 = HiddenLayer(2 * src_lstm_hid_dim, tgt_emb_dim, name='proj_layer2', activation='tanh')
    # proj_layer2 = HiddenLayer(2 * src_lstm_hid_dim, tgt_emb_dim, name='proj_layer2', activation='tanh')
    params += proj_layer1.params + proj_layer2.params

    beta = 500

    # Train status
    is_train = T.iscalar('is_train')
    # Input sentence
    src_sentence = T.ivector()
    # Current output translation
    tgt_sentence = T.ivector()
    # Gold translation
    tgt_gold = T.ivector()

    src_sentence_emb = src_lookup.link(src_sentence)
    tgt_sentence_emb = tgt_lookup.link(tgt_sentence)

    src_lstm_for.link(src_sentence_emb)
    src_lstm_rev.link(src_sentence_emb[::-1, :])

    src_context = T.concatenate(
        [src_lstm_for.h, src_lstm_rev.h[::-1, :]], axis=1)

    tgt_lstm.h_0 = src_context[-1]
    repeated_src_context = T.repeat(src_context[-1].dimshuffle('x', 0), tgt_sentence_emb.shape[0], axis=0)
    repeated_src_context = proj_layer2.link(repeated_src_context)

    tgt_sentence_emb = T.concatenate((tgt_sentence_emb, repeated_src_context), axis=1)
    tgt_lstm.link(tgt_sentence_emb)

    # Attention
    transition = tgt_lstm.h.dot(src_context.transpose())
    transition = transition.dot(src_context)

    transition_last = T.concatenate([transition, tgt_lstm.h], axis=1)

    prediction = proj_layer1.link(transition_last)

    cost = T.nnet.categorical_crossentropy(prediction, tgt_gold).mean()
    # Regularization of RNNs from http://arxiv.org/pdf/1511.08400v6.pdf
    cost += beta * T.mean((tgt_lstm.h[:-1] ** 2 - tgt_lstm.h[1:] ** 2) ** 2)

    updates = LearningMethod(clip=5.0).get_updates('adam', cost, params)

    f_train = theano.function(
        inputs=[src_sentence, tgt_sentence, tgt_gold],
        outputs=cost,
        updates=updates
    )

    f_eval = theano.function(
        inputs=[src_sentence, tgt_sentence],
        outputs=prediction,
    )

    best_valid_preds = None
    best_valid_score = -sys.maxint
    best_test_preds = None

    log = open('blue_valid_log.txt', 'w')
    all_costs = []
    batch_size = 50
    n_epochs = 10
    for i in xrange(n_epochs):
        print 'Starting epoch %i' % i
        indices = range(len(train_src))
        np.random.shuffle(indices)
        train_src_batch = [train_src[ind] for ind in indices]
        train_tgt_batch = [train_tgt[ind] for ind in indices]
        assert len(train_src_batch) == len(train_tgt_batch)
        costs = []
        for j in xrange(len(train_src_batch)):
            new_cost = f_train(
                np.array([source_word2idx[x] for x in train_src_batch[j]]).astype(np.int32),
                np.array([target_word2idx[x] for x in train_tgt_batch[j]][:-1]).astype(np.int32),
                np.array([target_word2idx[x] for x in train_tgt_batch[j]][1:]).astype(np.int32)
            )
            all_costs.append((j, new_cost))
            costs.append(new_cost)
            if j % 300 == 0:
                print j, np.mean(costs)
                costs = []
            if np.isnan(new_cost):
                print 'NaN detected.'
                break
            if j % 10000 == 0 and j != 0:
                valid_preds = get_predictions(
                    source_word2idx, target_word2idx, target_idx2word, f_eval, mode="validation")
                bleu = get_validation_bleu(valid_preds)
                print '==================================================================='
                print 'Epoch %i BLEU on Validation : %s ' % (i, bleu)
                print '==================================================================='
                if float(bleu) >= best_valid_score:
                    best_valid_score = float(bleu)
                    best_valid_preds = copy.deepcopy(valid_preds)
                    best_test_preds = get_predictions(
                        source_word2idx, target_word2idx, target_idx2word, f_eval, mode="test")
                    print 'Found new best validation score %f ' % (best_valid_score)
                log.write(
                    'Epoch %d Minibatch %d BLEU on Validation : %s \n' % (i, j, bleu))

        # Store after epoch
        fout = open('output' + str(i) + '.txt', 'w')
        for line in best_test_preds:
            fout.write(' '.join(line) + '\n')
        fout.close()

    log.close()

# res = f_eval(
#     np.array([source_word2ind['<s>']] + [source_word2ind[x] for x in train_src_batch[j]] + [source_word2ind['</s>']]).astype(np.int32),
#     np.array([target_word2ind['<s>']] + [target_word2ind[x] for x in train_tgt_batch[j]][:-1]).astype(np.int32),
# ).argmax(axis=1)
#
#
# # In[110]:
#
# print ' '.join([target_ind2word[x] for x in res])
#
#
# # In[111]:
#
# yy = np.array([target_word2ind['<s>']] + [target_word2ind[x] for x in train_tgt_batch[j]][:-1]).astype(np.int32)
# print ' '.join([target_ind2word[x] for x in yy])


if __name__ == '__main__':
    main()
