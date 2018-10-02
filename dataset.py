from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import copy

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


# Vocabulary
# -----------------------------------------------------------------------------

class Voc:
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token

    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)


def get_pairs(datafile, delimiter='\t', max_length=None):
    ''' Read query/response pairs and return a voc object '''

    def filterPairs(pairs, max_length=10):
        ''' Filter pairs using filterPair condition '''
        def filterPair(p):
            return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length
        return [pair for pair in pairs if filterPair(pair)]

    def normalizeString(s):
        ''' Lowercase, trim, and remove non-letter characters'''
        def unicodeToAscii(s):
            ''' Turn a Unicode string to plain ASCII, thanks to
            http://stackoverflow.com/a/518232/2809427'''
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
            )
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    print("Reading from {} and extracting dialog pairs...".format(datafile))
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split(delimiter)] for l in lines]

    if max_length is not None:
        print('Filter data by max sentence lengths...')
        pairs = filterPairs(pairs, max_length)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))

    return pairs


def trim_rare_words(voc, pairs, MIN_COUNT):
    '''Trim words used under the MIN_COUNT from the voc'''
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.1f}% of total".format(len(pairs),
                                                                len(keep_pairs),
                                                                100*len(keep_pairs) / len(pairs)))
    return keep_pairs


def load_pairs_trim_and_save(min_count=4):
    '''
    Argument:
        min_count: int, Trhow out words used less than `min_count`

    min_count   : kept      : #pairs
    --------------------------------
    3           : 88.9%     : 196772
    4           : 85.5%     : 189173
    5           : 82.1%     : 181592
    6           : 79.4%     : 175622
    7           : 76.9%     : 170092
    8           : 74.6%     : 165050
    '''
    datafile = 'data/lines_of_pairs_movie_lines.txt'  # where to load data from
    corpus_name = "cornell movie-dialogs corpus"
    max_length = None  # if none all sentences are used.

    orig_pairs = get_pairs(datafile=datafile,
                           max_length=max_length)
    print("Read {!s} sentence pairs".format(len(orig_pairs)))

    pairs = copy.deepcopy(orig_pairs)
    voc = Voc(corpus_name)
    print("Adding words to Vocab...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    # Load data and create vocabulary
    pairs = trim_rare_words(voc, pairs, min_count)  # Trim voc and pairs


    filename = 'data/pairs_voc_trimmed_min{}_pairs.pt'.format(min_count)
    torch.save({'pairs':pairs, 'vocab':voc},filename)
    print('saving vocab and pairs as: ', filename)


# Dataset
# -----------------------------------------------------------------------------

# Batch magic and Tensor force
def indexesFromSentence(voc, sentence, EOS_token=2):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=0):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=0, PAD_token=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def inputVar(l, voc):
    '''Returns padded input sequence tensor and lengths'''
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len



if __name__ == "__main__":

    # min_count : pairs kept : #pairs
    # 3 : 88.9% : 196772
    # 4 : 85.5% : 189173
    # 5 : 82.1% : 181592
    # 6 : 79.4% : 175622
    # 7 : 76.9% : 170092
    # 8 : 74.6% : 165050

    # load_pairs_trim_and_save(min_count=4)


    # Load vocabulary and pairs
    min_count = 4
    data = torch.load('data/pairs_voc_trimmed_min{}_pairs.pt'.format(min_count))
    pairs = data['pairs']
    voc = data['vocab']


    # Example for validation
    small_batch_size = 5
    batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable: ", input_variable.shape)
    print("lengths: ", lengths)
    print("target_variable:", target_variable.shape)
    print("mask:", mask.shape)
    print("max_target_len:", max_target_len)
