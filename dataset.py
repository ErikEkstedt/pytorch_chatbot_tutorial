import random
from io import open
import itertools

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


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


# Batch magic and Tensor force_----------------------------------
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


# Dataset
# -----------------------------------------------------------------------------

class TorchDataset(Dataset):
    def __init__(self, pairs, vocab, pad_idx=0):
        self.pairs = pairs
        self.vocab = vocab
        self.pad_token = torch.LongTensor([pad_idx])

    def __len__(self):
        return len(pairs)

    def decode(self, data):
        if isinstance(data, torch.Tensor):
            data = [x.item() for x in data]
        sentence = []
        for word_idx in data:
            sentence.append(self.vocab.index2word[word_idx])
        return sentence

    def indexesFromSentence(self, sentence):
        return [self.vocab.word2index[word] for word in sentence.split(' ')]

    def __getitem__(self, idx):
        pair = pairs[idx]

        # Torchify words -> idx -> tensors
        context = self.indexesFromSentence(pair[0])
        response = self.indexesFromSentence(pair[1])
        context = torch.LongTensor(context)
        response = torch.LongTensor(response)
        return context, response


def collate_fn(data):
    '''
    Arguments:
        data:  list of tensors (context, response)
    '''
    # The data should be sorted by length.
    # Sort by length of contexts
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences.
    context_batch, response_batch = zip(*data)  # returns tuples

    # Context
    context_lengths = [len(d) for d in context_batch]
    context_padded = pad_sequence(list(context_batch), batch_first=True)

    # Alternative 1
    # sort in order to pad. keep original index and permutate back.
    # make list of tuples: (original_index, length, tensor)

    # response_info = [(i, len(d), d) for i, d in enumerate(response_batch)]
    # response_info.sort(key=lambda x: x[1], reverse=True)
    # idx, response_lengths, responses = respone_infor
    # response_padded = pad_sequence(list(responses), batch_first=True)
    # print('response_padded: ', response_padded.shape)
    # TODO:
    # permutate rows to original order. Matching responses with correct contexts

    # Alternative 2
    # use preexisting code: 
    response_lengths = [len(d) for d in response_batch]
    response_np = [x.numpy() for x in response_batch]
    response_padded_list = zeroPadding(response_np)
    response_padded = torch.LongTensor(response_padded_list)

    # I like to return dict such that the names and meaning of the data is shown
    # in the training loop.
    context = {'context_padded': context_padded,
               'context_lengths': context_lengths}
    response = {'response_padded': response_padded,
                'response_lengths': response_lengths}

    return context, response

if __name__ == "__main__":
    # load_pairs_trim_and_save(min_count=4)

    # data_maxlen_None_trim_3_pairs_196772.pt
    # data_maxlen_None_trim_4_pairs_189173.pt
    # data_maxlen_None_trim_8_pairs_165050.pt

    # Load vocabulary and pairs
    data = torch.load('data/data_maxlen_None_trim_8_pairs_165050.pt')

    pairs = data['pairs']
    vocab_dict = data['vocab_dict']
    vocab = Voc('')
    vocab.__dict__ = vocab_dict

    dset = TorchDataset(pairs, vocab)
    context, response = dset[0]
    print(dset.decode(context))
    print(dset.decode(response))

    dloader = DataLoader(dset, batch_size=16, collate_fn=collate_fn)
    for context, response in dloader:
        inputs = context['context_padded']
        inputs_length = context['context_lengths']
        outputs = response['response_padded']
        outputs_length = response['response_lengths']
        print('context ({}): {}'.format(inputs.dtype, inputs.shape))
        print('response ({}): {}'.format(inputs.dtype, inputs.shape))
        break



    # Example for validation
    small_batch_size = 5
    batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches
    print("input_variable: ", input_variable.shape)
    print("lengths: ", lengths)
    print("target_variable:", target_variable.shape)
    print("mask:", mask.shape)
    print("max_target_len:", max_target_len)



