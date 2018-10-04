import csv
import os
import codecs
import re
import unicodedata
from io import open
import torch

from dataset import Voc

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


def preprocess(corpus_path, datafile):
    """
    Writing a txt file with context-response pairs
    """
    printLines(os.path.join(corpus_path, "movie_lines.txt"))
    # PREPROCESS
    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines dict, conversations list, and field ids
    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    print("\nProcessing corpus_path...")
    lines = loadLines(os.path.join(corpus_path, "movie_lines.txt"), MOVIE_LINES_FIELDS)

    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus_path, "movie_conversations.txt"),
                                      lines, MOVIE_CONVERSATIONS_FIELDS)

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter)
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    # Print a sample of lines
    print("\nSample lines from file:")
    printLines(datafile)

# read data file and extract training pairs
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


def load_pairs_trim_and_save(datafile='data/lines_of_pairs_movie_lines.txt',
                             min_count=4,
                             max_length=None,
                             vocab_name='cornell-movie-dialogs'):
    '''
    Argument:
        datafile:  path to data file
        min_count: int, Trhow out words used less than `min_count`
        max_length: int or None, max sequence length
        vocab_name: str, name of vocabulary

    min_count   : kept      : #pairs
    --------------------------------
    3           : 88.9%     : 196772
    4           : 85.5%     : 189173
    5           : 82.1%     : 181592
    6           : 79.4%     : 175622
    7           : 76.9%     : 170092
    8           : 74.6%     : 165050
    '''
    pairs = get_pairs(datafile=datafile, max_length=max_length)
    print("Extracted {!s} sentence pairs".format(len(pairs)))

    voc = Voc(vocab_name)
    print("Adding words to Vocab...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])

    print('Pruning rare words < ', min_count)
    pairs = trim_rare_words(voc, pairs, min_count)  # Trim voc and pairs

    print('Saving data')
    filename = 'data/data_maxlen_{}_trim_{}_pairs_{}.pt'.format(max_length,
                                                                min_count,
                                                                len(pairs))
    data = {'pairs': pairs, 'vocab_dict': voc.__dict__}
    torch.save(data,filename)
    print('saving vocab and pairs as: ', filename)


if __name__ == "__main__":
    corpus_name = "cornell movie-dialogs corpus"

    print('Read corpus and save txt file with rows of context-response pairs')
    corpus_path = os.path.join("../data", corpus_name)
    output_file = os.path.join("data", "lines_of_pairs_movie_lines.txt")
    preprocess(corpus_path, output_file)

    min_count = 4
    print('Normalize pairs, trim rare words (< {}) \
          and save pairs and vocab dict'.format(min_count))

    datafile='data/lines_of_pairs_movie_lines.txt'
    min_count=8
    max_length=None
    vocab_name='movie-dialogs-min_{}-max_{}'.format(min_count, max_length)

    load_pairs_trim_and_save(datafile, min_count, max_length, vocab_name)
